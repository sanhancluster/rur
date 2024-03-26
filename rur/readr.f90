module readr
    use omp_lib
    implicit none

    ! Tables, used for communication with Pyhton
    ! cell: x(3), varh(5~7)
    ! part: x(3), v(3), m, (age, metal)
    ! sink: m, x(3), v(3), j(3), dMBHoverdt, dMEdoverdt, dMsmbh, d_avgptr, c_avgptr, v_avgptr, Esave,
    ! bhspin, spinmag, eps_sink, rho_star, rho_dm, low_star, low_dm, fast_star, fast_dm
    real(kind=8),    dimension(:,:),   allocatable :: real_table

    ! cell: level, cpu
    ! part: id, level, cpu
    ! sink: id
    integer(kind=4), dimension(:,:),   allocatable :: integer_table

    ! part: long id
    integer(kind=8), dimension(:,:),   allocatable :: long_table

    ! part: family, tag
    integer(kind=1), dimension(:,:),   allocatable :: byte_table

    ! Table to count number of cells per level, cpu
    integer(kind=4), dimension(:,:),   allocatable :: ncell_table

    integer :: nhvar

    ! Some useful header informations...
    real(kind=8) :: aexp

contains
!#####################################################################
    subroutine read_cell(repo, iout, cpu_list, mode, grav, verbose, nthread)
!#####################################################################
        implicit none

        integer :: i, icpu, jcpu, ilevel, idim, jdim, igrid, jcell, ihvar, ncursor
        integer :: ncell_tot, ncpu, ndim, nlevelmax, nboundary, ngrid_a, ndim1,ngridmax

        integer :: amr_n, hydro_n, grav_n, twotondim, skip_amr, skip_hydro, skip_grav
        logical :: ok, output_particle_density
        real, dimension(1:8, 1:3) :: oct_offset
        character(LEN=10) :: ordering
        character(len=128) :: file_path

        ! temporary arrays
        integer(kind=4), dimension(:,:),   allocatable :: ngridfile
        real(kind=8),    dimension(:,:),   allocatable :: xg
        real(kind=8),    dimension(:,:),   allocatable :: gvar
        integer(kind=4), dimension(:),     allocatable :: son
        real(kind=8),    dimension(:,:,:), allocatable :: hvar
        logical,         dimension(:,:),   allocatable :: leaf

        character(len=128),    intent(in) :: repo
        integer,               intent(in) :: iout
        integer,               intent(in) :: nthread
        integer, dimension(:), intent(in) :: cpu_list
        character(len=10),     intent(in) :: mode
        logical,               intent(in) :: verbose
        logical,               intent(in) :: grav
        integer(kind=4), dimension(:),     allocatable :: cursors_temp, cursors
        real(kind=8),    dimension(:,:),   allocatable :: real_table_temp
        integer(kind=4), dimension(:,:),   allocatable :: integer_table_temp

        amr_n = 10
        hydro_n = 20
        if(grav) grav_n = 30
        ! Positional offset for 3-dimensional oct-tree
        oct_offset = reshape((/&
            -0.5,  0.5, -0.5,  0.5, -0.5,  0.5, -0.5,  0.5, &
            -0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5, &
            -0.5, -0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5  &
        /), shape(oct_offset))

        ordering='hilbert'

        file_path = amr_filename(repo, iout, cpu_list(1), mode)

        ! Step 1: Check if there is file
        inquire(file=file_path, exist=ok)
        if ( .not. ok ) then
            print *,'File not found in repo: '//file_path
            stop
        endif

        ! Step 2: Read first file for header
        open(unit=amr_n, file=amr_filename(repo, iout, cpu_list(1), mode), status='old', form='unformatted')
        read(amr_n) ncpu
        read(amr_n) ndim
        read(amr_n)
        read(amr_n) nlevelmax
        read(amr_n)
        read(amr_n) nboundary

        allocate(ngridfile(1:ncpu+nboundary, 1:nlevelmax))
        close(amr_n)

        open(unit=hydro_n, file=hydro_filename(repo, iout, cpu_list(1), mode), status='old', form='unformatted')
        read(hydro_n)
        read(hydro_n) nhvar
        close(hydro_n)
        if(grav) then
            open(unit=grav_n, file=grav_filename(repo, iout, cpu_list(1), mode), status='old', form='unformatted')
            read(grav_n)
            read(grav_n) ndim1
            close(grav_n)
            output_particle_density = .false.
            if(ndim1 == ndim+2) output_particle_density = .true.
        endif

        twotondim = 2**ndim
        skip_amr = 3 * (2**ndim + ndim) + 1
        skip_hydro = nhvar * 2**ndim
        if(grav) skip_grav = twotondim * (1 + ndim)

        ! Check total number of grids
        ncell_tot = 0
        ngridmax = 0
        allocate(cursors(1:SIZE(cpu_list)))
        allocate(cursors_temp(1:SIZE(cpu_list)))
        cursors = 0
        cursors_temp = 0
        !!! $OMP PARALLEL DO SHARED(ncell_tot, cursors_temp) &
        !!! $OMP PRIVATE(i,icpu,amr_n,igrid,ngrid_a,ngridmax,son,ngridfile,ilevel,jcpu,jdim) &
        !!! $OMP NUM_THREADS(nthread)
        do i = 1, SIZE(cpu_list)
            icpu = cpu_list(i)
            amr_n = 10!! + omp_get_thread_num()

            open(unit=amr_n, file=amr_filename(repo, iout, icpu, mode), status='old', form='unformatted')
            call skip_read(amr_n, 21)
            ! Read grid numbers
            read(amr_n) ngridfile(1:ncpu,1:nlevelmax)
            ngridmax=maxval(ngridfile)!!!
            call skip_read(amr_n, 7)
            ! For non-periodic boundary conditions (not tested!)
            if(nboundary>0) then
                call skip_read(amr_n, 3)
            endif
            allocate(son(1:ngridmax))
            do ilevel = 1, nlevelmax
                ngrid_a = ngridfile(icpu, ilevel)
                ! Loop over domains
                do jcpu = 1, nboundary + ncpu
                    if(ngridfile(jcpu, ilevel) > 0) then
                        call skip_read(amr_n, 3)
                        ! Read grid center
                        if(jcpu == icpu) then
                            call skip_read(amr_n, ndim) ! Skip position
                            read(amr_n) ! Skip father index
                            call skip_read(amr_n, 2*ndim) ! Skip nbor index
                            ! Read son index to check refinement
                            do jdim = 1, twotondim
                                read(amr_n) son(1:ngrid_a)
                                do igrid=1, ngrid_a
                                    if(son(igrid) == 0) then
                                        !!!$OMP ATOMIC
                                        ncell_tot = ncell_tot+1
                                        cursors_temp(i) = cursors_temp(i)+1
                                    end if
                                end do
                            end do
                            call skip_read(amr_n, 2*twotondim) ! Skip cpu, refinement map
                        else
                            call skip_read(amr_n, skip_amr)
                        end if
                    end if
                end do
            end do
            deallocate(son)
            close(amr_n)
        end do
        !!!$OMP END PARALLEL DO

        ncursor = 1
        do i = 1, SIZE(cpu_list)
            cursors(i) = ncursor
            ncursor = ncursor + cursors_temp(i)
        end do

        call close()
        if(grav) then
            allocate(real_table(1:ndim+nhvar+1, 1:ncell_tot))
        else
            allocate(real_table(1:ndim+nhvar, 1:ncell_tot))
        endif
        allocate(integer_table(1:2, 1:ncell_tot))

        ! icell = 1

        ! Step 3: Read the actual hydro/amr data
        ! if(verbose)write(6, '(a)', advance='no') 'Progress: '
        !$OMP PARALLEL DO &
        !$OMP SHARED(integer_table, real_table, twotondim, nhvar,grav,iout,mode,repo) &
        !$OMP PRIVATE(integer_table_temp, real_table_temp) &
        !$OMP PRIVATE(i,icpu,amr_n,hydro_n,grav_n,ngridfile,ngridmax,son,xg,hvar,gvar) &
        !$OMP PRIVATE(ilevel,ngrid_a,jcpu,idim,igrid,leaf,ihvar,jcell,jdim) &
        !$OMP NUM_THREADS(nthread)
        do i = 1, SIZE(cpu_list)
            ! !$OMP CRITICAL
            ! if(verbose)call progress_bar(i, SIZE(cpu_list))
            ! !$OMP END CRITICAL
            icpu = cpu_list(i)

            ! Open amr file
            amr_n = 10 + omp_get_thread_num()
            hydro_n = amr_n + 1 + nthread
            open(unit=amr_n, file=amr_filename(repo, iout, icpu, mode), status='old', form='unformatted')
            call skip_read(amr_n, 21)
            ! Read grid numbers
            read(amr_n) ngridfile(1:ncpu,1:nlevelmax)
            read(amr_n)

            ! For non-periodic boundary conditions (not tested!)
            if(nboundary>0) then
                call skip_read(amr_n, 2)
                read(amr_n) ngridfile(ncpu+1:ncpu+nboundary, 1:nlevelmax)
            endif
            ngridmax=maxval(ngridfile)
            allocate(son(1:ngridmax))
            allocate(xg(1:ngridmax, 1:ndim))
            allocate(hvar(1:ngridmax, 1:twotondim, 1:nhvar))
            allocate(leaf(1:ngridmax, 1:twotondim))
            if(grav) allocate(gvar(1:ngridmax, 1:twotondim))

            call skip_read(amr_n, 6)

            ! Open hydro file
            open(unit=hydro_n, file=hydro_filename(repo, iout, icpu, mode), status='old', form='unformatted')
            call skip_read(hydro_n, 6)

            if(grav) then
                ! Open grav file
                grav_n = hydro_n + 1 + nthread
                open(unit=grav_n, file=grav_filename(repo, iout, icpu, mode), status='old', form='unformatted')
                call skip_read(grav_n, 4)
            endif
            
            if(grav) then
                allocate(real_table_temp(1:ndim+nhvar+1, 1:cursors_temp(i)))
            else
                allocate(real_table_temp(1:ndim+nhvar, 1:cursors_temp(i)))
            endif
            allocate(integer_table_temp(1:2, 1:cursors_temp(i)))

            ! Loop over levels
            jcell = 1
            do ilevel = 1, nlevelmax
                ngrid_a = ngridfile(icpu, ilevel)

                ! Loop over domains
                do jcpu = 1, nboundary + ncpu

                    call skip_read(hydro_n, 2)
                    if(grav) call skip_read(grav_n, 2)

                    ! Read AMR data
                    if(ngridfile(jcpu, ilevel) > 0) then
                        call skip_read(amr_n, 3)
                        ! Read grid center
                        if(jcpu == icpu) then
                            do idim=1, ndim
                                read(amr_n) xg(1:ngrid_a, idim)
                            end do

                            read(amr_n) ! Skip father index
                            call skip_read(amr_n, 2*ndim) ! Skip nbor index
                            ! Read son index to check refinement
                            do jdim = 1, twotondim
                                read(amr_n) son(1:ngrid_a)
                                do igrid=1, ngrid_a
                                    leaf(igrid, jdim) = (son(igrid) == 0)
                                end do
                            end do
                            call skip_read(amr_n, 2*twotondim) ! Skip cpu, refinement map

                            ! Read hydro variables
                            do jdim = 1, twotondim
                                do ihvar = 1, nhvar
                                    read(hydro_n) hvar(1:ngrid_a, jdim, ihvar)
                                end do
                                if(grav) then
                                    if(output_particle_density) call skip_read(grav_n, 1)
                                    read(grav_n) gvar(1:ngrid_a, jdim)
                                    call skip_read(grav_n, ndim)
                                endif
                            end do

                            ! Merge amr & hydro data
                            do igrid=1, ngrid_a
                                do jdim = 1, twotondim
                                    if(leaf(igrid, jdim)) then
                                        do idim = 1, ndim
                                            real_table_temp(idim, jcell) = xg(igrid, idim) &
                                                    + oct_offset(jdim, idim) * 0.5**ilevel
                                        end do
                                        do ihvar = 1, nhvar
                                            real_table_temp(idim + ihvar-1, jcell) = hvar(igrid, jdim, ihvar)
                                        end do
                                        if(grav) real_table_temp(idim + nhvar, jcell) = gvar(igrid, jdim)
                                        integer_table_temp(1, jcell) = ilevel
                                        integer_table_temp(2, jcell) = icpu
                                        jcell = jcell + 1
                                    end if
                                end do
                            end do
                            ! icell = icell + jcell

                        else
                            call skip_read(amr_n, skip_amr)
                            call skip_read(hydro_n, skip_hydro)
                            if(grav) call skip_read(grav_n, skip_grav)
                            if(grav .and. output_particle_density) call skip_read(grav_n, 1)
                        end if
                    end if

                end do
            end do
            if(grav) then
                real_table(1:ndim+nhvar+1, cursors(i):cursors(i)+cursors_temp(i)-1) = &
                        real_table_temp(1:ndim+nhvar+1,1:cursors_temp(i))
            else
                real_table(1:ndim+nhvar, cursors(i):cursors(i)+cursors_temp(i)-1) = &
                        real_table_temp(1:ndim+nhvar,1:cursors_temp(i))
            endif
            integer_table(1:2, cursors(i):cursors(i)+cursors_temp(i)-1) = integer_table_temp(1:2,1:cursors_temp(i))
            close(amr_n)
            close(hydro_n)
            if(grav) close(grav_n)
            deallocate(son)
            deallocate(xg)
            deallocate(hvar)
            deallocate(leaf)
            deallocate(integer_table_temp)
            deallocate(real_table_temp)
            if(grav) deallocate(gvar)
        end do
        !$OMP END PARALLEL DO
        deallocate(ngridfile)
        deallocate(cursors)
        deallocate(cursors_temp)

    end subroutine read_cell

!#####################################################################
    subroutine read_part(repo, iout, cpu_list, mode, verbose, longint, nthread)
!#####################################################################
        implicit none

        integer :: i, j, icpu, idim
        integer :: ncpu, ndim
        integer :: npart, nstar_int, nsink, ncursor
        integer(kind=8) :: npart_tot, nstar, npart_c

        integer :: part_n, nreal, nint, nbyte, nlong, nchem, nthread
        integer :: pint
        logical :: ok

        character(len=128) :: file_path

        character(len=128),    intent(in) :: repo
        integer,               intent(in) :: iout
        integer, dimension(:), intent(in) :: cpu_list
        character(len=10),     intent(in) :: mode
        logical,               intent(in) :: verbose
        logical,               intent(in) :: longint
        integer(kind=4), dimension(:),     allocatable :: cursors_temp, cursors

        part_n = 30

        file_path = part_filename(repo, iout, cpu_list(1), mode)

        ! Step 1: Verify there is file
        inquire(file=file_path, exist=ok)
        if ( .not. ok ) then
            print *, file_path, ' not found.'
            stop
        endif

        ! Step 2: Count the total number of particles.
        open(unit=part_n, file=file_path, status='old', form='unformatted')
        read(part_n) ncpu
        read(part_n) ndim
        call skip_read(part_n, 2)
        if(longint) then
            read(part_n) nstar
        else
            read(part_n) nstar_int
            nstar = nstar_int
        end if
        call skip_read(part_n, 2)
        read(part_n) nsink
        close(part_n)

        npart_tot = 0
        allocate(cursors(1:SIZE(cpu_list)))
        allocate(cursors_temp(1:SIZE(cpu_list)))
        cursors_temp = 0
        !$OMP PARALLEL DO SHARED(npart_tot, cursors_temp) PRIVATE(i, icpu, npart, part_n) NUM_THREADS(nthread)
        do i = 1, SIZE(cpu_list)
            icpu = cpu_list(i)
            part_n = 30 + omp_get_thread_num()
            call get_npart(part_n, part_filename(repo, iout, icpu, mode), npart)
            !$OMP ATOMIC
            npart_tot = npart_tot + npart
            cursors_temp(i) = npart
        end do
        !$OMP END PARALLEL DO

        ncursor = 1
        do i = 1, SIZE(cpu_list)
            cursors(i) = ncursor
            ncursor = ncursor + cursors_temp(i)
        end do
        deallocate(cursors_temp)

        ! Set coulum spaces of each datatype for different versions of RAMSES
        if(mode == 'nh' .or. mode == 'yzics') then ! New Horizon / YZiCS / Horizon-AGN
            if(nstar > 0 .or. nsink > 0) then
                nreal = 2*ndim + 3
            else
                nreal = 2*ndim + 1
            end if
            nint = 3
            nbyte = 0
        elseif(mode == 'hagn') then
            if(nstar > 0 .or. nsink > 0) then
                nchem = 7
                nreal = 2*ndim + 3 + nchem
            else
                nreal = 2*ndim + 1
            end if
            nint = 3
            nbyte = 0
        elseif(mode == 'iap' .or. mode == 'gem' .or. mode == 'fornax' .or. mode == 'none') then ! New RAMSES version that includes family, tag
            nreal = 2*ndim + 3
            nint = 3
            nbyte = 2
        elseif(mode == 'y2') then ! New RAMSES version that includes family, tag
            nreal = 2*ndim + 12
            nint = 4
            nbyte = 2
        elseif(mode == 'y3') then ! New RAMSES version that includes family, tag, 8 chems, stellar densities
            nreal = 2*ndim + 13
            nint = 4
            nbyte = 2
            nchem = 8
        elseif(mode == 'y4' .or. mode == 'nc' .or. mode == 'nh2') then ! New RAMSES version that includes family, tag, 9 chems, stellar densities
            nreal = 2*ndim + 14
            nint = 4
            nbyte = 2
            nchem = 9
        elseif(mode == 'y5') then ! New RAMSES version that includes family, tag, 9 chems, stellar densities
            nreal = 2*ndim + 14
            nint = 3
            nbyte = 2
            nchem = 9
        end if

        if(longint) then
            nint = nint-1
            nlong = 1
        else
            nlong = 0
        end if

        call close()

        ! Allocate space for particle data
        allocate(real_table(1:nreal, 1:npart_tot))
        allocate(integer_table(1:nint, 1:npart_tot))
        allocate(byte_table(1:nbyte, 1:npart_tot))
        if(longint)allocate(long_table(1:nlong, 1:npart_tot))

        ! Step 3: Read the actual particle data
        ! Current position for particle
        ! if(verbose)write(6, '(a)', advance='no') 'Progress: '
        !$OMP PARALLEL DO &
        !$OMP SHARED(integer_table, long_table, real_table, byte_table,ndim,nstar,nsink,nchem) &
        !$OMP PRIVATE(i,j, icpu,npart,part_n,npart_c,idim,pint) &
        !$OMP NUM_THREADS(nthread)
        do i = 1, SIZE(cpu_list)
            ! !$OMP CRITICAL
            ! if(verbose)call progress_bar(i, SIZE(cpu_list))
            ! !$OMP END CRITICAL
            icpu = cpu_list(i)
            part_n = 30 + omp_get_thread_num()
            npart_c = cursors(i)
            open(unit=part_n, file=part_filename(repo, iout, icpu, mode), status='old', form='unformatted')
            ! Skip headers
            call skip_read(part_n, 2)
            read(part_n) npart
            call skip_read(part_n, 5)

            ! Read position(3), velocity(3), mass
            do idim = 1, 2*ndim+1
                read(part_n) real_table(idim, npart_c:npart_c+npart-1)
            end do

            ! Read id
            pint=1
            if(longint) then
                read(part_n) long_table(1, npart_c:npart_c+npart-1)
            else
                read(part_n) integer_table(pint, npart_c:npart_c+npart-1)
                pint = pint+1
            end if

            ! Read level
            read(part_n) integer_table(pint, npart_c:npart_c+npart-1)
            pint = pint+1
            if(mode == 'nh' .or. mode == 'yzics' .or. mode == 'hagn') then
                ! If star or sink particles are activated, RAMSES adds epoch, metallicity information for particles.
                if(nstar > 0 .or. nsink > 0) then
                    read(part_n) real_table(2*ndim+2, npart_c:npart_c+npart-1)
                    read(part_n) real_table(2*ndim+3, npart_c:npart_c+npart-1)
                    if(mode=='hagn') then
                        do j=1,nchem
                            read(part_n) real_table(2*ndim+3+j, npart_c:npart_c+npart-1)
                        end do
                    endif
                end if
                ! Add CPU information
                integer_table(pint, npart_c:npart_c+npart-1) = icpu
            elseif(mode == 'iap' .or. mode == 'gem' .or. mode == 'none' .or. mode == 'fornax' &
                & .or. mode == 'y2' .or. mode == 'y3' .or. mode == 'y4' .or. mode == 'nc' .or. mode=='nh2' .or. mode=='y5') then
                ! family, tag
                read(part_n) byte_table(1, npart_c:npart_c+npart-1)
                read(part_n) byte_table(2, npart_c:npart_c+npart-1)

                ! If star or sink particles are activated, RAMSES adds epoch, metallicity information for particles.
                if(nstar > 0 .or. nsink > 0) then
                    read(part_n) real_table(2*ndim+2, npart_c:npart_c+npart-1)
                    read(part_n) real_table(2*ndim+3, npart_c:npart_c+npart-1)
                    if(mode == 'y2' .or. mode == 'y3' .or. mode == 'y4' .or. mode=='nc' .or. mode=='nh2' .or. mode=='y5') then
                        ! Initial mass
                        read(part_n) real_table(2*ndim+4, npart_c:npart_c+npart-1)
                        ! Chemical elements
                        do j=1,nchem
                            read(part_n) real_table(2*ndim+4+j, npart_c:npart_c+npart-1)
                        end do
                    end if
                    if(mode == 'y3' .or. mode == 'y4' .or. mode=='nc' .or. mode=='nh2' .or. mode=='y5') then
                        ! Stellar densities at formation
                        read(part_n) real_table(2*ndim+nchem+5, npart_c:npart_c+npart-1)
                    end if
                else
                    real_table(2*ndim+2:2*ndim+3, npart_c:npart_c+npart-1) = 0d0
                end if
                if(mode == 'y2' .or. mode == 'y3' .or. mode == 'y4' .or. mode=='nc' .or. mode=='nh2') then
                    ! Parent indices
                    read(part_n) integer_table(pint+1, npart_c:npart_c+npart-1)
                end if

                ! Add CPU information
                integer_table(pint, npart_c:npart_c+npart-1) = icpu
            end if
            close(part_n)
        end do
        !$OMP END PARALLEL DO
        deallocate(cursors)
    end subroutine read_part

!#####################################################################
    subroutine read_sinkprop(repo, iprop, drag_part, mode)
!#####################################################################
        implicit none

        integer :: nsink, ndim, i, ireal, iint
        integer :: sink_n, nreal, nint

        character(len=128),    intent(in) :: repo
        integer,               intent(in) :: iprop
        logical,               intent(in) :: drag_part
        character(len=10),     intent(in) :: mode

        ! possible sinkprop data formats
        ! default: 1 integer, 22 real
        ! fornax: 1 integer, 23 real
        ! nh2: 1 integer, 26 real
        ! +drag part: +2 integer, +12 real

        sink_n = 40
        call close()

        open(unit=sink_n, file=sinkprop_filename(repo, iprop), status='old', form='unformatted')
        read(sink_n) nsink
        read(sink_n) ndim
        read(sink_n) aexp
        call skip_read(sink_n, 3)

        if(drag_part) then
            nreal = 34
            nint = 3
        else
            nreal = 22
            nint = 1
        end if
        if(mode == 'fornax') nreal = nreal + 1
        if(mode == 'y2' .or. mode == 'y3' .or. mode == 'y4' .or. mode == 'nc' .or. mode=='y5') nreal = nreal + 4
        allocate(real_table(1:nreal, 1:nsink))
        allocate(integer_table(1:nint, 1:nsink))

        iint = 1
        ireal = 1

        read(sink_n) integer_table(iint, :)
        iint = iint + 1
        do i=1,22
            read(sink_n) real_table(ireal, :)
            ireal = ireal + 1
        end do
        if(mode == 'fornax' .or. mode == 'y2' .or. mode == 'y3' .or. mode == 'y4' .or. mode == 'nc' .or. mode=='y5') then
            read(sink_n) real_table(ireal, :)
            ireal = ireal + 1
        end if
        if(drag_part) then
            do i=1,8
                read(sink_n) real_table(ireal, :)
                ireal = ireal + 1
            end do
            do i=1,2
                read(sink_n) integer_table(iint, :)
                iint = iint + 1
            end do
            do i=1,4
                read(sink_n) real_table(ireal, :)
                ireal = ireal + 1
            end do
        end if
        if(mode == 'y2' .or. mode == 'y3' .or. mode == 'y4' .or. mode == 'nc' .or. mode=='y5') then
            do i=1,3
                read(sink_n) real_table(ireal, :)
                ireal = ireal + 1
            end do
        end if
        close(sink_n)

    end subroutine read_sinkprop

!#####################################################################
    subroutine read_sink(repo, iout, icpu, levelmin, nlevelmax)
!#####################################################################
        implicit none

        integer :: nsink, nindsink, ndim, i
        integer :: sink_n, nreal, nint, nstat

        character(len=128),    intent(in) :: repo
        integer,               intent(in) :: iout, icpu, levelmin, nlevelmax

        ndim = 3
        sink_n = 50
        call close()

        open(unit=sink_n, file=sink_filename(repo, iout, icpu), form='unformatted')
        rewind(sink_n)
        read(sink_n) nsink
        read(sink_n) nindsink

        nstat = (ndim*2+1)*(nlevelmax-levelmin+1)
        nreal = 20+nstat
        nint = 1

        allocate(real_table(1:nreal, 1:nsink))
        allocate(integer_table(1:nint, 1:nsink))

        if(nsink>0) then
            read(sink_n) integer_table(1, 1:nsink) ! id
            do i=1,20
                read(sink_n) real_table(i, :) ! mass, pos*3, vel*3, t, dM*3, Esave, j*3, spin*3, spinmag, eps
            end do
            do i=21,21+nstat-1
                read(sink_n) real_table(i, :) ! stats
            end do
        endif
        close(sink_n)

    end subroutine read_sink
!#####################################################################
    subroutine count_cell(repo, iout, cpu_list, mode)
!#####################################################################
        ! Counts the number of leaf cells per level
        implicit none

        integer :: i, icpu, jcpu, ilevel, jdim, igrid
        integer :: ncell_loc, ncpu, ndim, nlevelmax, nboundary, ngrid_a, ngridmax

        integer :: amr_n, hydro_n, twotondim, skip_amr
        logical :: ok
        character(len=128) :: file_path

        ! temporary arrays
        integer(kind=4), dimension(:,:),   allocatable :: ngridfile
        integer(kind=4), dimension(:),     allocatable :: son

        character(len=128),    intent(in) :: repo
        integer,               intent(in) :: iout
        integer, dimension(:), intent(in) :: cpu_list
        character(len=10),     intent(in) :: mode

        amr_n = 10
        hydro_n = 20
        ! Positional offset for 3-dimensional oct-tree

        file_path = amr_filename(repo, iout, cpu_list(1), mode)

        ! Step 1: Check if there is file
        inquire(file=file_path, exist=ok)
        if ( .not. ok ) then
            print *,'File not found in repo: '//file_path
            stop
        endif

        ! Step 2: Read first file for header
        open(unit=amr_n, file=amr_filename(repo, iout, cpu_list(1), mode), status='old', form='unformatted')
        read(amr_n) ncpu
        read(amr_n) ndim
        read(amr_n)
        read(amr_n) nlevelmax
        read(amr_n)
        read(amr_n) nboundary

        allocate(ngridfile(1:ncpu+nboundary, 1:nlevelmax))
        close(amr_n)

        if(allocated(ncell_table)) deallocate(ncell_table)
        allocate(ncell_table(1:ncpu+nboundary, 1:nlevelmax))
        ncell_table(:,:) = 0

        twotondim = 2**ndim
        skip_amr = 3 * (2**ndim + ndim) + 1

        ! Check total number of grids
        ngridmax = 0
        do i = 1, SIZE(cpu_list)
            icpu = cpu_list(i)
            amr_n = 10 + omp_get_thread_num()

            open(unit=amr_n, file=amr_filename(repo, iout, icpu, mode), status='old', form='unformatted')
            call skip_read(amr_n, 21)
            ! Read grid numbers
            read(amr_n) ngridfile(1:ncpu,1:nlevelmax)
            ngridmax=maxval(ngridfile)!!!
            call skip_read(amr_n, 7)
            ! For non-periodic boundary conditions (not tested!)
            if(nboundary>0) then
                call skip_read(amr_n, 3)
            endif

            allocate(son(1:ngridmax))
            do ilevel = 1, nlevelmax
                ncell_loc = 0
                ngrid_a = ngridfile(icpu, ilevel)
                ! Loop over domains
                do jcpu = 1, nboundary + ncpu
                    if(ngridfile(jcpu, ilevel) > 0) then
                        call skip_read(amr_n, 3)
                        ! Read grid center
                        if(jcpu == icpu) then
                            call skip_read(amr_n, ndim) ! Skip position
                            read(amr_n) ! Skip father index
                            call skip_read(amr_n, 2*ndim) ! Skip nbor index
                            ! Read son index to check refinement
                            do jdim = 1, twotondim
                                read(amr_n) son(1:ngrid_a)
                                do igrid=1, ngrid_a
                                    if(son(igrid) == 0) ncell_loc = ncell_loc+1
                                end do
                            end do
                            call skip_read(amr_n, 2*twotondim) ! Skip cpu, refinement map
                        else
                            call skip_read(amr_n, skip_amr)
                        end if
                    end if
                end do
                ncell_table(icpu, ilevel) = ncell_table(icpu, ilevel) + ncell_loc
            end do

            deallocate(son)
            close(amr_n)
        end do
    end subroutine count_cell
!#####################################################################
    subroutine close()
!#####################################################################
        ! Clean old data table is used more then once
        implicit none
        if(allocated(real_table)) deallocate(real_table)
        if(allocated(integer_table)) deallocate(integer_table)
        if(allocated(byte_table)) deallocate(byte_table)
        if(allocated(long_table)) deallocate(long_table)
    end subroutine close

!#####################################################################
    subroutine skip_read(unit,nskip)
!#####################################################################
        ! skip the given number of reads

        implicit none
        integer,intent(in) :: unit, nskip
        integer :: i
        do i=1,nskip
            read(unit)
        end do
    end subroutine skip_read

!#####################################################################
    character(len=5) function charind(iout)
!#####################################################################
        implicit none
        integer, intent(in) :: iout

        write(charind, '(I0.5)') iout

    end function charind

!#####################################################################
    character(len=128) function amr_filename(repo, iout, icpu, mode)
!#####################################################################
        implicit none
        character(len=128), intent(in)  :: repo
        integer,            intent(in)  :: iout, icpu
        character(len=10),  intent(in)  :: mode

        if(mode == 'ng') then
            amr_filename = TRIM(repo)//'/output_'//charind(iout)//'/amr.out'//charind(icpu)
        else
            amr_filename = TRIM(repo)//'/output_'//charind(iout)//'/amr_'//charind(iout)//'.out'//charind(icpu)
        end if
    end function amr_filename

!#####################################################################
    character(len=128) function hydro_filename(repo, iout, icpu, mode)
!#####################################################################
        implicit none
        character(len=128), intent(in)  :: repo
        integer,            intent(in)  :: iout, icpu
        character(len=10),  intent(in)  :: mode

        if(mode == 'ng') then
            hydro_filename = TRIM(repo)//'/output_'//charind(iout)//'/hydro.out'//charind(icpu)
        else
            hydro_filename = TRIM(repo)//'/output_'//charind(iout)//'/hydro_'//charind(iout)//'.out'//charind(icpu)
        end if
    end function hydro_filename

!#####################################################################
    character(len=128) function grav_filename(repo, iout, icpu, mode)
!#####################################################################
        implicit none
        character(len=128), intent(in)  :: repo
        integer,            intent(in)  :: iout, icpu
        character(len=10),  intent(in)  :: mode

        if(mode == 'ng') then
            grav_filename = TRIM(repo)//'/output_'//charind(iout)//'/grav.out'//charind(icpu)
        else
            grav_filename = TRIM(repo)//'/output_'//charind(iout)//'/grav_'//charind(iout)//'.out'//charind(icpu)
        end if
    end function grav_filename

!#####################################################################
    character(len=128) function part_filename(repo, iout, icpu, mode)
!#####################################################################
        implicit none
        character(len=128), intent(in)  :: repo
        integer,            intent(in)  :: iout, icpu
        character(len=10),  intent(in)  :: mode

        if(mode == 'ng') then
            part_filename = TRIM(repo)//'/output_'//charind(iout)//'/part.out'//charind(icpu)
        else
            part_filename = TRIM(repo)//'/output_'//charind(iout)//'/part_'//charind(iout)//'.out'//charind(icpu)
        end if
    end function part_filename

!#####################################################################
    character(len=128) function sinkprop_filename(repo, iout)
!#####################################################################
        implicit none
        character(len=128), intent(in)  :: repo
        integer,            intent(in)  :: iout

        sinkprop_filename = TRIM(repo)//'/sink_'//charind(iout)//'.dat'
    end function sinkprop_filename

!#####################################################################
    character(len=128) function sink_filename(repo, iout, icpu)
!#####################################################################
        implicit none
        character(len=128), intent(in)  :: repo
        integer,            intent(in)  :: iout, icpu

        sink_filename = TRIM(repo)//'/output_'//charind(iout)//'/sink_'//charind(iout)//'.out'//charind(icpu)
    end function sink_filename

!#####################################################################
    subroutine progress_bar(iteration, maximum)
!#####################################################################
        implicit none
        integer :: iteration, maximum
        if(iteration == maximum) then
            write(6, '(a)') '# - Done!'
        elseif(MOD(iteration, MAX(1, maximum/50)) == 0) then
            write(6, '(a)', advance='no') '#'
        end if
    end subroutine progress_bar

!#####################################################################
    subroutine get_npart(part_n, file_path, npart)
!#####################################################################
        ! Suggested by ChatGPT
        implicit none
        integer, intent(in) :: part_n
        character(len=*), intent(in) :: file_path
        integer, intent(out) :: npart
        open(unit=part_n, file=file_path, status='old', form='unformatted')
        call skip_read(part_n, 2)
        read(part_n) npart
        close(part_n)
        return
    end subroutine get_npart


end module readr
