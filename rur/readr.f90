module readr
    implicit none

    ! Tables, used for communication with Pyhton
    ! cell: x(3), varh(5~7)
    ! part: x(3), v(3), m, (age, metal)
    real(kind=8),    dimension(:,:),   allocatable::real_table

    ! cell: level, cpu
    ! part: id, level, cpu
    integer(kind=4), dimension(:,:),   allocatable::integer_table

    ! part: long id
    integer(kind=8), dimension(:,:),   allocatable::long_table

    ! part: family, tag
    integer(kind=1), dimension(:,:),   allocatable::byte_table

    integer :: nhvar

contains
!#####################################################################
    subroutine read_cell(repo, iout, cpu_list, mode, verbose)
!#####################################################################
        implicit none

        integer :: i, j, icpu, jcpu, ilevel, idim, jdim, igrid, icell, jcell, ihvar
        integer :: ncell_tot, ncpu, ndim, nlevelmax, nboundary, ngrid_a

        integer :: amr_n, hydro_n, twotondim, skip_amr, skip_hydro
        logical :: ok, is_leaf
        real, dimension(1:8, 1:3) :: oct_offset
        character(LEN=10) :: ordering
        character(len=128) :: file_path

        ! temporary arrays
        integer(kind=4), dimension(:,:),   allocatable :: ngridfile
        real(kind=8),    dimension(:,:),   allocatable :: xg
        integer(kind=4), dimension(:),     allocatable :: son
        real(kind=8),    dimension(:,:,:), allocatable :: hvar
        logical,         dimension(:,:),   allocatable :: leaf

        character(len=128),    intent(in) :: repo
        integer,               intent(in) :: iout
        integer, dimension(:), intent(in) :: cpu_list
        character(len=10),     intent(in) :: mode
        logical,               intent(in) :: verbose

        amr_n = 10
        hydro_n = 20
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

        twotondim = 2**ndim
        skip_amr = 3 * (2**ndim + ndim) + 1
        skip_hydro = nhvar * 2**ndim

        ! Check total number of grids
        ncell_tot = 0
        do i = 1, SIZE(cpu_list)
            icpu = cpu_list(i)

            open(unit=amr_n, file=amr_filename(repo, iout, icpu, mode), status='old', form='unformatted')
            call skip_read(amr_n, 21)
            ! Read grid numbers
            read(amr_n) ngridfile(1:ncpu,1:nlevelmax)
            call skip_read(amr_n, 7)
            ! For non-periodic boundary conditions (not tested!)
            if(nboundary>0) then
                call skip_read(amr_n, 3)
            endif
            do ilevel = 1, nlevelmax
                ngrid_a = ngridfile(icpu, ilevel)
                ! Loop over domains
                if(ngrid_a > 0) then
                    allocate(son(1:ngrid_a))
                end if
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
                                read(amr_n) son
                                do igrid=1, ngrid_a
                                    if(son(igrid) == 0) ncell_tot = ncell_tot+1
                                end do
                            end do
                            call skip_read(amr_n, 2*twotondim) ! Skip cpu, refinement map
                        else
                            call skip_read(amr_n, skip_amr)
                        end if
                    end if
                end do
                if(ngrid_a>0) then
                    deallocate(son)
                end if
            end do
            close(amr_n)
        end do

        call close()

        allocate(real_table(1:ncell_tot, 1:ndim+nhvar))
        allocate(integer_table(1:ncell_tot, 1:2))

        icell = 1

        ! Step 3: Read the actual hydro/amr data
        if(verbose)write(6, '(a)', advance='no') 'Progress: '
        do i = 1, SIZE(cpu_list)
            if(verbose)call progress_bar(i, SIZE(cpu_list))
            icpu = cpu_list(i)

            ! Open amr file
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

            call skip_read(amr_n, 6)

            ! Open hydro file
            open(unit=hydro_n, file=hydro_filename(repo, iout, icpu, mode), status='old', form='unformatted')
            call skip_read(hydro_n, 6)

            ! Loop over levels
            do ilevel = 1, nlevelmax
                ngrid_a = ngridfile(icpu, ilevel)

                if(ngrid_a > 0) then
                    allocate(son(1:ngrid_a))
                    allocate(xg(1:ngrid_a, 1:ndim))
                    allocate(hvar(1:ngrid_a, 1:twotondim, 1:nhvar))
                    allocate(leaf(1:ngrid_a, 1:twotondim))
                endif
                ! Loop over domains
                do jcpu = 1, nboundary + ncpu

                    call skip_read(hydro_n, 2)

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
                                read(amr_n) son
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
                            end do

                            ! Merge amr & hydro data
                            jcell = 0
                            do igrid=1, ngrid_a
                                do jdim = 1, twotondim
                                    if(leaf(igrid, jdim)) then
                                        do idim = 1, ndim
                                            real_table(icell + jcell, idim) = xg(igrid, idim) &
                                                    + oct_offset(jdim, idim) * 0.5**ilevel
                                        end do
                                        do ihvar = 1, nhvar
                                            real_table(icell + jcell, idim + ihvar-1) = hvar(igrid, jdim, ihvar)
                                        end do

                                        integer_table(icell + jcell, 1) = ilevel
                                        integer_table(icell + jcell, 2) = icpu
                                        jcell = jcell + 1
                                    end if
                                end do
                            end do
                            icell = icell + jcell

                        else
                            call skip_read(amr_n, skip_amr)
                            call skip_read(hydro_n, skip_hydro)
                        end if
                    end if

                end do
                if(ngrid_a>0) then
                    deallocate(son)
                    deallocate(xg)
                    deallocate(hvar)
                    deallocate(leaf)
                endif
            end do
            close(amr_n)
            close(hydro_n)
        end do
        deallocate(ngridfile)

    end subroutine read_cell

!#####################################################################
    subroutine read_part(repo, iout, cpu_list, mode, verbose, longint)
!#####################################################################
        implicit none

        integer :: i, j, icpu, idim, ipart
        integer :: ncpu, ndim
        integer :: npart, nstar_int, nsink
        integer(kind=8) :: npart_tot, nstar, npart_c

        integer :: part_n, nreal, nint, nbyte, nlong
        integer :: pint
        logical :: ok

        character(len=128) :: file_path

        character(len=128),    intent(in) :: repo
        integer,               intent(in) :: iout
        integer, dimension(:), intent(in) :: cpu_list
        character(len=10),     intent(in) :: mode
        logical,               intent(in) :: verbose
        logical,               intent(in) :: longint

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
        do i = 1, SIZE(cpu_list)
            icpu = cpu_list(i)

            open(unit=part_n, file=part_filename(repo, iout, icpu, mode), status='old', form='unformatted')
            call skip_read(part_n, 2)
            read(part_n) npart
            close(part_n)
            npart_tot = npart_tot + npart
        end do

        ! Set coulum spaces of each datatype for different versions of RAMSES
        if(mode == 'nh' .or. mode == 'yzics') then ! New Horizon / YZiCS / Horizon-AGN
            if(nstar > 0 .or. nsink > 0) then
                nreal = 2*ndim + 3
            else
                nreal = 2*ndim + 1
            end if
            nint = 3
            nbyte = 0
        elseif(mode == 'iap' .or. mode == 'gem' .or. mode == 'none') then ! New RAMSES version that includes family, tag
            nreal = 2*ndim + 3
            nint = 3
            nbyte = 2
        end if

        if(longint) then
            nint = nint-1
            nlong = 1
        else
            nlong = 0
        end if

        call close()

        ! Allocate space for particle data
        allocate(real_table(1:npart_tot, 1:nreal))
        allocate(integer_table(1:npart_tot, 1:nint))
        allocate(byte_table(1:npart_tot, 1:nbyte))
        if(longint)allocate(long_table(1:npart_tot, 1:nlong))

        ! Step 3: Read the actual particle data
        ! Current position for particle
        npart_c = 1

        if(verbose)write(6, '(a)', advance='no') 'Progress: '
        do i = 1, SIZE(cpu_list)
            if(verbose)call progress_bar(i, SIZE(cpu_list))
            icpu = cpu_list(i)

            open(unit=part_n, file=part_filename(repo, iout, icpu, mode), status='old', form='unformatted')
            ! Skip headers
            call skip_read(part_n, 2)
            read(part_n) npart
            call skip_read(part_n, 5)

            ! Read position(3), velocity(3), mass
            do idim = 1, 2*ndim+1
                read(part_n) real_table(npart_c:npart_c+npart-1, idim)
            end do

            ! Read id
            pint=1
            if(longint) then
                read(part_n) long_table(npart_c:npart_c+npart-1, 1)
            else
                read(part_n) integer_table(npart_c:npart_c+npart-1, pint)
                pint = pint+1
            end if
            ! Read level
            read(part_n) integer_table(npart_c:npart_c+npart-1, pint)
            pint = pint+1
            if(mode == 'nh' .or. mode == 'yzics') then
                ! If star or sink particles are activated, RAMSES adds epoch, metallicity information for particles.
                if(nstar > 0 .or. nsink > 0) then
                    read(part_n) real_table(npart_c:npart_c+npart-1, 2*ndim+2)
                    read(part_n) real_table(npart_c:npart_c+npart-1, 2*ndim+3)
                end if

                ! Add CPU information
                integer_table(npart_c:npart_c+npart-1, pint) = icpu

            elseif(mode == 'iap' .or. mode == 'gem' .or. mode == 'none') then
                ! family, tag
                read(part_n) byte_table(npart_c:npart_c+npart-1, 1)
                read(part_n) byte_table(npart_c:npart_c+npart-1, 2)

                ! If star or sink particles are activated, RAMSES adds epoch, metallicity information for particles.
                if(nstar > 0 .or. nsink > 0) then
                    read(part_n) real_table(npart_c:npart_c+npart-1, 2*ndim+2)
                    read(part_n) real_table(npart_c:npart_c+npart-1, 2*ndim+3)
                else
                    real_table(npart_c:npart_c+npart-1, 2*ndim+2:2*ndim+3) = 0d0
                end if

                ! Add CPU information
                integer_table(npart_c:npart_c+npart-1, pint) = icpu
            end if
            npart_c = npart_c + npart
            close(part_n)
        end do
    end subroutine read_part

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
end module readr

