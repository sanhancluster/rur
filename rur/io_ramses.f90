! Created by San Han on 20. 2. 14..

module io_ramses

   ! Header data (AMR)
   integer :: ncpu, ndim, nx, ny, nz, nlevelmax, ngridmax, nboundary, ngrid_current, noutput, iout1, ifout
   integer :: nstep, nstep_coarse
   real(kind=8), dimension(:), allocatable :: tout, aout, dtold, dtnew
   real(kind=8) :: boxlen, t, einit, mass_tot_0, rho_tot
   real(kind=8) :: omega_m, omega_l, omega_k, omega_b, h0, aexp_ini, boxlen_ini
   real(kind=8) :: aexp, hexp, aexp_old, epot_tot_int, epot_tot_old, mass_sph
   integer, dimension(:, :), allocatable :: headl, taill, numbl, numbtot
   integer :: headf, tailf, numbf, used_mem, used_mem_tot
   character(len=128) :: ordering

   real(kind=16), dimension(:), allocatable :: bound_keyl
   real(kind=8), dimension(:), allocatable :: bound_key

   ! Header data (Added from Hydro)
   integer :: nvar
   real(kind=8) :: gamma

   ! Additional parameters
   integer :: twotondim, skip_amr, skip_hydro
   real(kind=8) :: half = 0.5

   ! Data containers
   integer :: ncell_tot, ngrid_tot, ngrid_act, ngrid_rec, nleaf
   real(kind=8), dimension(:, :), allocatable :: xc
   real(kind=8), dimension(:, :), allocatable :: uc
   integer, dimension(:), allocatable :: lvlc
   integer, dimension(:), allocatable :: cpuc

   real(kind=8), dimension(:, :), allocatable :: xg
   real(kind=8), dimension(:, :, :), allocatable :: ug
   logical, dimension(:, :), allocatable :: refg
   integer, dimension(:), allocatable :: lvlg

   real(kind=8), dimension(:, :), allocatable :: xp, vp
   integer, dimension(:), allocatable :: idp
   integer(kind=8), dimension(:), allocatable :: idpl
   byte, dimension(:), allocatable :: fam, tag
   real(kind=8), dimension(:), allocatable :: mp, tp, zp

   logical :: quadhilbert = .false.
   logical :: star = .false.
   logical :: family = .true.
   logical :: metal = .false.
   logical :: longint = .false.

contains
!#####################################################################
   subroutine read_grid_header(repo, iout, icpu)
!#####################################################################
      implicit none
      character(len=128), intent(in) :: repo
      integer, intent(in) :: iout, icpu

      integer :: amr_n, hydro_n
      character(len=128) :: file_path
      logical :: ok

      amr_n = 10
      hydro_n = 20
      ok = .true.

      file_path = amr_filename(repo, iout, icpu)

      ! Step 1: Check if there is file
      inquire(file=file_path, exist=ok)
      if ( .not. ok ) then
         print *,'File not found in repo: '//file_path
         stop
      endif

      ! Step 2: Read header
      open(unit=amr_n, file=file_path, status='old', form='unformatted')
      read(amr_n) ncpu
      read(amr_n) ndim
      read(amr_n) nx, ny, nz
      read(amr_n) nlevelmax
      read(amr_n) ngridmax
      read(amr_n) nboundary
      read(amr_n) ngrid_current
      read(amr_n) boxlen
      read(amr_n) noutput, iout1, ifout

      if(nboundary>0) then
         write(*,*) 'Sorry, current version does not support nboundary > 0.'
         stop
      end if

      allocate(tout(1:noutput), aout(1:noutput))
      read(amr_n) tout
      read(amr_n) aout
      read(amr_n) t

      allocate(dtold(1:nlevelmax), dtnew(1:nlevelmax))
      read(amr_n) dtold
      read(amr_n) dtnew

      read(amr_n) nstep, nstep_coarse
      read(amr_n) einit, mass_tot_0, rho_tot
      read(amr_n) omega_m, omega_l, omega_k, omega_b, h0, aexp_ini, boxlen_ini
      read(amr_n) aexp, hexp, aexp_old, epot_tot_int, epot_tot_old
      read(amr_n) mass_sph

      allocate(headl(1:ncpu, 1:nlevelmax), taill(1:ncpu, 1:nlevelmax), numbl(1:ncpu, 1:nlevelmax), numbtot(1:10, 1:nlevelmax))
      read(amr_n) headl
      read(amr_n) taill
      read(amr_n) numbl
      read(amr_n) numbtot

      read(amr_n) headf, tailf, numbf, used_mem, used_mem_tot
      read(amr_n) ordering

      if(ordering /= 'hilbert') then
         write(*,*) 'Sorry, current version does not support non-Hilbert ordering.'
         stop
      end if

      if(quadhilbert) then
         allocate(bound_keyl(0:ncpu))
         read(amr_n) bound_keyl(0:ncpu)
      else
         allocate(bound_key(0:ncpu))
         read(amr_n) bound_key(0:ncpu)
      end if

      ! Skip coarse data
      call skip_read(amr_n, 3)

      close(amr_n)

      file_path = hydro_filename(repo, iout, icpu)

      ! Step 1: Check if there is file
      inquire(file=file_path, exist=ok)
      if ( .not. ok ) then
         print *,'File not found in repo: '//file_path
         stop
      endif

      open(unit=hydro_n, file=file_path, status='old', form='unformatted')
      read(hydro_n)
      read(hydro_n) nvar
      call skip_read(hydro_n, 3)
      read(hydro_n) gamma

      close(hydro_n)

      twotondim = 2**ndim
      skip_amr = 3 * (2**ndim + ndim) + 4
      skip_hydro = nvar * 2**ndim

   end subroutine read_grid_header

!#####################################################################
   subroutine read_cell_single(repo, iout, icpu)
!#####################################################################
      implicit none
      character(len=128), intent(in) :: repo
      integer, intent(in) :: iout, icpu

      integer :: amr_n, hydro_n
      integer :: ilevel, jcpu, igrid, icell, jcell, ivar, idim, ind
      integer :: ncache, ncache_max

      integer, dimension(1:ncpu, 1:nlevelmax) :: numbl2

      integer,      dimension(:),     allocatable :: son
      real(kind=8), dimension(:,:),   allocatable :: xgt
      real(kind=8), dimension(:,:,:), allocatable :: hvar
      logical,      dimension(:,:),   allocatable :: leaf

      real(kind=8), dimension(1:8, 1:3) :: oct_offset

      amr_n = 10
      hydro_n = 20

      ! Positional offset for 3-dimensional oct-tree
      oct_offset = reshape((/&
          -0.5,  0.5, -0.5,  0.5, -0.5,  0.5, -0.5,  0.5, &
          -0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5, &
          -0.5, -0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5  &
      /), shape(oct_offset))

      ! Step 1: Count the total number of grids
      ncell_tot = 0
      ncache = 0
      open(unit=amr_n, file=amr_filename(repo, iout, icpu), status='old', form='unformatted')
         call skip_read(amr_n, 21)
         ! Read grid numbers
         read(amr_n) numbl2 ! Read ngridmap for safety
         call skip_read(amr_n, 7)

      do ilevel = 1, nlevelmax
         ncache_max = MAX(ncache_max, numbl2(icpu, ilevel))
         ncache = numbl2(icpu, ilevel)
         ! Loop over domains
         if(ncache > 0) then
            allocate(son(1:ncache))
         end if
         do jcpu = 1, nboundary + ncpu
            if(numbl2(jcpu, ilevel) > 0) then
               if(jcpu == icpu) then
                  call skip_read(amr_n, 3) ! Skip grid index, next, prev

                  ! Read grid center
                  call skip_read(amr_n, ndim) ! Skip position
                  read(amr_n) ! Skip father index
                  call skip_read(amr_n, 2*ndim) ! Skip nbor index
                  ! Read son index to check refinement
                  do ind = 1, twotondim
                     read(amr_n) son
                     do igrid=1, ncache
                        if(son(igrid) == 0) ncell_tot = ncell_tot+1
                     end do
                  end do
                  call skip_read(amr_n, 2*twotondim) ! Skip cpu, refinement map
               else
                  call skip_read(amr_n, skip_amr)
               end if
            end if
         end do
         if(ncache>0) then
            deallocate(son)
         end if
      end do
      close(amr_n)

      allocate(son(1:ncache_max))
      allocate(xgt(1:ncache_max, 1:ndim))
      allocate(hvar(1:ncache_max, 1:twotondim, 1:nvar))
      allocate(leaf(1:ncache_max, 1:twotondim))

      allocate(xc(1:ncell_tot, 1:ndim))
      allocate(uc(1:ncell_tot, 1:nvar))

      allocate(lvlc(1:ncell_tot))

      ! Step 2: Read the actual hydro/amr data
      icell = 1
      ! Open amr file
      open(unit=amr_n, file=amr_filename(repo, iout, icpu), status='old', form='unformatted')
      call skip_read(amr_n, 29)

      ! Open hydro file
      open(unit=hydro_n, file=hydro_filename(repo, iout, icpu), status='old', form='unformatted')
      call skip_read(hydro_n, 6)

      ! Loop over levels
      do ilevel = 1, nlevelmax
         ncache = numbl2(icpu, ilevel)

         ! Loop over domains
         do jcpu = 1, nboundary + ncpu
            ! Skip ilevel, ncache
            call skip_read(hydro_n, 2)

            ! Read AMR data
            if(numbl2(jcpu, ilevel) > 0) then
               if(jcpu == icpu) then
                  call skip_read(amr_n, 3) ! Skip grid index, next, prev

                  ! Read grid center
                  do idim=1, ndim
                     read(amr_n) xgt(1:ncache, idim)
                  end do

                  read(amr_n) ! Skip father index
                  call skip_read(amr_n, 2*ndim) ! Skip nbor index

                  ! Read son index to check refinement
                  do ind = 1, twotondim
                     read(amr_n) son(1:ncache)
                     do igrid=1, ncache
                        leaf(igrid, ind) = (son(igrid) == 0)
                     end do
                  end do
                  call skip_read(amr_n, 2*twotondim) ! Skip cpu, refinement map

                  ! Read hydro variables
                  do ind = 1, twotondim
                     do ivar = 1, nvar
                        read(hydro_n) hvar(1:ncache, ind, ivar)
                     end do
                  end do

                  ! Merge amr & hydro data
                  jcell = 0

                  do igrid=1, ncache
                     do ind = 1, twotondim
                        if(leaf(igrid, ind)) then
                           xc(icell + jcell, :) = xgt(igrid, :) &
                                   + oct_offset(ind, :) * half ** ilevel
                           uc(icell + jcell, :) = hvar(igrid, ind, :)
                           lvlc(icell + jcell) = ilevel
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
      end do
      close(amr_n)
      close(hydro_n)

      deallocate(son)
      deallocate(xgt)
      deallocate(hvar)
      deallocate(leaf)

   end subroutine read_cell_single

!#####################################################################
   subroutine read_grid_single(repo, iout, icpu)
!#####################################################################
      implicit none
      character(len=128), intent(in) :: repo
      integer, intent(in) :: iout, icpu

      integer :: amr_n, hydro_n
      integer :: ilevel, jcpu, igrid, jgrid, kgrid, ivar, idim, ind
      integer :: ncache, ncache2, ncache_max, nref
      logical :: ok

      integer, dimension(1:ncpu, 1:nlevelmax) :: numbl2

      integer,      dimension(:,:),   allocatable :: son
      real(kind=8), dimension(:,:),   allocatable :: xgt
      real(kind=8), dimension(:,:,:), allocatable :: hvar
      logical,      dimension(:,:),   allocatable :: leaf


      amr_n = 10
      hydro_n = 20

      ! Step 1: Count the total number of grids
      ngrid_tot = 0
      ncache = 0
      ngrid_act = 0
      ngrid_rec = 0
      nleaf = 0
      open(unit=amr_n, file=amr_filename(repo, iout, icpu), status='old', form='unformatted')
         call skip_read(amr_n, 21)
         ! Read grid numbers
         read(amr_n) numbl2 ! Read ngridmap for safety
         call skip_read(amr_n, 7)

      do ilevel = 1, nlevelmax
         ncache_max = MAX(ncache_max, numbl2(icpu, ilevel))
         ncache = numbl2(icpu, ilevel)
         ! Loop over domains
         if(ncache > 0) then
            allocate(son(1:ncache,1:twotondim))
         end if
         do jcpu = 1, nboundary + ncpu
            ncache2 = numbl2(jcpu, ilevel)
            if(ncache2 > 0) then
               if(jcpu == icpu) then
                  ngrid_act = ngrid_act + ncache2
                  call skip_read(amr_n, 3) ! Skip grid index, next, prev

                  ! Read grid center
                  call skip_read(amr_n, ndim) ! Skip position
                  read(amr_n) ! Skip father index
                  call skip_read(amr_n, 2*ndim) ! Skip nbor index
                  ! Read son index to check refinement
                  do ind = 1, twotondim
                     read(amr_n) son(:, ind)
                  end do

                  do igrid = 1, ncache
                     ok = .false.
                     do ind = 1, twotondim
                        ok = ok .or. (son(igrid, ind) == 0)
                        if(son(igrid, ind) == 0) nleaf = nleaf + 1
                     end do
                     if(ok) ngrid_tot = ngrid_tot + 1

                  end do
                  call skip_read(amr_n, 2*twotondim) ! Skip cpu, refinement map
               else
                  ngrid_rec = ngrid_rec + ncache2
                  call skip_read(amr_n, skip_amr)
               end if
            end if
         end do
         if(ncache>0) then
            deallocate(son)
         end if
      end do
      close(amr_n)

      !write(*,*) "leaf / refined / reception grids :", ngrid_tot, ngrid_act - ngrid_tot, ngrid_rec

      allocate(son(1:ncache_max, 1:twotondim))
      allocate(xgt(1:ncache_max, 1:ndim))
      allocate(hvar(1:ncache_max, 1:twotondim, 1:nvar))
      allocate(leaf(1:ncache_max, 1:twotondim))

      allocate(xg(1:ngrid_tot, 1:ndim))
      allocate(ug(1:ngrid_tot, 1:twotondim, 1:nvar))
      allocate(refg(1:ngrid_tot, 1:twotondim))
      allocate(lvlg(1:ngrid_tot))

      ! Step 2: Read the actual hydro/amr data
      jgrid = 1
      ! Open amr file
      open(unit=amr_n, file=amr_filename(repo, iout, icpu), status='old', form='unformatted')
      call skip_read(amr_n, 29)

      ! Open hydro file
      open(unit=hydro_n, file=hydro_filename(repo, iout, icpu), status='old', form='unformatted')
      call skip_read(hydro_n, 6)

      ! Loop over levels
      do ilevel = 1, nlevelmax
         ! Loop over domains
         do jcpu = 1, nboundary + ncpu
            ! Skip ilevel, ncache
            call skip_read(hydro_n, 2)

            ! Read AMR data
            ncache = numbl2(jcpu, ilevel)
            if(ncache > 0) then
               if(jcpu == icpu) then
                  call skip_read(amr_n, 3) ! Skip grid index, next, prev

                  ! Read grid center
                  do idim=1, ndim
                     read(amr_n) xgt(1:ncache, idim)
                  end do

                  read(amr_n) ! Skip father index
                  call skip_read(amr_n, 2*ndim) ! Skip nbor index

                  ! Read son index to check refinement
                  do ind = 1, twotondim
                     read(amr_n) son(1:ncache, ind)
                  end do

                  do ind = 1, twotondim
                     do igrid=1, ncache
                        leaf(igrid, ind) = (son(igrid, ind) == 0)
                     end do
                  end do

                  call skip_read(amr_n, 2*twotondim) ! Skip cpu, refinement map

                  ! Read hydro variables
                  do ind = 1, twotondim
                     do ivar = 1, nvar
                        read(hydro_n) hvar(1:ncache, ind, ivar)
                     end do
                  end do

                  ! Merge amr & hydro data
                  kgrid = 0

                  do igrid=1, ncache
                     ok = .false.
                     do ind = 1, twotondim
                        ok = ok .or. leaf(igrid, ind)
                     end do

                     if(ok) then
                        xg(jgrid + kgrid, :) = xgt(igrid, :)
                        ug(jgrid + kgrid, :, :) = hvar(igrid, :, :)
                        refg(jgrid + kgrid, :) = leaf(igrid, :)
                        lvlg(jgrid + kgrid) = ilevel
                        kgrid = kgrid + 1
                     end if
                  end do
                  jgrid = jgrid + kgrid

               else
                  call skip_read(amr_n, skip_amr)
                  call skip_read(hydro_n, skip_hydro)
               end if
            end if
         end do
      end do
      close(amr_n)
      close(hydro_n)

      deallocate(son)
      deallocate(xgt)
      deallocate(hvar)
      deallocate(leaf)

   end subroutine read_grid_single

!#####################################################################
   subroutine write_ripses(repo, iout, icpu)
!#####################################################################
      ! Write currently cached grid data to ripses format
      implicit none
      character(len=128), intent(in) :: repo
      integer, intent(in) :: iout, icpu

      integer :: grid_n, nskip

      grid_n = 50
      nskip = 19 ! number of reads to skip header

      open(unit=grid_n, file=grid_filename(repo, iout, icpu), form='unformatted')
      write(grid_n)ncpu
      write(grid_n)ndim
      write(grid_n)ngrid_tot, nleaf
      write(grid_n)nvar
      write(grid_n)nskip

      write(grid_n)nx,ny,nz
      write(grid_n)nlevelmax
      write(grid_n)ngridmax
      write(grid_n)nboundary
      write(grid_n)ngrid_current
      write(grid_n)boxlen

      write(grid_n)noutput,iout,ifout
      write(grid_n)tout(1:noutput)
      write(grid_n)aout(1:noutput)
      write(grid_n)t
      write(grid_n)dtold(1:nlevelmax)
      write(grid_n)dtnew(1:nlevelmax)
      write(grid_n)nstep,nstep_coarse
      write(grid_n)einit,mass_tot_0,rho_tot
      write(grid_n)omega_m,omega_l,omega_k,omega_b,h0,aexp_ini,boxlen_ini
      write(grid_n)aexp,hexp,aexp_old,epot_tot_int,epot_tot_old
      write(grid_n)mass_sph
      write(grid_n)gamma

      write(grid_n)numbl(1:ncpu,1:nlevelmax)

      write(grid_n) xg
      write(grid_n) ug
      write(grid_n) refg
      write(grid_n) lvlg
      close(grid_n)

   end subroutine write_ripses

!#####################################################################
   subroutine read_ripses_single(repo, iout, icpu)
!#####################################################################
      implicit none
      ! Read grid data from ripses-format file
      character(len=128), intent(in) :: repo
      integer, intent(in) :: iout, icpu

      integer :: grid_n, nskip

      grid_n = 50

      open(unit=grid_n, file=grid_filename(repo, iout, icpu), form='unformatted')
      read(grid_n)ncpu
      read(grid_n)ndim
      read(grid_n)ngrid_tot, nleaf
      read(grid_n)nvar
      read(grid_n)nskip

      call skip_read(grid_n, nskip)

      twotondim = 2**ndim
      allocate(xg(1:ngrid_tot, 1:ndim))
      allocate(ug(1:ngrid_tot, 1:twotondim, 1:nvar))
      allocate(refg(1:ngrid_tot, 1:twotondim))
      allocate(lvlg(1:ngrid_tot))

      read(grid_n) xg
      read(grid_n) ug
      read(grid_n) refg
      read(grid_n) lvlg
      close(grid_n)

   end subroutine read_ripses_single

!#####################################################################
   subroutine read_ripses_cell(repo, iout, cpu_list)
!#####################################################################
      implicit none

      character(len=128), intent(in) :: repo
      integer, intent(in) :: iout
      integer, dimension(:), intent(in) :: cpu_list

      integer :: i, icpu, igrid, ind, icell, jcell, ngrid_now, nleaf_now
      integer :: grid_n, nskip

      real(kind=8), dimension(1:8, 1:3) :: oct_offset

      ! Positional offset for 3-dimensional oct-tree
      oct_offset = reshape((/&
          -0.5,  0.5, -0.5,  0.5, -0.5,  0.5, -0.5,  0.5, &
          -0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5, &
          -0.5, -0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5  &
      /), shape(oct_offset))

      grid_n = 50

      ncell_tot = 0
      do i = 1, SIZE(cpu_list)
         icpu = cpu_list(i)
         open(unit=grid_n, file=grid_filename(repo, iout, icpu), form='unformatted')
         read(grid_n) ! ncpu
         read(grid_n) ndim
         read(grid_n) ngrid_now, nleaf_now
         read(grid_n) nvar
         close(grid_n)
         ncell_tot = ncell_tot + nleaf_now
      end do

      allocate(xc(1:ncell_tot, 1:ndim), uc(1:ncell_tot, 1:nvar), lvlc(1:ncell_tot), cpuc(1:ncell_tot))

      icell = 1
      do i = 1, SIZE(cpu_list)
         icpu = cpu_list(i)
         call read_ripses_single(repo, iout, icpu)

         ! Convert to cell format
         jcell = 0
         do ind = 1, twotondim
            do igrid = 1, ngrid_tot
               if(refg(igrid, ind)) then
                  xc(icell + jcell, :) = xg(igrid, :) &
                          + oct_offset(ind, :) * half ** lvlg(igrid)
                  uc(icell + jcell, :) = ug(igrid, ind, :)
                  lvlc(icell + jcell) = lvlg(igrid)
                  cpuc(icell + jcell) = icpu
                  jcell = jcell + 1
               end if
            end do
         end do
         icell = icell + jcell

         call close_grid
      end do

   end subroutine read_ripses_cell

!#####################################################################
! RAMSES particle data format
!#####################################################################
! ver.1 : x(3), v(3), mass, id
! ver.2 : x(3), v(3), mass, id, (family, tag), (epoch), (metal)
! option : star, metal, fam, longint

!#####################################################################
   subroutine read_part(repo, iout, cpulist)
!#####################################################################
      implicit none

      integer :: part_n
      integer :: i, j, icpu, idim
      integer :: npart
      integer(kind=8) :: npart_tot, ipart

      part_n = 30

      ! Count total number of particles
      npart_tot = 0
      do i = 1, SIZE(cpu_list)
         icpu = cpu_list(i)
         open(unit=part_n, file=part_filename(repo, iout, icpu), form='unformatted')
         call skip_read(part_n, 2)
         read(part_n) npart
         npart_tot = npart_tot + npart
         close(part_n)
      end do

      allocate(xp(1:npart_tot, 1:ndim), vp(1:npart_tot, 1:ndim), mp(1:npart_tot))
      if(longint) then
         allocate(idpl(1:npart_tot))
      else
         allocate(idp(1:npart_tot))
      end if

      if(star) allocate(tp(1:npart_tot))
      if(metal) allocate(zp(1:npart_tot))
      if(family) allocate(fam(1:npart_tot), tag(1:npart_tot))

      jpart = 0
      do i = 1, SIZE(cpu_list)
         icpu = cpu_list(i)
         open(unit=part_n, file=part_filename(repo, iout, icpu), form='unformatted')
         read(part_n) ncpu
         read(part_n) ndim
         read(part_n) npart
         call skip_read(part_n, 4)

         ipart = jpart + 1
         jpart = jpart + npart
         do idim = 1, ndim
            read(part_n) xp(ipart:jpart, idim)
         end do
         do idim = 1, ndim
            read(part_n) vp(ipart:jpart, idim)
         end do
         read(part_n) mp(ipart:jpart)
         if(longint) then
            read(part_n) idp(ipart:jpart)
         else
            read(part_n) idpl(ipart:jpart)
         end if

      end do

   end subroutine read_part
!#####################################################################
    subroutine read_part_single(repo, iout, icpu, longint)
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
        elseif(mode == 'iap' .or. mode == 'gem' .or. mode == 'fornax' .or. mode == 'none') then ! New RAMSES version that includes family, tag
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

            elseif(mode == 'iap' .or. mode == 'gem' .or. mode == 'none' .or. mode == 'fornax') then
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
      call close_grid
      call close_cell
   end subroutine close

!#####################################################################
   subroutine close_grid()
!#####################################################################
      ! Clean old data table is used more then once
      implicit none
      if(allocated(xg)) deallocate(xg)
      if(allocated(ug)) deallocate(ug)
      if(allocated(refg)) deallocate(refg)
      if(allocated(lvlg)) deallocate(lvlg)

      if(allocated(tout)) deallocate(tout)
      if(allocated(aout)) deallocate(aout)
      if(allocated(dtold)) deallocate(dtold)
      if(allocated(dtnew)) deallocate(dtnew)
      if(allocated(headl)) deallocate(headl)
      if(allocated(taill)) deallocate(taill)
      if(allocated(numbl)) deallocate(numbl)
      if(allocated(numbtot)) deallocate(numbtot)
      if(allocated(bound_key)) deallocate(bound_key)
   end subroutine close_grid

!#####################################################################
   subroutine close_cell()
!#####################################################################
      ! Clean old data table is used more then once
      implicit none
      if(allocated(xc)) deallocate(xc)
      if(allocated(uc)) deallocate(uc)
      if(allocated(lvlc)) deallocate(lvlc)
      if(allocated(cpuc)) deallocate(cpuc)

      if(allocated(tout)) deallocate(tout)
      if(allocated(aout)) deallocate(aout)
      if(allocated(dtold)) deallocate(dtold)
      if(allocated(dtnew)) deallocate(dtnew)
      if(allocated(headl)) deallocate(headl)
      if(allocated(taill)) deallocate(taill)
      if(allocated(numbl)) deallocate(numbl)
      if(allocated(numbtot)) deallocate(numbtot)
      if(allocated(bound_key)) deallocate(bound_key)
   end subroutine close_cell

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
   character(len=128) function amr_filename(repo, iout, icpu)
!#####################################################################
      implicit none
      character(len=128), intent(in)  :: repo
      integer,            intent(in)  :: iout, icpu

      amr_filename = TRIM(repo)//'/output_'//charind(iout)//'/amr_'//charind(iout)//'.out'//charind(icpu)

   end function amr_filename

!#####################################################################
   character(len=128) function cell_filename(repo, iout, icpu)
!#####################################################################
      implicit none
      character(len=128), intent(in)  :: repo
      integer,            intent(in)  :: iout, icpu

      cell_filename = TRIM(repo)//'/output_'//charind(iout)//'/cell_'//charind(iout)//'.out'//charind(icpu)

   end function cell_filename

!#####################################################################
   character(len=128) function grid_filename(repo, iout, icpu)
!#####################################################################
      implicit none
      character(len=128), intent(in)  :: repo
      integer,            intent(in)  :: iout, icpu

      grid_filename = TRIM(repo)//'/output_'//charind(iout)//'/grid_'//charind(iout)//'.out'//charind(icpu)

   end function grid_filename

!#####################################################################
   character(len=128) function hydro_filename(repo, iout, icpu)
!#####################################################################
      implicit none
      character(len=128), intent(in)  :: repo
      integer,            intent(in)  :: iout, icpu

      hydro_filename = TRIM(repo)//'/output_'//charind(iout)//'/hydro_'//charind(iout)//'.out'//charind(icpu)

   end function hydro_filename

end module io_ramses