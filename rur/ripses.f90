! Created by San Han on 20. 1. 13..

program ripses
#ifdef MPI
    use mpi
#endif
    use omp_lib
    use io_ramses
    implicit none
    integer :: nrank, nthr, nfile, istart, iend
    integer :: myid, mythr
    integer :: iout, icpu, ierr
    integer :: ngrid_all, ngrid_rip, ngrid_all1, ngrid_rip1, ngrid_all2, ngrid_rip2
    real :: frac

    character(len=128) :: repo

#ifdef MPI
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    myid=myid+1
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nrank, ierr)
    if(myid == 1) write(*,'("Running with MPI mode, NRANK = ",I5)'), nrank
#else
    myid = 1
    nrank = 1
#endif

    if(myid == 1) then
        print*, "==========================================="
        print*, "                                           "
        print*, "  RIPSES : RAMSES AMR - Hydro data ripper  "
        print*, "      By San Han (Yonsei University)       "
        print*, "                 v 1.0                     "
        print*, "                                           "
        print*, "==========================================="

        ngrid_all2 = 0
        ngrid_rip2 = 0

        print*, "Enter repository path (.: on-path mode)"
        read (*,'(A)') repo

        print*, "Enter output number range to rip (start, end)"
        read (*,*) istart, iend
    end if
    !repo = '~/testruns/ripses_sample'

#ifdef MPI
    call MPI_BCAST(repo, 128, MPI_CHARACTER, 0, MPI_COMM_WORLD, ierr)
    call MPI_BCAST(istart, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
    call MPI_BCAST(iend, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
#endif

    if(myid == 1) write(*,'("Ripping cell data for iout = ",I5," ~ ",I5,"...")') istart, iend
    do iout = istart, iend
        call read_grid_header(repo, iout, 1)
        call close()

        ngrid_all = 0
        ngrid_rip = 0
        do icpu = myid, ncpu, nrank
            !write(*,*) "icpu = ", icpu
            call read_grid_header(repo, iout, icpu)
            call read_grid_single(repo, iout, icpu)
            ngrid_all = ngrid_all + ngrid_act + ngrid_rec
            ngrid_rip = ngrid_rip + ngrid_act + ngrid_rec - ngrid_tot
            call write_ripses(repo, iout, icpu)
            call close()
        end do

#ifdef MPI
        call MPI_REDUCE(ngrid_all, ngrid_all1, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(ngrid_rip, ngrid_rip1, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
#else
        ngrid_all1 = ngrid_all
        ngrid_rip1 = ngrid_rip
#endif
        if(myid == 1) then
            frac = real(ngrid_rip1) / real(ngrid_all1) * 100
            write(*,'("IOUT = ",I5,": ",I10," / ",I10," grids (",F5.2,"%) has been removed")') iout, ngrid_rip1, ngrid_all1, frac
            ngrid_all2 = ngrid_all2 + ngrid_all1
            ngrid_rip2 = ngrid_rip2 + ngrid_rip1
        end if
    end do

#ifdef MPI
    call MPI_BARRIER(MPI_COMM_WORLD, ierr)
#endif
    if(myid == 1) then
        frac = real(ngrid_rip2) / real(ngrid_all2) * 100
        write(*,'("Ripping complete! ",I10," / ",I10," grids (",F5.2,"%) has been removed")') ngrid_rip2, ngrid_all2, frac
    end if
end program ripses
