!234567
MODULE jsamr2cell_py
      REAL(KIND=8), DIMENSION(:,:), ALLOCATABLE :: mesh_xg, mesh_hd
      REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: mesh_dx
      INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: mesh_lv
CONTAINS
!234567
      SUBROUTINE jsamr2cell(larr, darr, fname_a, fname_h, fname_i, &
                    mg_ind, domlist)

      USE omp_lib
      IMPLICIT NONE

      INTEGER(KIND=4), DIMENSION(:), INTENT(IN) :: larr
      REAL(KIND=8), DIMENSION(:), INTENT(IN) :: darr

      CHARACTER(1000), INTENT(IN) :: fname_a
      CHARACTER(1000), INTENT(IN) :: fname_h
      CHARACTER(1000), INTENT(IN) :: fname_i

      INTEGER(KIND=4), DIMENSION(:), INTENT(IN) ::mg_ind
      INTEGER(KIND=4), DIMENSION(:), INTENT(IN) :: domlist

!!!!! LOCAL VARIABLES
      INTEGER(KIND=4) i, j, k, l, impi, ilevel, ind, icpu, idim, ivar
      INTEGER(KIND=4) n_thread, cpu0, cpu1, uout, uout2, ndom, i2

      CHARACTER(100) domnum, fdum_a, fdum_h

      INTEGER(KIND=4) ind_tmp, merge_ind, merge_ind2
      !!INFO VAR
      INTEGER(KIND=4) levelmin, levelmax
      REAL(KIND=8) t
      CHARACTER(LEN=80) ordering
      CHARACTER(LEN=128) temp_label
      REAL(KIND=8) :: xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1
      INTEGER(KIND=4) nx_full, ny_full, nz_full
      INTEGER(KIND=4) imin, imax, jmin, jmax, kmin, kmax

      !!AMR VAR
      INTEGER(KIND=4) ncpu, ndim, nx, ny, nz, ngridmax
      INTEGER(KIND=4) nboundary, ngrid_current, twotondim
      REAL(KIND=8) boxlen
      REAL(KIND=8), DIMENSION(1:3) :: xbound=(/0d0,0d0,0d0/)
      INTEGER(KIND=4), DIMENSION(:,:), ALLOCATABLE :: ngridfile
      INTEGER(KIND=4), DIMENSION(:,:), ALLOCATABLE :: ngridlevel
      INTEGER(KIND=4), DIMENSION(:,:), ALLOCATABLE :: ngridbound

      INTEGER(KIND=4) nvarh, ntot

      !!LOOP VAR
      REAL(KIND=8) dx, dx2
      INTEGER(KIND=4) ix, iy, iz, ngrida
      REAL(KIND=8), DIMENSION(1:8,1:3) :: xc
      REAL(KIND=8), DIMENSION(:), ALLOCATABLE ::rho
      REAL(KIND=8), DIMENSION(:,:), ALLOCATABLE ::x, xg
      REAL(KIND=8), DIMENSION(:,:,:), ALLOCATABLE ::var
      INTEGER(KIND=4), DIMENSION(:,:), ALLOCATABLE :: son
      LOGICAL, DIMENSION(:), ALLOCATABLE ::ref
      LOGICAL ok_cell

      IF(ALLOCATED(mesh_xg)) DEALLOCATE(mesh_xg)
      IF(ALLOCATED(mesh_dx)) DEALLOCATE(mesh_dx)
      IF(ALLOCATED(mesh_hd)) DEALLOCATE(mesh_hd)
      IF(ALLOCATED(mesh_lv)) DEALLOCATE(mesh_lv)

      ALLOCATE(mesh_xg(1:larr(11),1:larr(8)))
      ALLOCATE(mesh_dx(1:larr(11)))
      ALLOCATE(mesh_hd(1:larr(11),1:larr(12)))
      ALLOCATE(mesh_lv(1:larr(11)))
      mesh_lv   = -10

      !OPEN(20, file='/home/jinsu/amrtest.txt', status='unknown')
      !TYPE level
      !  INTEGER(KIND=4) ilevel, ngrid, imin, imax
      !  INTEGER(KIND=4) jmin, jmax, kmin, kmax
      !  REAL(KIND=4), DIMENSION(:,:,:), POINTER :: cube
      !END TYPE level

      !TYPE(level), DIMENSION(1:100) :: grid

      ndom = larr(1)
      !cpu0 = larr(1)
      !cpu1 = larr(2)
      n_thread = larr(3)
      ncpu = larr(7)
      ndim = larr(8)
      levelmin = larr(9)
      levelmax = larr(10)
      ntot = larr(11)
      nvarh = larr(12)

      twotondim = 2**ndim

      CALL OMP_SET_NUM_THREADS(n_thread)
      !!!!!
      !! READ INFO FIRST
      !!!!!
      OPEN(UNIT=10, FILE=fname_i, FORM='formatted', STATUS='old')
      DO i=1, 19
        READ(10, *)
      ENDDO
      READ(10, '(A14,A80)') temp_label, ordering
      CLOSE(10)

      !k, j, ilevel, idim, ivar
      !domnum, fdum_a, fdum_h, uout, uout2, icpu
      !nx, ny, nz, ngridmax
      !nboundary, ngrid_current, boxlen,xbound
      !ngridfile, ngridlevel, ngridbound
      !dx, dx2, nx_full, ny_full, nz_full, ind, ix, iy, iz
      !xc, vxc, ngrida
      !xg, son, var, x, rho, ref, ok_cell, merge_ind

      DO i2=1, ndom
        i = domlist(i2)
        WRITE(domnum, '(I5.5)') i
        fdum_a = TRIM(fname_a)//TRIM(domnum)
        fdum_h = TRIM(fname_h)//TRIM(domnum)
        uout = OMP_GET_THREAD_NUM() + 10
        uout2 = OMP_GET_THREAD_NUM() + ncpu

        icpu = i
        merge_ind = mg_ind(i)
        !!!!!
        !! READ AMR HEADER
        !!!!!
        OPEN(UNIT=uout, FILE=fdum_a, FORM='unformatted', STATUS='old')
        READ(uout); READ(uout); READ(uout) nx, ny, nz
        READ(uout); READ(uout) ngridmax; READ(uout) nboundary
        READ(uout) ngrid_current; READ(uout) boxlen

        xbound=(/dble(nx/2),dble(ny/2),dble(nz/2)/)

        IF(ALLOCATED(ngridlevel)) DEALLOCATE(ngridlevel)
        IF(ALLOCATED(ngridfile)) DEALLOCATE(ngridfile)
        IF(ALLOCATED(ngridbound)) DEALLOCATE(ngridbound)
        ALLOCATE(ngridlevel(1:ncpu,1:levelmax))
        ALLOCATE(ngridfile(1:ncpu+nboundary,1:levelmax))
        IF(nboundary>0) ALLOCATE(ngridbound(1:nboundary,1:levelmax))

        DO k=1,13
          READ(uout)
        ENDDO
        READ(uout) ngridlevel
        ngridfile(1:ncpu,1:levelmax) = ngridlevel

        READ(uout)
        IF(nboundary>0)THEN
          READ(uout); READ(uout)
          READ(uout) ngridbound
          ngridfile(ncpu+1:ncpu+nboundary,1:levelmax)=ngridbound
        ENDIF
        READ(uout); READ(uout)

        IF(TRIM(ordering) .EQ. 'bisection') THEN
          READ(uout); READ(uout); READ(uout); READ(uout); READ(uout)
        ELSE
          READ(uout)
        ENDIF
        READ(uout); READ(uout); READ(uout)

        !!!!!
        !! READ HYDRO
        !!!!!
        OPEN(UNIT=uout2, FILE=fdum_h, FORM='unformatted', STATUS='old')
        READ(uout2); READ(uout2); READ(uout2)
        READ(uout2); READ(uout2); READ(uout2)
        
        !!!!!
        !! LOOP OVER LEVELS
        !!!!!

        DO ilevel=1, levelmax
          dx = 0.5 ** ilevel
          dx2 = 0.5*dx
          nx_full = 2** ilevel
          ny_full = 2** ilevel
          nz_full = 2** ilevel
          DO ind=1, twotondim
            iz = (ind-1)/4
            iy = (ind-1-4*iz)/2
            ix = (ind-1-2*iy-4*iz)
            xc(ind,1) = (dble(ix)-0.5D0)*dx 
            xc(ind,2) = (dble(iy)-0.5D0)*dx
            xc(ind,3) = (dble(iz)-0.5D0)*dx
          ENDDO

          ngrida = ngridfile(icpu,ilevel)
          !! ALLOCATE
          IF(ngrida>0) THEN
            ALLOCATE(xg (1:ngrida,1:ndim))
            ALLOCATE(son(1:ngrida,1:twotondim))
            ALLOCATE(var(1:ngrida,1:twotondim,1:nvarh))
            ALLOCATE(x  (1:ngrida,1:ndim))
            ALLOCATE(rho(1:ngrida))
            ALLOCATE(ref(1:ngrida))
          ENDIF
          !What does ngridfile means?

          !!!!!
          !! LOOP OVER DOMAINS
          !!!!!
          DO j=1,nboundary+ncpu
            !! READ AMR
            !IF(ngrida>0) THEN
            IF(ngridfile(j,ilevel)>0)THEN
              READ(uout); READ(uout); READ(uout)
              DO idim=1, ndim !READ GRID CENTER
                IF(j.EQ.icpu)THEN
                  READ(uout) xg(:,idim)
                ELSE
                  READ(uout)
                ENDIF
              ENDDO

              READ(uout) ! SKIP FATHER INDEX

              DO ind=1, 2*ndim ! SKIP NBOR INDEX
                READ(uout)
              ENDDO

              DO ind=1, twotondim ! READ SON INDEX
                IF(j.EQ.icpu)THEN
                  READ(uout) son(:,ind)
                ELSE
                  READ(uout)
                ENDIF
              ENDDO

              DO ind=1, twotondim ! SKIP CPU MAP
                READ(uout)
              ENDDO

              DO ind=1, twotondim !SKIP REFINEMENT MAP
                READ(uout)
              ENDDO
            ENDIF
            !!READ HYDRO
            READ(uout2); READ(uout2)
            !IF(ngrida>0) THEN
            IF(ngridfile(j,ilevel)>0)THEN
              DO ind=1, twotondim
                DO ivar=1, nvarh
                  IF(j.EQ.icpu)THEN
                    READ(uout2) var(:,ind,ivar)
                  ELSE
                    READ(uout2)
                  ENDIF
                ENDDO
              ENDDO
            ENDIF
          ENDDO

          !!!!!
          !! LOOP OVER CELLS
          !!!!!
          IF(ngrida>0) THEN
            !DO ind=1, twotondim
              IF(ngrida>1000)THEN
                !$OMP PARALLEL DO default(shared) schedule(static,100) &
                !$OMP & private(merge_ind2, ok_cell, k, ind)
                DO k=1, ngrida !MERGE DATA
                  merge_ind2 = merge_ind + (k-1)*twotondim
                  DO ind=1, twotondim
                    x(k,1) = (xg(k,1)+xc(ind,1)-xbound(1))
                    x(k,2) = (xg(k,2)+xc(ind,2)-xbound(2))
                    x(k,3) = (xg(k,3)+xc(ind,3)-xbound(3))

                    ref(k) = son(k,ind)>0 .AND. ilevel<levelmax

                    merge_ind2 = merge_ind2 + 1
                    mesh_lv(merge_ind2) = -1
                    ok_cell= .NOT. ref(k) .AND. &
                            & (x(k,1) + dx2) >= xmin .AND. &
                            & (x(k,2) + dx2) >= ymin .AND. &
                            & (x(k,3) + dx2) >= zmin .AND. &
                            & (x(k,1) - dx2) <= xmax .AND. &
                            & (x(k,2) - dx2) <= ymax .AND. &
                            & (x(k,3) - dx2) <= zmax
                    IF(ok_cell) THEN
                      mesh_xg(merge_ind2,1) = x(k,1)
                      mesh_xg(merge_ind2,2) = x(k,2)
                      mesh_xg(merge_ind2,3) = x(k,3)
                      mesh_dx(merge_ind2)   = dx
                      mesh_lv(merge_ind2)= ilevel
                      DO ivar=1, nvarh
                        mesh_hd(merge_ind2,ivar) = var(k, ind, ivar)
                      ENDDO
                    ENDIF
                  ENDDO
                ENDDO
!#if ngrida>1000
                !$OMP END PARALLEL DO
!#endif
              ELSE
                DO k=1, ngrida !MERGE DATA
                  merge_ind2 = merge_ind + (k-1)*twotondim
                  DO ind=1, twotondim
                    x(k,1) = (xg(k,1)+xc(ind,1)-xbound(1))
                    x(k,2) = (xg(k,2)+xc(ind,2)-xbound(2))
                    x(k,3) = (xg(k,3)+xc(ind,3)-xbound(3))
                        
                    ref(k) = son(k,ind)>0 .AND. ilevel<levelmax

                    merge_ind2 = merge_ind2 + 1
                    mesh_lv(merge_ind2) = -1
                    ok_cell= .NOT. ref(k) .AND. &
                            & (x(k,1) + dx2) >= xmin .AND. &
                            & (x(k,2) + dx2) >= ymin .AND. &
                            & (x(k,3) + dx2) >= zmin .AND. &
                            & (x(k,1) - dx2) <= xmax .AND. &
                            & (x(k,2) - dx2) <= ymax .AND. &
                            & (x(k,3) - dx2) <= zmax
                    IF(ok_cell) THEN
                      mesh_xg(merge_ind2,1) = x(k,1)
                      mesh_xg(merge_ind2,2) = x(k,2)
                      mesh_xg(merge_ind2,3) = x(k,3)
                      mesh_dx(merge_ind2)   = dx
                      mesh_lv(merge_ind2)= ilevel
                      DO ivar=1, nvarh
                        mesh_hd(merge_ind2,ivar) = var(k, ind, ivar)
                      ENDDO
                    ENDIF
                  ENDDO
                ENDDO

              ENDIF
              merge_ind = merge_ind + ngrida*twotondim
              !ENDDO !END LOOP FOR CELLS

              DEALLOCATE(xg, son, var, ref, rho, x)
            ENDIF
          !ENDDO ! END LOOP FOR DOMAINS
        ENDDO ! END LOOP FOR LEVELS
        CLOSE(uout)
        CLOSE(uout2)

        !CLOSE(20)
        DEALLOCATE(ngridfile, ngridlevel)
        IF(nboundary>0) DEALLOCATE(ngridbound)
      ENDDO ! END LOOP FOR CPUs
      !!$OMP END PARALLEL DO

      END SUBROUTINE

      SUBROUTINE jsamr2cell_free()
              IF(ALLOCATED(mesh_xg)) DEALLOCATE(mesh_xg)
              IF(ALLOCATED(mesh_hd)) DEALLOCATE(mesh_hd)
              IF(ALLOCATED(mesh_dx)) DEALLOCATE(mesh_dx)
              IF(ALLOCATED(mesh_lv)) DEALLOCATE(mesh_lv)
      END SUBROUTINE
END MODULE
