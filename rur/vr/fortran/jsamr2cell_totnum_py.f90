!234567
MODULE jsamr2cell_totnum_py
      INTEGER(KIND=4) ntot, nvarh
      INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: mg_ind
CONTAINS
!234567
      SUBROUTINE jsamr2cell_totnum(larr, darr, fname_a, fname_h, &
                      domlist)

      USE omp_lib
      IMPLICIT NONE

      INTEGER(KIND=4), DIMENSION(:), INTENT(IN) :: larr
      REAL(KIND=8), DIMENSION(:), INTENT(IN) :: darr

      CHARACTER(1000), INTENT(IN) ::fname_a
      CHARACTER(1000), INTENT(IN) :: fname_h

      INTEGER(KIND=4), DIMENSION(:), INTENT(IN) :: domlist

!!!!! LOCAL VARIABLES
      INTEGER(KIND=4) i, j, icpu, k
      INTEGER(KIND=4) n_thread, cpu0, cpu1, uout, uout2

      !!INFO RELATED
      INTEGER(KIND=4) ncpu, ndim, levelmin, levelmax, ndom
      !!AMR RELATED
      INTEGER(KIND=4) mg_num(larr(6),larr(9))
      INTEGER(KIND=4) mg_indtmp(larr(6))
      INTEGER(KIND=4) ind_tmp

      CHARACTER(100) domnum, fdum_a, fdum_h

      IF(ALLOCATED(mg_ind)) DEALLOCATE(mg_ind)
      ALLOCATE(mg_ind(1:larr(6)))
      ndom = larr(1)
      !cpu0 = larr(1)
      !cpu1 = larr(2)
      n_thread = larr(3)
      ncpu = larr(6)
      ndim = larr(7)
      levelmin = larr(8)
      levelmax = larr(9)
      CALL OMP_SET_NUM_THREADS(n_thread)

      !!!!!
      !! READ # OF HYDRO VARIBLES
      !!!!!
      WRITE(domnum, '(I5.5)') domlist(1)
      fdum_h = TRIM(fname_h)//TRIM(domnum)
      OPEN(UNIT=10, FILE=fdum_h, FORM='unformatted', STATUS='old')
      READ(10); READ(10) nvarh; CLOSE(10)

      !!!!!
      !! COUNT TOTAL NUM
      !!!!!
      !$OMP PARALLEL DO default(shared) &
      !$OMP & schedule(static) private(domnum, fdum_a) &
      !$OMP & private(uout, mg_num, ind_tmp, i, j, icpu)
      DO k=1, ndom
        icpu = domlist(k)
      !DO icpu=cpu0, cpu1
        WRITE(domnum, '(I5.5)') icpu
        fdum_a = TRIM(fname_a)//TRIM(domnum)
        uout = OMP_GET_THREAD_NUM() + 10
        OPEN(UNIT=uout, FILE=fdum_a, FORM='unformatted', STATUS='old')
        DO j=1, 21
          READ(uout)
        ENDDO
        READ(uout) mg_num
        CLOSE(uout)

        ind_tmp = 0
        DO i=1, ncpu
          DO j=1, levelmax
            ind_tmp = ind_tmp + mg_num(i,j) * (2**ndim)
          ENDDO
        ENDDO
        mg_indtmp(icpu) = ind_tmp
      ENDDO
      !$OMP END PARALLEL DO

      ind_tmp = 0
      DO i=1, ncpu
        mg_ind(i) = 0
        !IF(i.GE.cpu0 .AND. i.LE.cpu1)THEN
        DO j=1, ndom
          IF(i.EQ.domlist(j)) THEN
        !IF(i.GE.cpu0 .AND. i.LE.cpu1)THEN
            mg_ind(i) = ind_tmp
            ind_tmp = ind_tmp + mg_indtmp(i)
          ENDIF
        ENDDO
      ENDDO
      ntot = ind_tmp

      END SUBROUTINE

      SUBROUTINE jsamr2cell_totnum_free()
              IF(ALLOCATED(mg_ind)) DEALLOCATE(mg_ind)
      END SUBROUTINE
END MODULE
