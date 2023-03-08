!module
MODULE jsrd_part_totnum_py
      INTEGER(KIND=4) npart_tot
      INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: part_ind
CONTAINS
!234567
      SUBROUTINE jsrd_part_totnum(larr, darr, fname, domlist)

      USE omp_lib

      IMPLICIT NONE
      INTEGER(KIND=4), DIMENSION(:), INTENT(IN) :: larr
      REAL(KIND=8), DIMENSION(:), INTENT(IN) :: darr

      CHARACTER(1000), INTENT(IN) :: fname
      INTEGER(KIND=4), DIMENSION(:), INTENT(IN) :: domlist

!!!!!
!! LOCAL VARIABLES
!!!!!
      INTEGER(KIND=4) i, j, k, ncpu
      INTEGER(KIND=4) npartp, n_thread, cpu0, cpu1, uout
      INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: parts!(larr(1))
      CHARACTER*(100) domnum, fdum

      ncpu      = larr(1)
      n_thread  = larr(3)


      IF(ALLOCATED(part_ind)) DEALLOCATE(part_ind)
      IF(ALLOCATED(parts)) DEALLOCATE(parts)

      ALLOCATE(parts(1:ncpu))
      ALLOCATE(part_ind(1:ncpu))
      npart_tot = 0
      part_ind = 0
      CALL OMP_SET_NUM_THREADS(n_thread)

      !$OMP PARALLEL DO default(shared) &
      !$OMP & schedule(static) private(uout,fdum,domnum,npartp,i) &
      !$OMP & reduction(+:npart_tot)
      DO j=1, ncpu
        i = domlist(j)
        WRITE(domnum,'(I5.5)') i
        uout =  OMP_GET_THREAD_NUM()
        uout = uout + 10
        fdum = TRIM(fname)//TRIM(domnum)
        OPEN(NEWUNIT=uout, FILE=fdum, FORM='unformatted', STATUS='OLD')
        READ(uout); READ(uout); READ(uout) npartp
        CLOSE(uout)
        parts(j) = npartp
        npart_tot = npart_tot + npartp
      ENDDO
      !$OMP END PARALLEL DO
      part_ind(1) = 0
      IF(ncpu .GT. 1) THEN
        DO i=2, ncpu
          DO j=1, i-1
            part_ind(i) = part_ind(i) + parts(j)
          ENDDO
        ENDDO
      ENDIF

      DEALLOCATE(parts)

      END SUBROUTINE
      SUBROUTINE jsrd_part_totnum_free()
              IF(ALLOCATED(part_ind)) DEALLOCATE(part_ind)
      END SUBROUTINE
END MODULE

