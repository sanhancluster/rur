!234567
MODULE jsrd_part_py
      REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: xx, yy, zz, &
        vx, vy, vz, mp, ap, zp
      INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: fam, tag, domain
      INTEGER(KIND=8), DIMENSION(:), ALLOCATABLE :: idvar
CONTAINS
!234567
      SUBROUTINE jsrd_part(larr, darr, fname, part_ind, domlist)
        !dblvar, lonvar, idvar, domlist)

      USE omp_lib

      IMPLICIT NONE
      INTEGER(KIND=4), DIMENSION(:), INTENT(IN) :: larr
      REAL(KIND=8), DIMENSION(:), INTENT(IN) :: darr
      CHARACTER(1000), INTENT(IN) :: fname
      INTEGER(KIND=4), DIMENSION(:), INTENT(IN) :: part_ind
      INTEGER(KIND=4), DIMENSION(:), INTENT(IN) :: domlist

!!!!! LOCAL VARIABLES
      INTEGER(KIND=4) i, j, k, i2
      INTEGER(KIND=4) n_thread, cpu0, cpu1 , uout, npart, ncpu, npart_tot

      REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: tmp_dbl
      INTEGER(KIND=1), DIMENSION(:), ALLOCATABLE :: tmp_byte
      INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: tmp_int
      INTEGER(KIND=8), DIMENSION(:), ALLOCATABLE :: tmp_l64
      CHARACTER*(100) domnum, fdum

      ncpu = larr(1)
      n_thread = larr(3)
      npart_tot = larr(5)
      IF (n_thread .GT. ncpu) n_thread=ncpu
      CALL OMP_SET_NUM_THREADS(n_thread)
      !
      IF(ALLOCATED(xx)) DEALLOCATE(xx)
      IF(ALLOCATED(yy)) DEALLOCATE(yy)
      IF(ALLOCATED(zz)) DEALLOCATE(zz)

      IF(ALLOCATED(vx)) DEALLOCATE(vx)
      IF(ALLOCATED(vy)) DEALLOCATE(vy)
      IF(ALLOCATED(vz)) DEALLOCATE(vz)

      IF(ALLOCATED(mp)) DEALLOCATE(mp)
      IF(ALLOCATED(ap)) DEALLOCATE(ap)
      IF(ALLOCATED(zp)) DEALLOCATE(zp)

      IF(ALLOCATED(fam)) DEALLOCATE(fam)
      IF(ALLOCATED(tag)) DEALLOCATE(tag)
      IF(ALLOCATED(domain)) DEALLOCATE(domain)

      IF(ALLOCATED(idvar)) DEALLOCATE(idvar)

      ALLOCATE(xx(1:npart_tot))
      ALLOCATE(yy(1:npart_tot))
      ALLOCATE(zz(1:npart_tot))

      ALLOCATE(vx(1:npart_tot))
      ALLOCATE(vy(1:npart_tot))
      ALLOCATE(vz(1:npart_tot))

      ALLOCATE(mp(1:npart_tot))
      ALLOCATE(ap(1:npart_tot))
      ALLOCATE(zp(1:npart_tot))

      ALLOCATE(fam(1:npart_tot))
      ALLOCATE(tag(1:npart_tot))
      ALLOCATE(domain(1:npart_tot))

      ALLOCATE(idvar(1:npart_tot))

      !domnum, fdum, uout, npart, tmp_dbl, tmp_l64, tmp_int, tmp_byte, j, k
      !$OMP PARALLEL DO default(shared) &
      !$OMP & schedule(static,10) private(domnum, fdum, uout, npart) &
      !$OMP & private(tmp_dbl, tmp_l64, tmp_int, tmp_byte, j, k, i)
      DO i2=1, ncpu
        i = domlist(i2)
        WRITE(domnum, '(I5.5)') i
        fdum = TRIM(fname)//TRIM(domnum)

        uout = OMP_GET_THREAD_NUM()
        uout = uout + 10

        OPEN(UNIT=uout, FILE=fdum, FORM='unformatted', STATUS='old')
        READ(uout); READ(uout); READ(uout) npart; READ(uout);
        READ(uout); READ(uout); READ(uout); READ(uout);

        IF(npart .EQ. 0) CYCLE
        ALLOCATE(tmp_dbl(1:npart))
        ALLOCATE(tmp_l64(1:npart))
        ALLOCATE(tmp_int(1:npart))
        ALLOCATE(tmp_byte(1:npart))

        !x, y, z
        READ(uout) tmp_dbl
        DO k=1, npart
            xx(part_ind(i2)+k) = tmp_dbl(k)
        ENDDO

        READ(uout) tmp_dbl
        DO k=1, npart
            yy(part_ind(i2)+k) = tmp_dbl(k)
        ENDDO

        READ(uout) tmp_dbl
        DO k=1, npart
            zz(part_ind(i2)+k) = tmp_dbl(k)
        ENDDO

        !vx, vy, vz
        READ(uout) tmp_dbl
        DO k=1, npart
            vx(part_ind(i2)+k) = tmp_dbl(k)
        ENDDO

        READ(uout) tmp_dbl
        DO k=1, npart
            vy(part_ind(i2)+k) = tmp_dbl(k)
        ENDDO

        READ(uout) tmp_dbl
        DO k=1, npart
            vz(part_ind(i2)+k) = tmp_dbl(k)
        ENDDO

        !mass
        READ(uout) tmp_dbl
        DO k=1, npart
          mp(part_ind(i2)+k) = tmp_dbl(k)
        ENDDO

        !ID
        IF(larr(20) .GE. 10) THEN !long long ID
          READ(uout) tmp_l64
          DO k=1, npart
            idvar(part_ind(i2)+k) = tmp_l64(k)
          ENDDO
        ELSE !long ID
          READ(uout) tmp_int
          DO k=1, npart
            idvar(part_ind(i2)+k) = tmp_int(k)
          ENDDO
        ENDIF

        !Level
        READ(uout) tmp_int

        !FAM / TAG
        IF(larr(19) .GE. 10) THEN !READ THEM DIRECTLY
          READ(uout) tmp_byte !FAM
          DO k=1, npart
            fam(part_ind(i2)+k) = tmp_byte(k)
            IF(fam(part_ind(i2)+k) .GT. 100) &
                fam(part_ind(i2)+k) = fam(part_ind(i2)+k) - 255
          ENDDO

          READ(uout) tmp_byte !TAG
          DO k=1, npart
            tag(part_ind(i2)+k) = tmp_byte(k)
            IF(tag(part_ind(i2)+k) .GT. 100) &
                tag(part_ind(i2)+k) = tag(part_ind(i2)+k) - 255
          ENDDO
        ENDIF

        !AGE
        READ(uout) tmp_dbl
        DO k=1, npart
          ap(part_ind(i2)+k) = tmp_dbl(k)
        ENDDO

        !METALLICITY
        READ(uout) tmp_dbl
        DO k=1, npart
          zp(part_ind(i2)+k) = tmp_dbl(k)
        ENDDO

        !DOMAIN
        DO k=1, npart
          domain(part_ind(i2)+k) = i
        ENDDO

        !FAM Adding
        IF(larr(19) .LT. 10) THEN
          DO k=1, npart
          !DM
          IF(ap(part_ind(i2)+k) .EQ. 0 .AND. &
                idvar(part_ind(i2)+k).GT.0)THEN
            fam(part_ind(i2)+k) = 1
          ENDIF

          !STAR
          IF(ap(part_ind(i2)+k) .LT. 0.) THEN
            fam(part_ind(i2)+k) = 2
          ENDIF
          
          !SINK (NEGATIVE ID)

          !TRACER

          ENDDO
        ENDIF

        DEALLOCATE(tmp_dbl)
        DEALLOCATE(tmp_l64)
        DEALLOCATE(tmp_int)
        DEALLOCATE(tmp_byte)
        CLOSE(uout)
      ENDDO
      !$OMP END PARALLEL DO



      END SUBROUTINE

      SUBROUTINE jsrd_part_free()
              IF(ALLOCATED(xx))DEALLOCATE(xx)
              IF(ALLOCATED(yy))DEALLOCATE(yy)
              IF(ALLOCATED(zz))DEALLOCATE(zz)
              IF(ALLOCATED(vx))DEALLOCATE(vx)
              IF(ALLOCATED(vy))DEALLOCATE(vy)
              IF(ALLOCATED(vz))DEALLOCATE(vz)
              IF(ALLOCATED(mp))DEALLOCATE(mp)
              IF(ALLOCATED(ap))DEALLOCATE(ap)
              IF(ALLOCATED(fam))DEALLOCATE(fam)
              IF(ALLOCATED(tag))DEALLOCATE(tag)
              IF(ALLOCATED(domain))DEALLOCATE(domain)
              IF(ALLOCATED(idvar))DEALLOCATE(idvar)
      END SUBROUTINE
END MODULE
