!MODULE
MODULE get_ptcl_py
      REAL(KIND=8), DIMENSION(:, :), ALLOCATABLE :: ptcl
      INTEGER(KIND=8), DIMENSION(:) :: id_raw
      INTEGER(KIND=4) n_star

CONTAINS
!234567
      subroutine get_ptcl(dir_raw, ID, dom_list, larr, darr)
      USE omp_lib

      Implicit none
      Integer(kind=4), DIMENSION(:), INTENT(IN) :: larr
      Real(kind=8), DIMENSION(:), INTENT(IN) :: darr

      Integer(kind=8), DIMENSION(:), INTENT(IN) :: ID
      !Real(kind=8) ptcl(larr(1),9)
      Integer(kind=4), DIMENSION(:), INTENT(IN) :: dom_list
      !Character*(larr(11)) dir_raw
      Character(LEN=1000), INTENT(IN) ::dir_raw

!!!!!! Local Variables
      Integer(kind=4) n_thread, n_ptcl, n_snap, n_raw, n_dom, n_str
      INTEGER(KIND=4) longint 
      Integer(kind=4) i, j, k, l, ind0, ind1, mi, horg
      REAL(KIND=8), allocatable, dimension(:,:) :: raw_dbl, raw_dbl2
      INTEGER(KIND=8), allocatable, dimension(:,:) :: raw_int, raw_int2
      REAL(KIND=8) dmp_mass

      n_ptcl    = larr(1)
      n_dom     = larr(2)
      n_snap    = larr(3)
      n_thread  = larr(4)
      n_str     = larr(11)
      longint   = larr(20)
      horg      = larr(12)
      IF(horg .EQ.0) horg = 1   !! star particle search as a default
      dmp_mass  = darr(12)

      !ALLOCATE(ptcl(1:n_ptcl,1:9))
      !ptcl = -1.0d8
      IF(ALLOCATED(ptcl)) DEALLOCATE(ptcl)
      CALL OMP_SET_NUM_THREADS(n_thread)

!!!!! RD PART

      CALL RD_PART_NBODY(dir_raw, dom_list, n_dom, n_raw, n_str, n_snap)

      ALLOCATE(raw_dbl(1:n_raw,1:9))
      ALLOCATE(raw_int(1:n_raw,1:2))

      IF(larr(19) .GT. 10) &
        CALL RD_PART(dir_raw, dom_list, n_dom, n_raw, n_str, n_snap, &
              raw_dbl, raw_int, longint)

      IF(larr(19) .LT. 10) &
        CALL RD_PART_YZiCS(dir_raw, dom_list, n_dom, n_raw, n_str, n_snap, &
              raw_dbl, raw_int, longint)

      IF(larr(19) .GT. 10) &
        CALL GET_PTCL_NUM(raw_int, raw_dbl, n_raw, n_star, n_thread, &
                horg, dmp_mass)
      IF(larr(19) .LT. 10) &
        CALL GET_PTCL_NUM_YZiCS(raw_dbl, n_raw, n_star, n_thread, &
                horg, dmp_mass)

      ALLOCATE(raw_dbl2(1:n_star,1:9))
      ALLOCATE(raw_int2(1:n_star,1:1))

      IF(larr(19) .GT. 10) &
        CALL GET_STAR_PTCL(raw_dbl, raw_int, raw_dbl2, raw_int2, &
              n_raw, n_star, horg, dmp_mass)
      IF(larr(19) .LT. 10) &
        CALL GET_STAR_PTCL_YZiCS(raw_dbl, raw_int, raw_dbl2, raw_int2, &
              n_raw, n_star, horg, dmp_mass)


      DEALLOCATE(raw_dbl, raw_int)
      IF(larr(18) .GE. 10) THEN !Return for calling raw particles
        ALLOCATE(ptcl(1:n_star,1:9))
        ALLOCATE(id_raw(1:n_star))
        ptcl = -1.0d8
        !$OMP PARALLEL DO default(shared)
        DO i=1, n_star
          DO j=1, 9
            ptcl(i,j) = raw_dbl2(i,j)
          ENDDO
          id_raw(i) = raw_int2(i)
        ENDDO
        !$OMP END PARALLEL DO
        !larr(17) = n_star
        RETURN
      ENDIF

      CALL sort_ptcl(raw_dbl2, raw_int2, n_star)

      !!-----
      !! MATCHING
      !!  *) To do list: test w/ the hash search
      !!-----

      ALLOCATE(ptcl(1:n_ptcl,1:9))
      ptcl = -1.0d8
      !$OMP PARALLEL DO default(shared) private(ind0, ind1, l, mi) schedule(dynamic)
      DO i=1, n_ptcl
        IF(ID(i) .LE. raw_int2(n_star,1) .AND. &
                ID(i) .GE. raw_int2(1,1)) THEN

          ind0 = 1
          ind1 = n_star
          DO WHILE (ind1 - ind0 .GT. 10)
            l = int((ind1 + ind0) / 2)
            IF(ID(i) .GT. raw_int2(l,1)) ind0 = l
            IF(ID(i) .LT. raw_int2(l,1)) ind1 = l
            IF(ID(i) .EQ. raw_int2(l,1)) THEN
              ind0 = l; ind1 = l; mi = l
            ENDIF
          ENDDO

          IF(ind0 .NE. ind1) THEN
            mi = -1
            DO l=ind0, ind1
              IF(ID(i) .EQ. raw_int2(l,1)) mi = l
            ENDDO
          ELSE
            mi = ind0
          ENDIF

          IF(mi .GE. 1) THEN
            DO j=1,9
              ptcl(i,j) = raw_dbl2(mi,j)
            ENDDO
          ENDIF 
        ENDIF
      ENDDO
      !$OMP END PARALLEL DO


      end subroutine

!!!!!
!! SORT BY PARTICLE ID
!!!!!
      subroutine sort_ptcl(p_dbl, p_int, nptcl)

      Implicit none
      Integer(kind=4) nptcl, i, j, ind, nptcl_dum0, nptcl_dum1

      Real(kind=8) p_dbl(nptcl,9), dumdbl(nptcl)
      Integer(kind=8) p_int(nptcl,1), dumid, p_int2(nptcl,2)
      Integer(kind=8) dumint(nptcl)

      INTEGER(KIND=4) first, last

      Do i=1, nptcl
        p_int2(i,1) = p_int(i,1)
        p_int2(i,2) = i
      Enddo

      nptcl_dum0 = 1
      nptcl_dum1 = nptcl
      Call quicksort(p_int2, nptcl, nptcl_dum0, nptcl_dum1)

      DO i=1, nptcl
        dumint(i) = p_int(i,1)
      ENDDO

      DO i=1, nptcl
        p_int(i,1) = dumint(p_int2(i,2))
      ENDDO

      DO j=1,9
        DO i=1, nptcl
          dumdbl(i) = p_dbl(i,j)
        ENDDO

        DO i=1, nptcl
          p_dbl(i,j) = dumdbl(p_int2(i,2))
        ENDDO
      ENDDO

      end subroutine
!!!!!
!! GET TARGETTED PARTICLE
!!!!!
      subroutine GET_STAR_PTCL(p_dbl, p_int, p_dbl2, p_int2, &
                      n_raw, n_star, horg, dmp_mass)

      Implicit none
      Integer(kind=4) n_raw, n_star, horg

      Real(kind=8) p_dbl(n_raw,9), p_dbl2(n_star,9), dmp_mass
      Integer(kind=8) p_int(n_raw,2), p_int2(n_star,1)

      Integer(kind=4) i, j, nn

      nn = 1
      IF(horg .GT. 0.) THEN !STAR
        Do i=1, n_raw
          IF(p_int(i,2) .eq. 2) THEN
            DO j=1, 9
              p_dbl2(nn,j) = p_dbl(i,j)
            ENDDO

            p_int2(nn,1) = p_int(i,1)
            nn = nn + 1
          ENDIF
        ENDDO
      ELSE IF (horg .LT. 0) THEN !DM
        Do i=1, n_raw
          IF(p_int(i,2) .eq. 1) THEN
            IF(ABS(p_dbl(i,7)-dmp_mass)/dmp_mass .LT. 1e-5) THEN
              DO j=1, 9
                p_dbl2(nn,j) = p_dbl(i,j)
              ENDDO

              p_int2(nn,1) = p_int(i,1)
              nn = nn + 1
            ENDIF
          ENDIF
        ENDDO
      ENDIF

      end subroutine
!!!!!
!! GET TARGETTED PARTICLE_YZiCS
!!!!!
      SUBROUTINE GET_STAR_PTCL_YZiCS(p_dbl, p_int, p_dbl2, p_int2, &
                      n_raw, n_star, horg, dmp_mass)

      Implicit none
      Integer(kind=4) n_raw, n_star, horg

      Real(kind=8) p_dbl(n_raw,9), p_dbl2(n_star,9), dmp_mass
      Integer(kind=8) p_int(n_raw,2), p_int2(n_star,1)

      Integer(kind=4) i, j, nn

      nn = 1
      IF(horg .GT. 0) THEN !STAR
        Do i=1, n_raw
          IF(p_dbl(i,8) .LT. 0) THEN
            DO j=1, 9
              p_dbl2(nn,j) = p_dbl(i,j)
            ENDDO

            p_int2(nn,1) = p_int(i,1)
            nn = nn + 1
          ENDIF
        ENDDO
      ELSE IF (horg .LT. 0) THEN !DM
        Do i=1, n_raw
          IF(p_dbl(i,8) .EQ. 0) THEN
            IF(ABS(p_dbl(i,7)-dmp_mass)/dmp_mass .LT. 1e-5) THEN
              DO j=1, 9
                p_dbl2(nn,j) = p_dbl(i,j)
              ENDDO

              p_int2(nn,1) = p_int(i,1)
              nn = nn + 1
            ENDIF
          ENDIF
        ENDDO
      ENDIF

      END subroutine
!!!!!
!! GET PARTICLE NUM
!!!!!
      SUBROUTINE GET_PTCL_NUM(rint, rdbl, n_raw, n_star, n_thread, &
                horg, dmp_mass)

      IMPLICIT NONE
      INTEGER(KIND=4) n_raw, n_star, n_thread
      INTEGER(KIND=8) rint(n_raw, 2)
      REAL(KIND=8) rdbl(n_raw,9)

      INTEGER(KIND=4) i, j, k, horg
      REAL(KIND=8) dmp_mass

      CALL OMP_SET_NUM_THREADS(n_thread)
      n_star = 0

      IF(horg .GT. 0.) THEN ! STAR
        !$OMP PARALLEL DO default(shared) schedule(static) reduction(+:n_star)
        DO i=1, n_raw
          IF(rint(i,2) .EQ. 2) n_star = n_star + 1
        ENDDO
        !$OMP END PARALLEL DO
      ELSE IF (horg .LT. 0.) THEN !DM
        !$OMP PARALLEL DO default(shared) schedule(static) reduction(+:n_star)
        DO i=1, n_raw
          IF(rint(i,2) .EQ. 1) THEN
            IF(ABS(rdbl(i,7) - dmp_mass)/dmp_mass .LT. 1e-5) &
                n_star = n_star + 1
          ENDIF
        ENDDO
        !$OMP END PARALLEL DO
      ENDIF

      RETURN
      END

!!!!!
!! GET PARTICLE NUM_YZiCS
!!!!!
      SUBROUTINE GET_PTCL_NUM_YZiCS(age, n_raw, n_star, n_thread, &
        horg, dmp_mass)

      IMPLICIT NONE
      INTEGER(KIND=4) n_raw, n_star, n_thread
      REAL(KIND=8) age(n_raw,9), dmp_mass

      INTEGER(KIND=4) i, j, k, horg

      CALL OMP_SET_NUM_THREADS(n_thread)
      n_star = 0

      IF(horg .GT. 0) THEN !STAR
        !$OMP PARALLEL DO default(shared) schedule(static) reduction(+:n_star)
        DO i=1, n_raw
          IF(age(i,8) .LT. 0) n_star = n_star + 1
        ENDDO
        !$OMP END PARALLEL DO
      ELSE IF (horg .LT. 0) THEN
        !$OMP PARALLEL DO default(shared) schedule(static) reduction(+:n_star)
        DO i=1, n_raw
          IF(age(i,8) .EQ. 0) THEN
           IF(ABS(age(i,7) - dmp_mass)/dmp_mass .LT. 1e-5) &
             n_star = n_star + 1
          ENDIF
        ENDDO
        !$OMP END PARALLEL DO
      ENDIF


      END subroutine

!!!!!
!! RD_PART
!!!!!
      SUBROUTINE RD_PART(dir_raw, dom_list, &
                n_dom, n_raw, n_str, n_snap, &
                rdbl, rint, longint)
      IMPLICIT NONE
      INTEGER(KIND=4) n_dom, n_raw, n_str, n_snap
      INTEGER(KIND=4) dom_list(n_dom), longint
      CHARACTER*(n_str) dir_raw
      REAL(KIND=8) rdbl(n_raw,9)
      INTEGER(KIND=8) rint(n_raw,2)

!!!!! Local variables

      CHARACTER*(100) fname, snap, domnum
      INTEGER(KIND=4) uout, icpu, nbody
      INTEGER(KIND=4) i, j, k, nn
      REAL(KIND=8), ALLOCATABLE, DIMENSION(:) :: dum_dbl
      INTEGER(KIND=8), ALLOCATABLE, DIMENSION(:) :: dum_int_ll
      INTEGER(KIND=4), ALLOCATABLE, DIMENSION(:) :: dum_int
      INTEGER(KIND=1), ALLOCATABLE, DIMENSION(:) :: dum_int_byte

      nn = 0
      DO k=1, n_dom
        WRITE(snap, '(I5.5)') n_snap
        WRITE(domnum, '(I5.5)') dom_list(k)
        fname = TRIM(dir_raw)//'output_'//TRIM(snap)//'/part_'//&
          TRIM(snap)//'.out'//TRIM(domnum)

        open(newunit=uout, file=fname, form='unformatted', status='old')
        read(uout); read(uout); read(uout) nbody; read(uout)
        read(uout); read(uout); read(uout); read(uout)

        ALLOCATE(dum_dbl(1:nbody))
        ALLOCATE(dum_int_ll(1:nbody))
        ALLOCATE(dum_int(1:nbody))
        ALLOCATE(dum_int_byte(1:nbody))

        Do j=1, 3                         !! Position
          read(uout) dum_dbl
          Do i=1, nbody
            rdbl(i+nn,j) = dum_dbl(i)
          ENDDO
        ENDDO

        Do j=1, 3                         !! Velocity
          read(uout) dum_dbl
          Do i=1, nbody
            rdbl(i+nn,j+3) = dum_dbl(i)
          ENDDO
        ENDDO

        read(uout) dum_dbl                !! Mass
        Do i=1, nbody
          rdbl(i+nn,7) = dum_dbl(i)
        ENDDO

        IF(longint .ge. 10) THEN
          read(uout) dum_int_ll
          Do i=1, nbody
            rint(i+nn,1) = dum_int_ll(i)
          ENDDO
        ELSE
          read(uout) dum_int
          Do i=1, nbody
            rint(i+nn,1) = dum_int(i)
          ENDDO
        ENDIF

        read(uout) dum_int

        read(uout) dum_int_byte
        Do i=1, nbody
          rint(i+nn,2) = dum_int_byte(i)
          IF(rint(i+nn,2) .gt. 100) rint(i+nn,2) = rint(i+nn,2) - 255
        ENDDO

        read(uout) dum_int_byte

        read(uout) dum_dbl                !! Age
        Do i=1, nbody
          rdbl(i+nn,8) = dum_dbl(i)
        ENDDO

        read(uout) dum_dbl                !! Metallicity
        Do i=1, nbody
          rdbl(i+nn,9) = dum_dbl(i)
        ENDDO

        DEALLOCATE(dum_dbl, dum_int_ll)
        DEALLOCATE(dum_int, dum_int_byte)
        nn = nn + nbody
        CLOSE(uout)
      ENDDO

      END subroutine

!!!!!
!! RD_PART_YZiCS
!!!!!
      SUBROUTINE RD_PART_YZiCS(dir_raw, dom_list, &
                n_dom, n_raw, n_str, n_snap, &
                rdbl, rint, longint)
      IMPLICIT NONE
      INTEGER(KIND=4) n_dom, n_raw, n_str, n_snap
      INTEGER(KIND=4) dom_list(n_dom), longint
      CHARACTER*(n_str) dir_raw
      REAL(KIND=8) rdbl(n_raw,9)
      INTEGER(KIND=8) rint(n_raw,2)

!!!!! Local variables

      CHARACTER*(100) fname, snap, domnum
      INTEGER(KIND=4) uout, icpu, nbody
      INTEGER(KIND=4) i, j, k, nn
      REAL(KIND=8), ALLOCATABLE, DIMENSION(:) :: dum_dbl
      INTEGER(KIND=8), ALLOCATABLE, DIMENSION(:) :: dum_int_ll
      INTEGER(KIND=4), ALLOCATABLE, DIMENSION(:) :: dum_int
      INTEGER(KIND=1), ALLOCATABLE, DIMENSION(:) :: dum_int_byte

      nn = 0
      DO k=1, n_dom
        WRITE(snap, '(I5.5)') n_snap
        WRITE(domnum, '(I5.5)') dom_list(k)
        fname = TRIM(dir_raw)//'output_'//TRIM(snap)//'/part_'//&
          TRIM(snap)//'.out'//TRIM(domnum)

        open(newunit=uout, file=fname, form='unformatted', status='old')
        read(uout); read(uout); read(uout) nbody; read(uout)
        read(uout); read(uout); read(uout); read(uout)

        ALLOCATE(dum_dbl(1:nbody))
        ALLOCATE(dum_int_ll(1:nbody))
        ALLOCATE(dum_int(1:nbody))
        ALLOCATE(dum_int_byte(1:nbody))

        Do j=1, 3                         !! Position
          read(uout) dum_dbl
          Do i=1, nbody
            rdbl(i+nn,j) = dum_dbl(i)
          ENDDO
        ENDDO

        Do j=1, 3                         !! Velocity
          read(uout) dum_dbl
          Do i=1, nbody
            rdbl(i+nn,j+3) = dum_dbl(i)
          ENDDO
        ENDDO

        read(uout) dum_dbl                !! Mass
        Do i=1, nbody
          rdbl(i+nn,7) = dum_dbl(i)
        ENDDO

        IF(longint .ge. 10) THEN          !! ID
          read(uout) dum_int_ll
          Do i=1, nbody
            rint(i+nn,1) = dum_int_ll(i)
          ENDDO
        ELSE
          read(uout) dum_int
          Do i=1, nbody
            rint(i+nn,1) = dum_int(i)
          ENDDO
        ENDIF

        read(uout) dum_int                !! LEVEL

        !read(uout) dum_int_byte
        !Do i=1, nbody
        !  rint(i+nn,2) = dum_int_byte(i)
        !  IF(rint(i+nn,2) .gt. 100) rint(i+nn,2) = rint(i+nn,2) - 255
        !ENDDO

        !read(uout) dum_int_byte

        read(uout) dum_dbl                !! Age
        Do i=1, nbody
          rdbl(i+nn,8) = dum_dbl(i)
        ENDDO

        read(uout) dum_dbl                !! Metallicity
        Do i=1, nbody
          rdbl(i+nn,9) = dum_dbl(i)
        ENDDO

        DEALLOCATE(dum_dbl, dum_int_ll)
        DEALLOCATE(dum_int, dum_int_byte)
        nn = nn + nbody
        CLOSE(uout)
      ENDDO

      END subroutine

!!!!!
!! GET NBODY
!!!!!
      SUBROUTINE RD_PART_NBODY(dir_raw, dom_list, &
                n_dom, n_raw, n_str, n_snap)

      Implicit none
      Integer(kind=4) n_raw, n_dom, n_str, n_snap, dom_list(n_dom)
      Character*(n_str) dir_raw

!!!!! Local variables
      Character*(100) fname, snap, domnum
      Integer(kind=4) uout, icpu, nbody
      INTEGER(kind=4) i, j

      n_raw = 0
      DO i=1, n_dom
        write(snap, '(I5.5)') n_snap
        write(domnum, '(I5.5)') dom_list(i)
        fname = TRIM(dir_raw)//'output_'//TRIM(snap)//'/part_'//&
          TRIM(snap)//'.out'//TRIM(domnum)

        open(newunit=uout, file=fname, form='unformatted', status='old')
        read(uout)
        read(uout)
        read(uout) nbody
        close(uout)

        n_raw = n_raw + nbody
      ENDDO

      END subroutine
!!!!!
!! QUICK SORT
!!!!!
! quicksort.f -*-f90-*-
! Author: t-nissie
! License: GPLv3
! Gist: https://gist.github.com/t-nissie/479f0f16966925fa29ea
!!
      recursive subroutine quicksort(a, nn, first, last)
        implicit none
        integer*8  a(nn,2), x, t, n
        integer first, last
        integer i, j, nn
      
        x = a( (first+last) / 2 , 1)
        i = first
        j = last
        do
           do while (a(i,1) < x)
              i=i+1
           end do
           do while (x < a(j,1))
              j=j-1
           end do
           IF(i .GE. j) exit
           t = a(i,1);  a(i,1) = a(j,1);  a(j,1) = t
           n = a(i,2);  a(i,2) = a(j,2);  a(j,2) = n
           i=i+1
           j=j-1
        end do
        if (first < i-1) call quicksort(a, nn, first, i-1)
        if (j+1 < last)  call quicksort(a, nn, j+1, last)
        return
      end subroutine quicksort
END MODULE get_ptcl_py
