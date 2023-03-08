MODULE js_getpt

      REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: pot

      !!----- Tree related
      TYPE nodetype
        INTEGER(KIND=4) :: id, bstart, bend, ncount
        INTEGER(KIND=4) :: splitdim, splitind, leaf
        REAL(KIND=8) :: bnd(6,2), cen(6), mass, splitval
        INTEGER(KIND=4) :: numnode, level
        INTEGER(KIND=4) :: parent, sibling, left, right
      END TYPE nodetype

      TYPE dat
        REAL(KIND=8) :: pos(6), mm
      END TYPE dat

      TYPE infotype
        INTEGER(KIND=4) :: bsize, ndim, npart
        INTEGER(KIND=4) :: dtype, vtype
      END TYPE infotype
      !!-----
CONTAINS
!!---------------------------------------------
!! MAIN
!!---------------------------------------------
!234567
      SUBROUTINE js_getpt_ft(larr, darr, pos, mm)

      USE omp_lib

      IMPLICIT NONE

      REAL(KIND=8), DIMENSION(:) :: darr
      INTEGER(KIND=4), DIMENSION(:) :: larr

      REAL(KIND=8), DIMENSION(:,:) :: pos
      REAL(KIND=8), DIMENSION(:) :: mm

      TYPE(nodetype), DIMENSION(:), ALLOCATABLE :: root
      !!-----
      !! LOCAL VARIABLES
      !!-----
      INTEGER(KIND=4) i, j, k, l, m

      INTEGER(KIND=4) n_ptcl, n_thread, n_dim, p_type, n_leaf
      INTEGER(KIND=4) d_type, v_type, bsize
      INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: orgind
      INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: recind
      REAL(KIND=8) Gconst, dummy, dx, time(10)
      REAL(KIND=8) dummy_v(8), bnd(8,3)
      INTEGER(KIND=4) bs, be, e_type, tonoff
      TYPE(nodetype), DIMENSION(:), ALLOCATABLE :: lf

      n_ptcl    = larr(1)
      n_dim     = larr(2)
      n_thread  = larr(3)
      p_type    = larr(4)
      e_type    = larr(5)
      tonoff    = larr(6)

      d_type    = larr(11)      ! DIMENSION TYPE
      v_type    = larr(12)      ! VALUE TYP
      bsize     = larr(13)
      Gconst    = darr(1)

      CALL OMP_SET_NUM_THREADS(n_thread)

      IF(ALLOCATED(pot)) DEALLOCATE(pot)
      ALLOCATE(pot(1:n_ptcl))
      pot = 0.
      !!-----
      !! GET TREE
      !!-----
      IF(ALLOCATED(orgind)) DEALLOCATE(orgind)
      IF(ALLOCATED(recind)) DEALLOCATE(recind)
      ALLOCATE(orgind(1:n_ptcl))
      ALLOCATE(recind(1:n_ptcl))
      DO i=1, n_ptcl
        orgind(i) = i
        recind(i) = i
      ENDDO

      time(1)   = omp_get_wtime()

      IF(ALLOCATED(root)) DEALLOCATE(root)
      root = js_kdtree_mktree(pos, mm, orgind, bsize, d_type, v_type)

      time(2)   = omp_get_wtime()
      !!-----
      !! GET LEAF ONLY
      !!-----
      n_leaf    = js_kdtree_getleafnum(root)
      IF(ALLOCATED(lf)) DEALLOCATE(lf)
      ALLOCATE(lf(1:n_leaf))

      CALL js_kdtree_getleaf(root, lf)

      !!-----
      !! COMPUTE POTENTIAL
      !!-----

      !!----- FOR DS TEST
      !!$OMP PARALLEL DO default(shared) &
      !!$OMP & private(j, dx, k) schedule(static)
      !DO i=1, 10000
      !  PRINT *, i
      !  DO j=1, n_ptcl
      !    IF(i .EQ. j)CYCLE
      !    dx = 0.
      !    DO k=1, n_dim
      !      dx = dx + ( pos(i,k) - pos(j,k) )**2
      !    ENDDO
      !    dx = dx**0.5

      !    pot(i) = pot(i) + (-Gconst) * mm(j) / dx
      !  ENDDO
      !ENDDO
      !!$OMP END PARALLEL DO
      !RETURN


      !!----- Considering Particle position
      IF(e_type .EQ. 1) THEN
        !$OMP PARALLEL DO default(shared) &
        !$OMP & private(bs, be, dx, j, k, l) &
        !$OMP & schedule(static)
        DO i=1, n_ptcl
          pot(i) = 0.

          IF(p_type .EQ. 0) THEN !! mesh to mesh only

            DO j=1, n_leaf
              bs = lf(j)%bstart
              be = lf(j)%bend

              dx = 0.
              DO k=1, n_dim
                dx = dx + ( lf(j)%cen(k) - pos(i,k) )**2
              ENDDO
              dx = dx**0.5

              IF(i .GE. bs .AND. i .LE. be) THEN
                pot(i) = pot(i) + (-Gconst) * (lf(j)%mass - mm(i))/dx
              ELSE
                pot(i) = pot(i) + (-Gconst) * lf(j)%mass/dx
              ENDIF
            ENDDO

          ENDIF

          IF(p_type .EQ. 1) THEN !! particle to mesh
            DO j=1, n_leaf
              bs = lf(j)%bstart
              be = lf(j)%bend

              IF(i .GE. bs .AND. i .LE. be) THEN
                DO k=bs, be
                  IF(k .EQ. i) CYCLE

                  dx = 0.
                  DO l=1, n_dim
                    dx = dx + ( pos(i,l) - pos(k,l) )**2
                  ENDDO
                  dx = dx**0.5

                  pot(i) = pot(i) + (-Gconst) * mm(k)/dx
                ENDDO
              ELSE
                dx = 0.
                DO k=1, n_dim
                  dx = dx + ( pos(i,k) - lf(j)%cen(k) )**2
                ENDDO
                dx = dx**0.5

                pot(i) = pot(i) + (-Gconst) * lf(j)%mass/dx
              ENDIF
            ENDDO
          ENDIF
        ENDDO
        !$OMP END PARALLEL DO
      ENDIF

      IF(e_type .EQ. 0) THEN
        !----- Particle position is fixed to the center of containing
        !        node - cost efficient
        !j, dummy, dx, bs, be, k
        !$OMP PARALLEL DO default(shared) &
        !$OMP & private(dummy, dummy_v, bnd, j, k, l, bs, be, dx) &
        !$OMP & schedule(static)
        DO i=1, n_leaf

          ! Monopole term
          dummy_v = 0.
          bnd   = js_getpt_ft_getbnd(lf(i))

          DO j=1, n_leaf
            IF(i .EQ. j) CYCLE

            DO k=1, 8
              dx = js_getpt_ft_dist(lf(j)%cen(1:3),bnd(k,:))
              IF(dx .EQ. 0) CYCLE
                dummy_v(k) = dummy_v(k) + (-Gconst) * lf(j)%mass / dx
            ENDDO
          ENDDO

          ! Direct term
          bs      = lf(i)%bstart
          be      = lf(i)%bend

          DO j=bs, be
            dummy = js_getpt_ft_interpole(bnd, pos(j,1:3), dummy_v)
            pot(j) = pot(j) + dummy

            IF(p_type .EQ. 0) THEN !! mesh to mesh only
              dx = 0.
              DO k=1, n_dim
                dx = dx + ( lf(i)%cen(k) - pos(j,k) )**2
              ENDDO
              dx = dx**0.5
              IF(dx .EQ. 0) CYCLE
              pot(j) = pot(j) + (-Gconst) * (lf(i)%mass - mm(j)) / dx

            ENDIF

            IF(p_type .EQ. 1) THEN !! particle to mesh
              DO k=bs, be
                IF(j .EQ. k) CYCLE

                dx = 0.
                DO l=1, n_dim
                  dx = dx + ( pos(j,l) - pos(k,l) )**2
                ENDDO
                dx = dx**0.5
                IF(dx .EQ. 0) CYCLE
                pot(j) = pot(j) + (-Gconst) * mm(k) / dx
              ENDDO
            ENDIF
          ENDDO
        ENDDO
        !$OMP END PARALLEL DO
      ENDIF

      time(3)   = omp_get_wtime()

      !!-----
      !! REORDERING
      !!-----
      CALL QUICKSORT(orgind, recind, SIZE(orgind), 1, SIZE(orgind))

      DO i=1, n_dim
        pos(:,i) = pos(recind,i)
      ENDDO
      mm = mm(recind)
      pot = pot(recind)

      IF(tonoff .EQ. 1) THEN
        PRINT *, '%123123---------------'
        PRINT *, '        Wall-clock time Report'
        PRINT *, '        Tree in ', time(2) - time(1)
        PRINT *, '        Pot in ', time(3) - time(2)
        PRINT *, '%123123---------------'
      ENDIF
      DEALLOCATE(orgind, recind)
      DEALLOCATE(lf)
      DEALLOCATE(root)
      RETURN
CONTAINS
!!--------------------------------------------------
!! Util for computing Potential
!!--------------------------------------------------
      FUNCTION js_getpt_ft_getbnd(node) RESULT(bnd)
        IMPLICIT NONE
        TYPE(nodetype) node
        REAL(KIND=8) bnd(8,3)
        INTEGER(KIND=4) i, j

        DO i=1, 2
          j = 4*(i-1)
          bnd(j+1,1) = node%bnd(1,1)
          bnd(j+1,2) = node%bnd(2,1)
          bnd(j+1,3) = node%bnd(3,i)

          bnd(j+2,1) = node%bnd(1,1)
          bnd(j+2,2) = node%bnd(2,2)
          bnd(j+2,3) = node%bnd(3,i)

          bnd(j+3,1) = node%bnd(1,2)
          bnd(j+3,2) = node%bnd(2,2)
          bnd(j+3,3) = node%bnd(3,i)

          bnd(j+4,1) = node%bnd(1,2)
          bnd(j+4,2) = node%bnd(2,1)
          bnd(j+4,3) = node%bnd(3,i)
        ENDDO
        RETURN
      END FUNCTION js_getpt_ft_getbnd

      FUNCTION js_getpt_ft_dist(x0, x1) RESULT(dd)
        IMPLICIT NONE
        REAL(KIND=8) x0(3), x1(3)
        REAL(KIND=8) dd
        INTEGER(KIND=4) i

        dd = 0.
        DO i=1, 3
          dd = dd + ( x0(i) - x1(i) )**2
        ENDDO
        dd = dd**0.5

        RETURN
      END FUNCTION js_getpt_ft_dist

      FUNCTION js_getpt_ft_interpole(bnd, pos, dummy_v) RESULT(p)
        IMPLICIT NONE
        REAL(KIND=8) bnd(8,3), pos(3), dummy_v(8), p
        REAL(KIND=8) v1, v2, v0
        REAL(KIND=8) w1, w2, w0

        v1      = js_getpt_ft_lint(pos(2), bnd(1,2), bnd(2,2), dummy_v(1), dummy_v(2))
        v2      = js_getpt_ft_lint(pos(2), bnd(4,2), bnd(3,2), dummy_v(4), dummy_v(3))
        v0      = js_getpt_ft_lint(pos(1), bnd(1,1), bnd(3,1), v1, v2)

        w1      = js_getpt_ft_lint(pos(2), bnd(5,2), bnd(6,2), dummy_v(5), dummy_v(6))
        w2      = js_getpt_ft_lint(pos(2), bnd(8,2), bnd(7,2), dummy_v(8), dummy_v(7))
        w0      = js_getpt_ft_lint(pos(1), bnd(5,1), bnd(7,1), w1, w2)

        p       = js_getpt_ft_lint(pos(3), bnd(1,3), bnd(5,3), v0, w0)
        RETURN
      END FUNCTION js_getpt_ft_interpole

      FUNCTION js_getpt_ft_lint(x, x0, x1, v0, v1) RESULT(val)
        IMPLICIT NONE
        REAL(KIND=8) x, x0, x1, v0, v1, val
        REAL(KIND=8) dx

        dx = (x-x0) / (x1-x0)
        dx = MAX(dx,0d0)
        val= dx*v1 + (1d0 - dx)*v0
        RETURN
      END FUNCTION js_getpt_ft_lint
!!--------------------------------------------------
!! Main Routine for KD Tree
!!--------------------------------------------------
!!----- Tree Main routine
        FUNCTION js_kdtree_mktree(pos, mm, orgind, &
           bsize, dtype, vtype) RESULT(root)

        IMPLICIT NONE
        REAL(KIND=8), DIMENSION(:,:) :: pos
        REAL(KIND=8), DIMENSION(:) :: mm
        INTEGER(KIND=4), DIMENSION(:) :: orgind
        INTEGER(KIND=4), INTENT(IN) :: bsize, dtype, vtype

        INTEGER(KIND=4) :: ndim, npart, i
        INTEGER(KIND=4) :: numnode, level, bstart, bend
        TYPE(nodetype), DIMENSION(:), ALLOCATABLE :: root, root_dum
        TYPE(dat), DIMENSION(:), ALLOCATABLE :: part
        TYPE(infotype) info
        npart   = SIZE(pos(:,1))
        ndim    = SIZE(pos(1,:))

        info%bsize      = bsize
        info%ndim       = ndim
        info%npart      = npart
        info%dtype      = dtype
        info%vtype      = vtype

        !!----- ALLOCATE
        ALLOCATE(part(1:npart))

        DO i=1, ndim
          part%pos(i) = pos(:,i)
        ENDDO
        part%mm = mm(:)

        ALLOCATE(root(1:npart))
        ALLOCATE(root_dum(1:npart))

        !!----- ROOT NODE INFORMATION
        DO i=1, ndim
          root(1)%bnd(i,1) = js_kdtree_min(part%pos(i))
          root(1)%bnd(i,2) = js_kdtree_max(part%pos(i))
        ENDDO

        !!----- Build NODE
        numnode = 0
        level   = 1
        bstart  = 1
        bend    = npart

        CALL js_kdtree_buildnode(root, part, orgind, info, &
                numnode, bstart, bend, level)

        !!----- NEW INDEX
        DO i=1, ndim
          pos(:,i) = part%pos(i)
        ENDDO
        mm(:)   = part%mm

        DEALLOCATE(part)

        !!----- DEALLOCATE
        root_dum        = root
        DEALLOCATE(root)
        ALLOCATE(root(1:numnode))
        root    = root_dum(1:numnode)
        DEALLOCATE(root_dum)
        root(1)%numnode = numnode
      END FUNCTION js_kdtree_mktree

!!----- Build Node
      RECURSIVE SUBROUTINE js_kdtree_buildnode(node, part, orgind, info, &
                      numnode, bstart, bend, level)
        IMPLICIT NONE
        TYPE(nodetype), DIMENSION(:) :: node
        TYPE(dat), DIMENSION(:) :: part
        TYPE(infotype), INTENT(IN) :: info
        INTEGER(KIND=4), DIMENSION(:) :: orgind
        INTEGER(KIND=4) :: numnode, bstart, bend, level, level_up, nid, bs, be

        INTEGER(KIND=4) i, j, k

        !!----- SET NODE PROPs
        numnode = numnode + 1
        node(numnode)%id      = numnode
        node(numnode)%bstart  = bstart
        node(numnode)%bend    = bend
        node(numnode)%ncount  = bend-bstart+1
        node(numnode)%level   = level
        node(numnode)%leaf    = -1

        !!----- IS LEAF
        IF(node(numnode)%ncount .LE. info%bsize) THEN
          node(numnode)%mass  = js_kdtree_total(part(bstart:bend)%mm)

          DO i=1, info%ndim
            node(numnode)%cen(i) = js_kdtree_total( &
                    part(bstart:bend)%pos(i) * part(bstart:bend)%mm ) / &
                    js_kdtree_total( part(bstart:bend)%mm )
          ENDDO

          node(numnode)%leaf = 1
          RETURN
        ENDIF

        !!----- DETERMINE SPLITDIM
        node(numnode)%splitdim = js_kdtree_sdim(part, bstart, bend, info)

        !!----- DETERMIN SPLITVALUE
        CALL js_kdtree_sval(node(numnode), part, bstart, bend, orgind, info)

        !!----- BUILD Sons
        level_up        = level + 1
        nid     = numnode
        node(numnode)%left      = numnode + 1
        node(numnode+1)%parent  = nid
        node(node(nid)%left)%bnd        = node(nid)%bnd
        node(node(nid)%left)%bnd( (node(nid)%splitdim), 2) = node(nid)%splitval
        bs      = node(nid)%bstart
        be      = node(nid)%splitind - 1
        CALL js_kdtree_buildnode(node, part, orgind, info, &
                numnode, bs, be, level_up)

        level_up        = level + 1
        node(nid)%right = numnode + 1
        node(numnode+1)%parent = nid
        node(node(nid)%right)%bnd       = node(nid)%bnd
        node(node(nid)%right)%bnd( (node(nid)%splitdim), 1) = node(nid)%splitval

        bs      = node(nid)%splitind
        be      = node(nid)%bend
        CALL js_kdtree_buildnode(node, part, orgind, info, &
                numnode, bs, be, level_up)

        !!----- Set SONs
        !node%left       => left
        !node%right      => right
        !node%left       => lnode
        !node%right      => rnode

        !left%parent     => node
        !right%parent    => node

        !left%sibling    => right
        !right%sibling   => left

        !lnode%bnd     = node%bnd
        !rnode%bnd    = node%bnd

        !lnode%bnd(node%splitdim, 2)        = node%splitval
        !rnode%bnd(node%splitdim, 1)       = node%splitval

        !node(node(nid)%left)%bnd        = node(nid)%bnd
        !node(node(nid)%right)%bnd       = node(nid)%bnd

        !node(node(nid)%left)%bnd( (node(nid)%splitdim), 2) = node(nid)%splitval
        !node(node(nid)%right)%bnd( (node(nid)%splitdim), 1) = node(nid)%splitval

        RETURN
      END SUBROUTINE js_kdtree_buildnode

!!----- TREE SHAPE FTNs (split value)

      SUBROUTINE js_kdtree_sval(node, part, bstart, bend, orgind, info)
        IMPLICIT NONE
        TYPE(dat), DIMENSION(:) :: part
        INTEGER(KIND=4) bstart, bend
        INTEGER(KIND=4), DIMENSION(:) :: orgind
        TYPE(infotype) info
        TYPE(nodetype) node

        REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: tmp
        INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: ind
        INTEGER(KIND=4) i, j, k, sind

        IF(info%vtype .EQ. 0) THEN ! BALANCED
          ALLOCATE(tmp(1:(bend-bstart+1)))
          ALLOCATE(ind(1:(bend-bstart+1)))

          tmp     = part(bstart:bend)%pos(node%splitdim)
          DO i=1, SIZE(tmp)
            ind(i) = i
          ENDDO

          CALL js_kdtree_mediansort(tmp, ind, sind)
          ind = ind + bstart - 1
          sind = sind + bstart - 1
          part(bstart:bend) = part(ind)
          orgind(bstart:bend) = orgind(ind)

          node%splitval = part(sind)%pos(node%splitdim)
          node%splitind = sind

          DEALLOCATE(tmp)
          DEALLOCATE(ind)
        ELSE
          PRINT *, 'not implemented yet'
          STOP
        ENDIF
      END SUBROUTINE js_kdtree_sval

!!----- TREE SHAPE FTNs (split dimension)

      INTEGER(KIND=4) FUNCTION js_kdtree_sdim(part, bstart, bend, info)
        IMPLICIT NONE
        TYPE(dat), DIMENSION(:), INTENT(IN) :: part
        TYPE(infotype) :: info
        INTEGER(KIND=4) :: bstart, bend
        INTEGER(KIND=4) :: i, j, k
        REAL(KIND=8) :: dx, dx2

        js_kdtree_sdim = 1
        IF(info%dtype .EQ. 0) THEN ! MAX RANGE
          dx = 0.
          DO i=1,info%ndim
            dx2 = js_kdtree_max(part(bstart:bend)%pos(i)) - js_kdtree_min(part(bstart:bend)%pos(i))
            IF(dx2 .GE. dx) THEN
              dx = dx2
              js_kdtree_sdim = i
            ENDIF
          ENDDO
        ELSE
          PRINT *, 'not implemented yet'
          STOP
        ENDIF
      END FUNCTION js_kdtree_sdim

!!----- SIMPLE FTNs for KDTree
      !MIN
      REAL(KIND=8) FUNCTION js_kdtree_min(xx)
        IMPLICIT NONE
        REAL(KIND=8), DIMENSION(:), INTENT(IN) ::xx
        INTEGER(KIND=4) i

        js_kdtree_min = xx(1)
        DO i=2, SIZE(xx)
          IF(xx(i) .LE. js_kdtree_min) js_kdtree_min = xx(i)
        ENDDO
      END FUNCTION js_kdtree_min

      !MAX
      REAL(KIND=8) FUNCTION js_kdtree_max(xx)
        IMPLICIT NONE
        REAL(KIND=8), DIMENSION(:), INTENT(IN) :: xx
        INTEGER(KIND=4) i

        js_kdtree_max = xx(1)
        DO i=2, SIZE(xx)
          IF(xx(i) .GE. js_kdtree_max) js_kdtree_max = xx(i)
        ENDDO
      END FUNCTION js_kdtree_max

      !TOTAL
      REAL(KIND=8) FUNCTION js_kdtree_total(xx)
        IMPLICIT NONE
        REAL(KIND=8), DIMENSION(:), INTENT(IN) :: xx
        INTEGER(KIND=4) i

        js_kdtree_total = 0.
        DO i=1, SIZE(xx)
          js_kdtree_total = js_kdtree_total + xx(i)
        ENDDO
      END FUNCTION js_kdtree_total

      !SWAP FTN
      SUBROUTINE js_kdtree_swap_l(arr, i, j)
        IMPLICIT NONE
        INTEGER(KIND=4) i, j
        INTEGER(KIND=4), DIMENSION(:) :: arr
        INTEGER(KIND=4) dum

        dum = arr(i)
        arr(i) = arr(j)
        arr(j) = dum
      END SUBROUTINE js_kdtree_swap_l
      SUBROUTINE js_kdtree_swap_d(arr, i, j)
        IMPLICIT NONE
        INTEGER(KIND=4) i, j
        REAL(KIND=8), DIMENSION(:) :: arr
        REAL(KIND=8) dum

        dum = arr(i)
        arr(i) = arr(j)
        arr(j) = dum
      END SUBROUTINE js_kdtree_swap_d

!!----- MEDIAN SORT
      SUBROUTINE js_kdtree_mediansort(tmp, ind, sind)
        IMPLICIT NONE
        REAL(KIND=8), DIMENSION(:) :: tmp
        INTEGER(KIND=4), DIMENSION(:) :: ind

        INTEGER(KIND=4) nn, i0, i1, j0, j1, k, pivot_ind, sind
        REAL(KIND=8) x, pivot

        i0 = 1
        j0 = SIZE(tmp)
        k  = i0 + (j0-i0 + 1)/2

        DO WHILE(i0 .LT. j0)
          x = tmp(k)
          CALL js_kdtree_swap_d(tmp, j0, k)
          CALL js_kdtree_swap_l(ind, j0, k)

          pivot = tmp(k)
          i1    = i0 - 1
          j1    = j0

          DO WHILE (.TRUE.)
            DO WHILE(i1 .LT. j1)
              i1 = i1 + 1
              IF(tmp(i1) .GE. x) EXIT
            ENDDO

            DO WHILE(i1 .LT. j1)
              j1 = j1 - 1
              IF(tmp(j1) .LE. x) EXIT
            ENDDO

            CALL js_kdtree_swap_d(tmp, i1, j1)
            CALL js_kdtree_swap_l(ind, i1, j1)
            pivot = tmp(j1)
            pivot_ind = ind(j1)
            IF( j1 .LE. i1) EXIT
          ENDDO

          tmp(j1)     = tmp(i1)
          ind(j1)       = ind(i1)

          tmp(i1)     = tmp(j0)
          ind(j1)       = ind(j0)

          tmp(j0)     = pivot
          ind(j0)       = pivot_ind

          IF(i1 .GE. k) j0 = i1 - 1
          IF(i1 .LE. k) i0 = i1 + 1
        ENDDO
        sind = k
      END SUBROUTINE js_kdtree_mediansort

!!----- TREE CALL FTNs
       ! get number of leaf nodes
      RECURSIVE SUBROUTINE js_kdtree_getleafnum_wktree(node, nn, nid)
        TYPE(nodetype), DIMENSION(:) :: node
        INTEGER(KIND=4) nn, left, right, nid

        IF(node(nid)%leaf .GT. 0) THEN
          nn = nn + 1
          RETURN
        ELSE
          left = node(nid)%left
          CALL js_kdtree_getleafnum_wktree(node, nn, left)

          right = node(nid)%right
          CALL js_kdtree_getleafnum_wktree(node, nn, right)
        ENDIF
      END SUBROUTINE js_kdtree_getleafnum_wktree
      FUNCTION js_kdtree_getleafnum(node) RESULT(n_leaf)
        TYPE(nodetype), DIMENSION(:) :: node
        INTEGER(KIND=4) n_leaf, nid

        n_leaf = 0
        nid = 1
        CALL js_kdtree_getleafnum_wktree(node, n_leaf, nid)

        RETURN
      END FUNCTION js_kdtree_getleafnum

      ! GET LEAF ONLY
      RECURSIVE SUBROUTINE js_kdtree_getleaf_fill(node, lf, nid, lid)
        TYPE(nodetype), DIMENSION(:) :: node, lf
        INTEGER(KIND=4) nid, lid, left, right

        IF(node(nid)%leaf .GT. 0) THEN
          lf(lid) = node(nid)
          lid = lid + 1
        ELSE
          left = node(nid)%left
          right = node(nid)%right

          CALL js_kdtree_getleaf_fill(node, lf, left, lid)
          CALL js_kdtree_getleaf_fill(node, lf, right, lid)
        ENDIF
      END SUBROUTINE js_kdtree_getleaf_fill

      SUBROUTINE js_kdtree_getleaf(root, lf)
        TYPE(nodetype), DIMENSION(:) :: root, lf
        INTEGER(KIND=4) nid, lid

        nid = 1
        lid = 1
        CALL js_kdtree_getleaf_fill(root, lf, nid, lid)

      END SUBROUTINE js_kdtree_getleaf

!!--------------------------------------------------
!! Ftns for GET potential
!!--------------------------------------------------
      FUNCTION js_getpt_ft_d3d(node, i, j, ndim) RESULT(dx)
        IMPLICIT NONE
        TYPE(nodetype), DIMENSION(:) :: node
        INTEGER(KIND=4) i, j, k, ndim
        REAL(KIND=8) dx

        dx = 0.
        DO k=1, ndim
          dx = dx + (node(i)%cen(k) - node(j)%cen(k))**2
        ENDDO
        dx = dx**0.5
        RETURN
      END FUNCTION js_getpt_ft_d3d

!! QUICK SORT
!!!!!
! quicksort.f -*-f90-*-
! Author: t-nissie
! License: GPLv3
! Gist: https://gist.github.com/t-nissie/479f0f16966925fa29ea
!!
      recursive subroutine quicksort(a, b, nn, first, last)
        implicit none
        integer*4  a(nn), b(nn), x, t, n
        integer first, last
        integer i, j, nn

        x = a( (first+last) / 2 )
        i = first
        j = last
        do
           do while (a(i) < x)
              i=i+1
           end do
           do while (x < a(j))
              j=j-1
           end do
           IF(i .GE. j) exit
           t = a(i);  a(i) = a(j);  a(j) = t
           n = b(i); b(i) = b(j); b(j) = n
           i=i+1
           j=j-1
        end do
        if (first < i-1) call quicksort(a, b, nn, first, i-1)
        if (j+1 < last)  call quicksort(a, b, nn, j+1, last)
        return
      end subroutine quicksort
      END SUBROUTINE js_getpt_ft
      SUBROUTINE js_getpt_ft_free()
              IF(ALLOCATED(pot)) DEALLOCATE(pot)
      END SUBROUTINE

END MODULE
