!module
module find_domain_py
      INTEGER(KIND=4), DIMENSION(:, :), ALLOCATABLE :: dom_list
contains

!234567
      subroutine find_domain(xc, yc, zc, rr, hindex, &
                      larr, darr)

      use omp_lib
      Implicit none
      Integer(Kind=4), DIMENSION(:), INTENT(IN) :: larr
      Real(kind=8), DIMENSION(:), INTENT(IN) :: darr

      REAL(KIND=8), DIMENSION(:), INTENT(IN) :: xc, yc, zc, rr
      REAL(KIND=8), DIMENSION(:,:), INTENT(IN) :: hindex
      !Real(kind=8) xc(larr(1)), yc(larr(1)), zc(larr(1)), rr(larr(1))
      !REAL(kind=8) hindex(larr(2),2)
      REAL(KIND=8) rfact
      Integer(kind=4) levmax
      !Integer(kind=4) dom_list(larr(1),larr(2))

!Local Variables

      Integer(kind=4) i, j, k, impi
      Integer(kind=4) n_ptcl, n_mpi, n_thread, n_dim, n_dom
      INTEGER(KIND=4) lmin, bit_length, maxdom, ncpu_read
      INTEGER(KIND=4) imin, imax, jmin, jmax, kmin, kmax
      REAL(KIND=8) dmax, deltax, dkey, bounding(8)
      REAL(KIND=8) bounding_min(8), bounding_max(8), order_min
      REAL(KIND=8) xmin, xmax, ymin, ymax, zmin, zmax
      INTEGER(KIND=4) idom(8), jdom(8), kdom(8)
      INTEGER(KIND=4) cpu_min(8), cpu_max(8), cpu_list(larr(2))
      LOGICAL cpu_read(larr(2))

      n_ptcl    = larr(1)
      n_mpi     = larr(2)
      n_thread  = larr(3)
      n_dim     = 3
      levmax    = larr(4)

      rfact     = darr(1)
      IF(ALLOCATED(dom_list)) DEALLOCATE(dom_list)
      ALLOCATE(dom_list(1:n_ptcl, 1:n_mpi))
      dom_list = -1

      call omp_set_num_threads(n_thread)

      DO i=1, n_ptcl
        xmin = MAX(xc(i) - rfact*rr(i), 0.0)
        xmax = MIN(xc(i) + rfact*rr(i), 1.0)
        ymin = MAX(yc(i) - rfact*rr(i), 0.0)
        ymax = MIN(yc(i) + rfact*rr(i), 1.0)
        zmin = MAX(zc(i) - rfact*rr(i), 0.0)
        zmax = MIN(zc(i) + rfact*rr(i), 1.0)
        dmax = MAX(xmax-xmin,ymax-ymin,zmax-zmin)
        DO j=1, levmax
          deltax=0.5d0**j
          IF(deltax.LT.dmax)EXIT
        ENDDO

        lmin = j
        bit_length = lmin-1
        maxdom = 2**bit_length
        imin=0; imax=0; jmin=0; jmax=0; kmin=0; kmax=0
        IF(bit_length>0) THEN
                imin = int(xmin*dble(maxdom))
                imax = imin+1
                jmin = int(ymin*dble(maxdom))
                jmax = jmin+1
                kmin = int(zmin*dble(maxdom))
                kmax = kmin+1
         ENDIF

         dkey=(dble(2**(levmax+1)/dble(maxdom)))**n_dim
         n_dom = 1
         IF(bit_length>0) n_dom = 8
         idom(1)=imin; idom(2)=imax
         idom(3)=imin; idom(4)=imax
         idom(5)=imin; idom(6)=imax
         idom(7)=imin; idom(8)=imax
         jdom(1)=jmin; jdom(2)=jmin
         jdom(3)=jmax; jdom(4)=jmax
         jdom(5)=jmin; jdom(6)=jmin
         jdom(7)=jmax; jdom(8)=jmax
         kdom(1)=kmin; kdom(2)=kmin
         kdom(3)=kmin; kdom(4)=kmin
         kdom(5)=kmax; kdom(6)=kmax
         kdom(7)=kmax; kdom(8)=kmax

         DO k=1,n_dom
            if(bit_length>0)then
               call find_hilbert(idom(k),jdom(k),kdom(k),bounding(1),bit_length,1)
               order_min=bounding(1)
            else
               order_min=0.0d0
            endif
            bounding_min(k)=(order_min)*dkey
            bounding_max(k)=(order_min+1.0D0)*dkey
          ENDDO

          cpu_min=0; cpu_max=0

          do impi=1,n_mpi
             do k=1,n_dom
                if (   hindex(impi,1).le.bounding_min(k).and.&
                     & hindex(impi,2).gt.bounding_min(k))then
                   cpu_min(k)=impi
                endif
                if (   hindex(impi,1).lt.bounding_max(k).and.&
                     & hindex(impi,2).ge.bounding_max(k))then
                   cpu_max(k)=impi
                endif
             end do
          end do

          cpu_read = .FALSE.
          cpu_list = 0
          ncpu_read = 0
          do k=1,n_dom
             do j=cpu_min(k),cpu_max(k)
                if(.not. cpu_read(j))then
                   ncpu_read=ncpu_read+1
                   cpu_list(ncpu_read)=j
                   cpu_read(j)=.TRUE.
                endif
             enddo
          enddo

          DO j=1, ncpu_read
            dom_list(i,cpu_list(j)) = 1
          ENDDO
      ENDDO
      !!$OMP PARALLEL DO default(shared) private(bnd, cpu, dum_int, j, cpu_list, ind_dom, bounding, dkey, order_min) schedule(static)
      !do i=1, n_ptcl
      !  Call find_dom(xc(i), yc(i), zc(i), rr(i), levmax, &
      !          n_dim, bnd, dum_int(1), ind_dom, dum_int(4), &
      !          rfact, dkey)

      !  Do j=1, dum_int(1)
      !    if(dum_int(4)>0)then
      !      Call find_hilbert(ind_dom(j,1),ind_dom(j,2),ind_dom(j,3), &
      !          bounding(1),dum_int(4),1)
      !      order_min=bounding(1)
      !    else
      !      order_min=0.0d0
      !    endif
      !     bnd(j,1)=(order_min)*dkey
      !     bnd(j,2)=(order_min+1.0D0)*dkey
      !  Enddo

      !  dum_int(2) = 0
      !  Call find_cpu(bnd, cpu, n_mpi, dum_int(1), hindex, cpu_list, dum_int(2))

      !  Do j=1, dum_int(2) - 1
      !      dom_list(i,cpu_list(j)) = 1
      !  Enddo

      !enddo
      !!$OMP END PARALLEL DO
      
      end subroutine find_domain
      !!!!!
      !!!!!
      
      subroutine find_cpu(bnd, cpu, n_mpi, n_dom, hindex, cpu_list, ncpu_read)

      Integer(kind=4) n_dom, n_mpi

      Real(kind=8) bnd(8,2)
      Real(kind=8) hindex(n_mpi,2)
      Integer(kind=4) cpu(8,2)

      Integer(kind=4) cpu_read(n_mpi), cpu_list(n_mpi)
      Integer(kind=4) i, j, k, ncpu_read

      Do i=1, n_mpi
        cpu_list(i) = -1
        cpu_read(i) = -1
      Enddo

      Do i=1, n_mpi
        Do j=1, n_dom
          if(hindex(i,1) .le. bnd(j,1) .and. &
                  hindex(i,2) .gt. bnd(j,1)) cpu(j,1) = i

          if(hindex(i,1) .lt. bnd(j,2) .and. &
                  hindex(i,2) .ge. bnd(j,2)) cpu(j,2) = i

          cpu(j,1) = MAX(cpu(j,1), 1)
          cpu(j,2) = MIN(cpu(j,2), n_mpi)
        Enddo
      Enddo

      ncpu_read = 1
      Do i=1, n_dom
        Do j=cpu(i,1), cpu(i,2)
          if(cpu_read(j) .lt. 0) then
            cpu_list(ncpu_read) = j
            cpu_read(j) = 1
            ncpu_read = ncpu_read + 1
          Endif
        Enddo
      Enddo

      end subroutine find_cpu

      !!!!!
      !!!!!
      !!!!!
      subroutine find_dom(xc, yc, zc, rr, levmax, ndim, bnd, ndom, &
                ind_dom, bit_length, rfact, dkey)
      IMPLICIT NONE
      Real(kind=8) xc, yc, zc, rr, rfact
      Real(kind=4) box(3,2), dmax, dx
      Real(kind=8) dkey
      Real(kind=8) bounding_min(8),bounding_max(8), bnd(8,2)
      Integer levmax, lmin, bit_length, maxdom, ind(3,2), ndom, ndim
      Integer idom(8), jdom(8), kdom(8), ind_dom(8,3)
      Integer i, j, k

      box(1,1) = MAX(xc - rfact*rr, 0.0)
      box(1,2) = MIN(xc + rfact*rr, 1.0)
      box(2,1) = MAX(yc - rfact*rr, 0.0)
      box(2,2) = MIN(yc + rfact*rr, 1.0)
      box(3,1) = MAX(zc - rfact*rr, 0.0)
      box(3,2) = MIN(zc + rfact*rr, 1.0)

      dmax = max(box(1,2) - box(1,1), box(2,2) - box(2,1), &
             box(3,2) - box(3,1))
      dmax = MIN(dmax, 0.49)

      do j=1, levmax
        dx = 0.5d0**j
        if(dx.lt.dmax)exit
      enddo

      lmin = j; bit_length = lmin - 1
      !maxdom = int(2.d0**bit_length)
      maxdom = 2**bit_length

      ind(1,1)=0; ind(1,2)=0; ind(2,1)=0; ind(2,2)=0; ind(3,1)=0; ind(3,2)=0

      if(bit_length .gt. 0) then
        ind(1,1) = int(box(1,1) * dble(maxdom))
        ind(1,2) = ind(1,1) + 1
        ind(2,1) = int(box(2,1) * dble(maxdom))
        ind(2,2) = ind(2,1) + 1
        ind(3,1) = int(box(3,1) * dble(maxdom))
        ind(3,2) = ind(3,1) + 1
      endif

      dkey=2.d0**(levmax+1) / dble(maxdom)
      dkey=dkey**ndim

      ndom=1
      if(bit_length>0)ndom=8
      idom(1)=ind(1,1); idom(2)=ind(1,2)
      idom(3)=ind(1,1); idom(4)=ind(1,2)
      idom(5)=ind(1,1); idom(6)=ind(1,2)
      idom(7)=ind(1,1); idom(8)=ind(1,2)
      jdom(1)=ind(2,1); jdom(2)=ind(2,1)
      jdom(3)=ind(2,2); jdom(4)=ind(2,2)
      jdom(5)=ind(2,1); jdom(6)=ind(2,1)
      jdom(7)=ind(2,2); jdom(8)=ind(2,2)
      kdom(1)=ind(3,1); kdom(2)=ind(3,1)
      kdom(3)=ind(3,1); kdom(4)=ind(3,1)
      kdom(5)=ind(3,2); kdom(6)=ind(3,2)
      kdom(7)=ind(3,2); kdom(8)=ind(3,2)

      Do i=1, 8
        ind_dom(i,1) = idom(i)
        ind_dom(i,2) = jdom(i)
        ind_dom(i,3) = kdom(i)
      Enddo

      end subroutine find_dom

      !!!!!
      !!!!!
      !!!!!
      subroutine find_hilbert(x,y,z,order,bit_length,npoint)
      implicit none
      
      integer     ,INTENT(IN)                   ::bit_length,npoint
      integer     ,INTENT(IN), dimension(1:npoint)::x,y,z
      real(kind=8),INTENT(OUT),dimension(1:npoint)::order
      
      logical,dimension(0:3*bit_length-1)::i_bit_mask
      logical,dimension(0:1*bit_length-1)::x_bit_mask,y_bit_mask,z_bit_mask
      integer,dimension(0:7,0:1,0:11)::state_diagram
      integer::i,ip,cstate,nstate,b0,b1,b2,sdigit,hdigit
      
      if(bit_length>bit_size(bit_length))then
         write(*,*)'Maximum bit length=',bit_size(bit_length)
         write(*,*)'stop in find_hilbert'
         stop
      endif
      
      state_diagram = RESHAPE( (/   1, 2, 3, 2, 4, 5, 3, 5,&
                                &   0, 1, 3, 2, 7, 6, 4, 5,&
                                &   2, 6, 0, 7, 8, 8, 0, 7,&
                                &   0, 7, 1, 6, 3, 4, 2, 5,&
                                &   0, 9,10, 9, 1, 1,11,11,&
                                &   0, 3, 7, 4, 1, 2, 6, 5,&
                                &   6, 0, 6,11, 9, 0, 9, 8,&
                                &   2, 3, 1, 0, 5, 4, 6, 7,&
                                &  11,11, 0, 7, 5, 9, 0, 7,&
                                &   4, 3, 5, 2, 7, 0, 6, 1,&
                                &   4, 4, 8, 8, 0, 6,10, 6,&
                                &   6, 5, 1, 2, 7, 4, 0, 3,&
                                &   5, 7, 5, 3, 1, 1,11,11,&
                                &   4, 7, 3, 0, 5, 6, 2, 1,&
                                &   6, 1, 6,10, 9, 4, 9,10,&
                                &   6, 7, 5, 4, 1, 0, 2, 3,&
                                &  10, 3, 1, 1,10, 3, 5, 9,&
                                &   2, 5, 3, 4, 1, 6, 0, 7,&
                                &   4, 4, 8, 8, 2, 7, 2, 3,&
                                &   2, 1, 5, 6, 3, 0, 4, 7,&
                                &   7, 2,11, 2, 7, 5, 8, 5,&
                                &   4, 5, 7, 6, 3, 2, 0, 1,&
                                &  10, 3, 2, 6,10, 3, 4, 4,&
                                &   6, 1, 7, 0, 5, 2, 4, 3 /), &
                                & (/8 ,2, 12 /) )
      
      do ip=1,npoint
      
         ! convert to binary
         do i=0,bit_length-1
            x_bit_mask(i)=btest(x(ip),i)
            y_bit_mask(i)=btest(y(ip),i)
            z_bit_mask(i)=btest(z(ip),i)
         enddo
      
         ! interleave bits
         do i=0,bit_length-1
            i_bit_mask(3*i+2)=x_bit_mask(i)
            i_bit_mask(3*i+1)=y_bit_mask(i)
            i_bit_mask(3*i  )=z_bit_mask(i)
         end do
      
         ! build Hilbert ordering using state diagram
         cstate=0
         do i=bit_length-1,0,-1
            b2=0 ; if(i_bit_mask(3*i+2))b2=1
            b1=0 ; if(i_bit_mask(3*i+1))b1=1
            b0=0 ; if(i_bit_mask(3*i  ))b0=1
            sdigit=b2*4+b1*2+b0
            nstate=state_diagram(sdigit,0,cstate)
            hdigit=state_diagram(sdigit,1,cstate)
            i_bit_mask(3*i+2)=btest(hdigit,2)
            i_bit_mask(3*i+1)=btest(hdigit,1)
            i_bit_mask(3*i  )=btest(hdigit,0)
            cstate=nstate
         enddo
      
         ! save Hilbert key as double precision real
         order(ip)=0.
         do i=0,3*bit_length-1
            b0=0 ; if(i_bit_mask(i))b0=1
            order(ip)=order(ip)+dble(b0)*dble(2)**i
         end do
      end do
      end subroutine find_hilbert
end module find_domain_py
