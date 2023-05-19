MODULE js_getsfe
      REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: sfe, mach2, alpha, &
        t_ff, dum_sf, sig_s, sig_c
CONTAINS
!234567
      SUBROUTINE js_getsfe_ft(larr, darr, x, y, z, &
	     vx, vy, vz, dx, den, cs2)

      use omp_lib
      IMPLICIT NONE

      INTEGER(KIND=4), DIMENSION(:) :: larr
      REAL(KIND=8), DIMENSION(:) :: darr

      REAL(KIND=8), DIMENSION(:), INTENT(IN) :: x, y, z, dx, vx, vy, vz, &
        den, cs2

      !!-----
      !! LOCAL
      !!-----
      INTEGER(KIND=4) i, j, k, l, m
      INTEGER(KIND=4) n_cell, n_thread, ind(6)
      REAL(KIND=8) d_thres, factG, pi, ul, ur
      REAL(KIND=8) xl_d, xl_vx, xl_vy, xl_vz
      REAL(KIND=8) yl_d, yl_vx, yl_vy, yl_vz
      REAL(KIND=8) zl_d, zl_vx, zl_vy, zl_vz
      REAL(KIND=8) xr_d, xr_vx, xr_vy, xr_vz
      REAL(KIND=8) yr_d, yr_vx, yr_vy, yr_vz
      REAL(KIND=8) zr_d, zr_vx, zr_vy, zr_vz
      REAL(KIND=8) d0, vx0, vy0, vz0, sigma2_comp, sigma2_sole, sigma2
      REAL(KIND=8) b_turb, theta, eps_star, phi_t, sigs, scrit
      REAL(KIND=8) t_to_gyr, unit_d, mdum, unit_l

      n_cell = larr(1)
      n_thread = larr(2)

      d_thres = darr(1)
      factG = darr(2)
      pi = darr(3)
      t_to_gyr = darr(4)
      unit_d = darr(5)
      unit_l = darr(6)

      b_turb    = 0.4
      eps_star  = 0.5
      theta     = 0.33
      phi_t     = 0.57

      IF (ALLOCATED(sfe)) DEALLOCATE(sfe)
      IF (ALLOCATED(mach2)) DEALLOCATE(mach2)
      IF (ALLOCATED(alpha)) DEALLOCATE(alpha)
      IF (ALLOCATED(t_ff)) DEALLOCATE(t_ff)
      IF (ALLOCATED(dum_sf)) DEALLOCATE(dum_sf)
      IF (ALLOCATED(sig_s)) DEALLOCATE(sig_s)
      IF (ALLOCATED(sig_c)) DEALLOCATE(sig_c)

      ALLOCATE(sfe(1:n_cell))
      ALLOCATE(mach2(1:n_cell))
      ALLOCATE(alpha(1:n_cell))
      ALLOCATE(t_ff(1:n_cell))
      ALLOCATE(dum_sf(1:n_cell))
      ALLOCATE(sig_s(1:n_cell))
      ALLOCATE(sig_c(1:n_cell))

      sfe = 0.; mach2 = 0.; alpha = 0.; t_ff = 0.; dum_sf = 0.; sig_s = 0.; sig_c = 0.
      CALL OMP_SET_NUM_THREADS(n_thread)

      !$OMP PARALLEL DO default(shared) &
      !$OMP & private(ind, sigma2_sole, sigma2_comp, sigma2) &
      !$OMP & private(xl_d, xl_vx, xl_vy, xl_vz) &
      !$OMP & private(xr_d, xr_vx, xr_vy, xr_vz) &
      !$OMP & private(yl_d, yl_vx, yl_vy, yl_vz) &
      !$OMP & private(yr_d, yr_vx, yr_vy, yr_vz) &
      !$OMP & private(zl_d, zl_vx, zl_vy, zl_vz) &
      !$OMP & private(zr_d, zr_vx, zr_vy, zr_vz) &
      !$OMP & private(d0, vx0, vy0, vz0, ul, ur, sigs, scrit)
      DO i=1, n_cell
        IF(den(i) .LT. d_thres) CYCLE

        !! FIND NEIGHBORING CELLS
        ind(1) = get_sfe_findneighbor(x, y, z, x(i)-dx(i), y(i), z(i), n_cell)
        ind(2) = get_sfe_findneighbor(x, y, z, x(i)+dx(i), y(i), z(i), n_cell)
        ind(3) = get_sfe_findneighbor(x, y, z, x(i), y(i)-dx(i), z(i), n_cell)
        ind(4) = get_sfe_findneighbor(x, y, z, x(i), y(i)+dx(i), z(i), n_cell)
        ind(5) = get_sfe_findneighbor(x, y, z, x(i), y(i), z(i)-dx(i), n_cell)
        ind(6) = get_sfe_findneighbor(x, y, z, x(i), y(i), z(i)+dx(i), n_cell)

        !PRINT *, ind

        !! Compute sigma2
        sigma2_sole = 0.
        sigma2_comp = 0.

        xl_d = den(ind(1)) ; xl_vx = vx(ind(1)) ; xl_vy = vy(ind(1)) ; xl_vz = vz(ind(1))
        xr_d = den(ind(2)) ; xr_vx = vx(ind(2)) ; xr_vy = vy(ind(2)) ; xr_vz = vz(ind(2))
        yl_d = den(ind(3)) ; yl_vx = vx(ind(3)) ; yl_vy = vy(ind(3)) ; yl_vz = vz(ind(3))
        yr_d = den(ind(4)) ; yr_vx = vx(ind(4)) ; yr_vy = vy(ind(4)) ; yr_vz = vz(ind(4))
        zl_d = den(ind(5)) ; zl_vx = vx(ind(5)) ; zl_vy = vy(ind(5)) ; zl_vz = vz(ind(5))
        zr_d = den(ind(6)) ; zr_vx = vx(ind(6)) ; zr_vy = vy(ind(6)) ; zr_vz = vz(ind(6))

        !! Vel to Momentum?
        !xl_vx = xl_vx*xl_d; xl_vy = xl_vy*xl_d; xl_vz = xl_vz*xl_d
        !xr_vx = xr_vx*xr_d; xr_vy = xr_vy*xr_d; xr_vz = xr_vz*xr_d

        !yl_vx = yl_vx*yl_d; yl_vy = yl_vy*yl_d; yl_vz = yl_vz*yl_d
        !yr_vx = yr_vx*yr_d; yr_vy = yr_vy*yr_d; yr_vz = yr_vz*yr_d

        !zl_vx = zl_vx*zl_d; zl_vy = zl_vy*zl_d; zl_vz = zl_vz*zl_d
        !zr_vx = zr_vx*zr_d; zr_vy = zr_vy*zr_d; zr_vz = zr_vz*zr_d

        !d0 = den(i); vx0 = vx(i)*d0; vy0 = vy(i)*d0; vz0 = vz(i)*d0
        d0 = den(i); vx0 = vx(i); vy0 = vy(i); vz0 = vz(i)

        !! Divergence Term
        ul = (xl_d * xl_vx + d0 * vx0) / (xl_d + d0)
        ur = (xr_d * xr_vx + d0 * vx0) / (xr_d + d0)
        sigma2_comp = sigma2_comp + (ul-ur)**2

        ul = (yl_d * yl_vy + d0 * vy0) / (yl_d + d0)
        ur = (yr_d * yr_vy + d0 * vy0) / (yr_d + d0)
        sigma2_comp = sigma2_comp + (ul-ur)**2

        ul = (zl_d * zl_vz + d0 * vz0) / (zl_d + d0)
        ur = (zr_d * zr_vz + d0 * vz0) / (zr_d + d0)
        sigma2_comp = sigma2_comp + (ul-ur)**2

        !! Curl Term
        ul = (zr_d * zr_vy + d0 * vy0) / (zr_d + d0)
        ur = (zl_d * zl_vy + d0 * vy0) / (zl_d + d0)
        sigma2_sole = sigma2_sole + (ur-ul)**2

        ul = (zr_d * zr_vx + d0 * vx0) / (zr_d + d0)
        ur = (zl_d * zl_vx + d0 * vx0) / (zl_d + d0)
        sigma2_sole = sigma2_sole + (ur-ul)**2

        ul = (yr_d * yr_vx + d0 * vx0) / (yr_d + d0)
        ur = (yl_d * yl_vx + d0 * vx0) / (yl_d + d0)
        sigma2_sole = sigma2_sole + (ur-ul)**2

        ul = (yr_d * yr_vz + d0 * vz0) / (yr_d + d0)
        ur = (yl_d * yl_vz + d0 * vz0) / (yl_d + d0)
        sigma2_sole = sigma2_sole + (ur-ul)**2

        ul = (xr_d * xr_vz + d0 * vz0) / (xr_d + d0)
        ur = (xl_d * xl_vz + d0 * vz0) / (xl_d + d0)
        sigma2_sole = sigma2_sole + (ur-ul)**2

        ul = (xr_d * xr_vy + d0 * vy0) / (xr_d + d0)
        ur = (xl_d * xl_vy + d0 * vy0) / (xl_d + d0)
        sigma2_sole = sigma2_sole + (ur-ul)**2

        sigma2 = sigma2_comp + sigma2_sole

        sig_s(i)  = sigma2_sole
        sig_c(i)  = sigma2_comp
        !! Compute
        mach2(i) = sigma2 / cs2(i)

        alpha(i) = (5.0 * (sigma2 + cs2(i))) / (pi * factG * den(i) * dx(i)**2)
        !alpha(i) = (5.0 * (sigma2)) / (pi * factG * den(i) * dx(i)**2)

        sigs    = LOG(1.0 + (b_turb**2) * mach2(i))

        scrit   = LOG(0.067 / (theta**2) * alpha(i) * Mach2(i))

        sfe(i)  = eps_star / 2.0 * phi_t * EXP(3.0 / 8.0 * sigs) * &
                (2.0 - get_sfe_erf( (sigs - scrit) / SQRT(2.0*sigs) ))

        t_ff(i) = SQRT(pi * 3d0 / (32.0 * factG * den(i))) * t_to_gyr
        
        dum_sf(i) = sfe(i) / t_ff(i) * den(i) ! sim den / Gyr
        dum_sf(i) = dum_sf(i) * unit_d ! g/cc / Gyr
        dum_sf(i) = dum_sf(i) * (dx(i)*unit_l)**3 ! g / Gyr
        dum_sf(i) = dum_sf(i) / 1.98892e33 ! Msun / Gyr

        !PRINT *, sigma2, cs2(i), alpha(i), sigs, scrit, sfe(i)
        !PRINT *, eps_star / 2.0
        !PRINT *, phi_t * EXP(3./8. * sigs)
        !PRINT *, ((2.0 - get_sfe_erf( (sigs - scrit) / SQRT(2.0*sigs))))
        !PRINT *, (sigs - scrit) / SQRT(2.0*sigs)
        !STOP
      ENDDO
      !$OMP END PARALLEL DO


      END SUBROUTINE

      FUNCTION get_sfe_findneighbor(x, y, z, x0, y0, z0, n_cell) RESULT(ind0)
      IMPLICIT NONE

      REAL(KIND=8) x(n_cell), y(n_cell), z(n_cell)
      REAL(KIND=8) x0, y0, z0, mind, d3d
      INTEGER(KIND=4) ind0, i, n_cell

      mind = 1d32
      DO i=1, n_cell
        d3d = (x(i) - x0)**2 + (y(i) - y0)**2 + (z(i) - z0)**2
        IF(d3d .LE. mind) THEN
          mind = d3d
          ind0 = i
        ENDIF
      ENDDO

      RETURN
      END FUNCTION

      FUNCTION get_sfe_erf(x) RESULT(y)
              IMPLICIT NONE

              REAL(KIND=8) x, y
              REAL(KIND=8) pv, ph, p0, p1, p2, p3, p4, p5, p6, p7
              REAL(KIND=8) q0, q1, q2, q3, q4, q5, q6, q7

              pv = 1.26974899965115684d+01 ; ph= 6.10399733098688199d+00
              p0 = 2.96316885199227378d-01 ; p1= 1.81581125134637070d-01
              p2 = 6.81866451424939493d-02 ; p3= 1.56907543161966709d-02
              p4 = 2.21290116681517573d-03 ; p5= 1.91395813098742864d-04
              p6 = 9.71013284010551623d-06 ; p7= 1.66642447174307753d-07
              q0 = 6.12158644495538758d-02 ; q1= 5.50942780056002085d-01
              q2 = 1.53039662058770397d+00 ; q3= 2.99957952311300634d+00
              q4 = 4.95867777128246701d+00 ; q5= 7.41471251099335407d+00
              q6 = 1.04765104356545238d+01 ; q7= 1.48455557345597957d+01

              y       = x*x
              y       = EXP(-y) * x * (p7/(y+q7)+p6/(y+q6) + p5/(y+q5)+p4/(y+q4)+p3/(y+q3) &
              + p2/(y+q2)+p1/(y+q1)+p0/(y+q0))
              IF(x .LT. ph) y = y + 2d0/(EXP(pv*x)+1.0)


              RETURN
      END FUNCTION
      SUBROUTINE js_getsfe_free

      IF (ALLOCATED(sfe)) DEALLOCATE(sfe)
      IF (ALLOCATED(mach2)) DEALLOCATE(mach2)
      IF (ALLOCATED(alpha)) DEALLOCATE(alpha)
      IF (ALLOCATED(t_ff)) DEALLOCATE(t_ff)
      IF (ALLOCATED(dum_sf)) DEALLOCATE(dum_sf)
      IF (ALLOCATED(sig_s)) DEALLOCATE(sig_s)
      IF (ALLOCATED(sig_c)) DEALLOCATE(sig_c)

      END SUBROUTINE
END MODULE
