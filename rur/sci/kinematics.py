from rur.uri import *
from rur.sci import geometry as geo
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator, NearestNDInterpolator
from scipy.fft import rfft
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from rur.vr.fortran.js_getpt_ft import js_getpt
from rur.vr.fortran.js_getsfe_ft import js_getsfe
import rur.vrload as vl

# This module contains useful functions related to galaxy kinematics.

def sig_rad(part: Particle, gal):
    # measure radial velocity dispersion
    vcen = get_vector(gal, 'v')
    rcen = get_vector(gal)

    vrel = part['vel', 'km/s'] - vcen
    rrel = part['pos', 'km/s'] - rcen
    vrad = np.sum(vrel * rrel, axis=-1) / utool.rss(rrel)
    return np.std(vrad)

def measure_amon(part: Particle, gal):
    vcen = get_vector(gal, 'v')
    rcen = get_vector(gal)

    vrel = part['vel', 'km/s'] - vcen
    rrel = (part['pos'] - rcen) / part.snap.unit['kpc']

    return np.cross(rrel, vrel) * part['m', 'Msol'][..., np.newaxis]

def align_axis(part: Particle, gal: np.recarray, center_vel=False, prefix=None):
    coo = get_vector(part)
    vel = get_vector(part, prefix='v')

    if prefix is None:
        prefix_candidates = ['L', 'j']
        # if prefix is not defined, search over candidates in dtype.names
        for pf in prefix_candidates:
            if np.all(np.isin([pf+'x', pf+'y', pf+'z'], gal.dtype.names)):
                prefix = pf
                break
        if prefix is None:
            raise ValueError("axis not available for the target array.")
    j = get_vector(gal, prefix=prefix)
    coo_gal =  get_vector(gal)
    vel_gal =  get_vector(gal, prefix='v')
    coo = geo.align_to_vector(coo - coo_gal, j)
    if(center_vel):
        vel = vel - vel_gal * part.snap.unit['km/s']
    vel = geo.align_to_vector(vel, j)

    table = utool.set_vector(part.table, coo + coo_gal, copy=True)
    utool.set_vector(table, vel, prefix='v', copy=False)
    part = Particle(table, part.snap)
    return part

def align_axis_cell(cell: Cell, gal: np.recarray, center_vel=False, prefix=None):
    # Experimental
    coo = get_vector(cell)
    vel = get_vector(cell, prefix='v')

    if prefix is None:
        prefix_candidates = ['L', 'j']
        # if prefix is not defined, search over candidates in dtype.names
        for pf in prefix_candidates:
            if np.all(np.isin([pf+'x', pf+'y', pf+'z'], gal.dtype.names)):
                prefix = pf
                break
        if prefix is None:
            raise ValueError("axis not available for the target array.")
    j = get_vector(gal, prefix=prefix)
    coo_gal =  get_vector(gal)
    vel_gal =  get_vector(gal, prefix='v')
    coo = geo.align_to_vector(coo - coo_gal, j)
    if(center_vel):
        vel = vel - vel_gal * cell.snap.unit['km/s']
    vel = geo.align_to_vector(vel, j)

    table = utool.set_vector(cell.table, coo + coo_gal, copy=True)
    utool.set_vector(table, vel, prefix='v', copy=False)
    cell = Cell(table, cell.snap)
    return cell

def vel_spherical(part: Particle, gal, pole):
    # measure radial velocity dispersion
    vcen = get_vector(gal, 'v')
    rcen = get_vector(gal)

    vrel = part['vel', 'km/s'] - vcen
    rrel = (part['pos'] - rcen) / part.snap.unit['kpc']
    vabs = utool.rss(vrel)

    vrad = np.sum(vrel * rrel, axis=-1) / utool.rss(rrel)
    urot = np.cross(rrel, pole) / utool.expand_shape(utool.rss(pole) * utool.rss(rrel), [0], 2)

    vphi = np.sum(vrel * urot, axis=-1)
    vtheta = np.sqrt(vabs**2 - vphi**2 - vrad**2)

    return np.stack([vrad, vphi, vtheta], axis=-1)

#def circularity(part, radius_kpc=None):
#    G = 6.674E-8
#    x0 = np.median(part['pos', 'cm'], axis=0)
#    v0 = np.average(part['vel', 'cm/s'], axis=0, weights=part['m'])
#
#    xr = part['pos', 'cm'] - x0
#    vr = part['vel', 'cm/s'] - v0
#    m = part['m', 'g']
#
#    dists = rss(xr)
#    key = np.argsort(dists)
#
#    dists = dists[key]
#    xr = xr[key]
#    vr = vr[key]
#    m = m[key]
#
#    if(radius_kpc is not None):
#        print(dists / kpc)
#        mask = dists < radius_kpc*kpc
#        dists = dists[mask]
#        xr = xr[mask]
#        vr = vr[mask]
#        m = m[mask]
#
#    j = np.cross(xr, vr)
#    jtot = np.sum(j, axis=0)
#    jax = jtot/rss(jtot)
#
#    mcum = np.cumsum(m)
#    mtot = np.sum(m)
#    rmax = np.max(dists)
#
#    drs = np.diff(dists)
#
#    ebin = G * mcum / dists**2 * drs
#    ebin_cum = G*mtot/rmax + np.cumsum(ebin[::-1])[::-1]
#
#    etot = np.sqrt(G*mcum/dists) + 0.5*rss(vr)**2
#    ebin = np.sqrt(G*mcum/dists)
#    ecir = G * M /
#
#    idxs = np.searchsorted(ecir, etot)
#    jcire = jcir[idxs-1]
#
#    rot = np.cross(jax, xr)
#    rot = rot/np.array([rss(rot)]).T
#    vrot = np.sum(rot * vr, axis=-1)
#    rrot = np.sqrt(dists**2 - np.sum(rot * xr, axis=-1)**2)
#    jrot = vrot * dists

#    return jrot/jcire

def surface_brightness(absmag, area_pc2):
    return absmag + 2.5*np.log10(area_pc2) - 5 + 5 * np.log10(3600*180/np.pi)

def rotate2d(p, angle=0, origin=(0, 0)):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

class kinemetry:
    # Kinemetry algorithm based on Krajnovic+ 2006
    @staticmethod
    def ellipse_sample(a, q, pa, n_sample=100, return_psi=False, draw=False, color='white'):
        # return given number of sampling points from ellipse.
        psi = np.linspace(0, 2*np.pi, n_sample + 1)[:-1]
        x, y = a*np.cos(psi), a*q*np.sin(psi)
        r, theta = np.sqrt(x**2+y**2), np.arctan2(y, x)
        theta %= 2*np.pi
        x, y = r*np.cos(theta + pa), r*np.sin(theta + pa)
        coo = np.stack([x, y], axis=-1)

        if(draw):
            plt.plot(list(coo[:, 0])+[coo[0, 0]], list(coo[:, 1])+[coo[0, 1]], color=color, linewidth=1)
            plt.scatter(coo[0, 0], coo[0, 1], color=color, s=5, zorder=100)

        if(return_psi):
            return coo, psi
        else:
            return coo

    @staticmethod
    def get_chisq(pars, a, lint, n_sample=100, n_subsample=5, a_width=0, moment='odd'):
        q, pa = tuple(pars)
        if(q < 0 or q > 1):
            return np.inf
        varr = []
        if(a_width > 0):
            for a_local in np.linspace(a-a_width, a+a_width, n_subsample):
                coo = kinemetry.ellipse_sample(a_local, q, pa, n_sample)
                v = lint(coo)
                varr.append(v)
            v = np.average(varr, axis=0)
        else:
            coo = kinemetry.ellipse_sample(a, q, pa, n_sample)
            v = lint(coo)
        ff = rfft(v)
        if(moment == 'odd'):
            return (ff[1].imag**2+np.sum(np.abs(ff[2:4])**2)) / ff[1].real**2
        elif(moment == 'even'):
            return np.var(v)

    @staticmethod
    def fit_ellipse(a, lint, points, weights, wlint=None, n_sample=100, init_iter=4, return_diff=True, moment='odd'):
        mask = utool.rss(points) < a*2
        if(np.sum(mask)>1):
            points = points[mask]
            weights = weights[mask]

            q_init = 1.
            pa_init = 0.
            if(moment == 'odd'):
                for _ in range(init_iter):

                    coo_init = kinemetry.ellipse_sample(a, q_init, pa_init, n_sample)
                    pa_offset = -np.angle(rfft(lint(coo_init))[1])
                    pa_init = pa_init + pa_offset

                    points_new = rotate2d(points, angle=-pa_init)
                    std = utool.weighted_std(points_new, weights=weights, axis=0)
                    q_init = np.divide(*std[::-1])
                    q_init = np.clip(q_init, 0., 1.)

            else:
                x2b = np.average(points**2, axis=0)
                xyb = np.average(np.product(points, axis=-1))
                aa = np.sqrt(np.sum(x2b)/2 + np.sqrt((np.diff(x2b)/2)**2 + xyb**2))
                b = np.sqrt(np.sum(x2b)/2 - np.sqrt((np.diff(x2b)/2)**2 + xyb**2))
                pa_init = np.arctan2(2*xyb, -np.diff(x2b))/2

                #aa, b, phi = phot.ellipse_fit(points, weights)
                #pa_init = phi
                q_init = b/aa
                pa_init %= np.pi
            coo_init = kinemetry.ellipse_sample(a, q_init, pa_init, n_sample)

            if(wlint is not None):
                width = np.average(wlint(coo_init))
            else:
                width = 0

            #pars_init = q_init, pa_init, 0
            #return pars_init, 0

            fit = minimize(kinemetry.get_chisq, (q_init, pa_init), method='Nelder-Mead', args=(a, lint, n_sample, 5, width, moment))
            pars = fit.x
            pars[1] %= (np.pi*2)
            chisq = kinemetry.get_chisq(pars, a, lint, n_sample, 5, width, moment)
        else:
            pars = np.nan, np.nan
            q_init = np.nan
            chisq = np.nan
        if(return_diff):
            diff = pars[0]/q_init
            return pars, diff, chisq
        else:
            return pars


    @staticmethod
    def do_kinemetry(a_arr, points, values, weights, widths=None, n_sample=100, interpolate='linear', moment='odd'):
        # a_arr: desired array of semi-major axis
        # points: coordinates of values
        # values: values to fit
        # weights: weights for values

        if(interpolate == 'linear'):
            interpolator = LinearNDInterpolator
        elif(interpolate == 'nearest'):
            interpolator = NearestNDInterpolator
        elif(interpolate == 'cubic'):
            interpolator = CloughTocher2DInterpolator
        else:
            raise ValueError("Unknown interpolator: ", interpolate)

        lint = interpolator(points, values)
        if(widths is not None):
            wlint = interpolator(points, widths)
        else:
            wlint = None

        pars_arr = np.zeros(len(a_arr), dtype=[('a', 'f8'), ('q', 'f8'), ('PA', 'f8'), ('q_diff', 'f8'), ('chisq', 'f8')])
        pars_arr['a'] = a_arr
        for line in pars_arr:
            pars, diff, chisq = kinemetry.fit_ellipse(line['a'], lint, points, weights, wlint, n_sample=n_sample, moment=moment)
            line['q'] = pars[0]
            line['PA'] = pars[1]
            line['q_diff'] = diff
            line['chisq'] = chisq
        return pars_arr

def f_getpot(pos, mm, num_thread=None, timereport=None, mesh_type=None, pole_type=None, splitval_type=None, splitdim_type=None, bsize=None):
    """ Potential calculation with KDTree

    This routine gives you the potential value (in (km/s)^2 unit) with efficient calculations.
    Kinetic energy can be simply derived by velocity^2 (velocity in km/s)
    If you include cell data, the internal energy can be obtained by
        temperature / density / (5.d/3. - 1.) * mH/kB * (scale_l / scale_t)^2. / mH * KB * 1e-10

    If you want to use both particles (DM and/or Star) and cell, just merge them to the same array

    Parameters
    ----------
    pos : numpy.array with shape (Npart, Ndim)
        position of particles and/or cell in kpc
        Npart is the number of particles (+cell)
        Ndim is the dimension (6 allowed but velocities are not used in the potential calculation)
    mm : numpy.array with shape (Npart)
        mass of particles and/or cell in Msun
    num_thread: integer
        # of threads to be used
        Values around 10 gives the best efficiency
    timereport: 'on' or 'off'
        Time log report
    mesh_type, pole_type, splitval_type, splitdim_type:
        Choices of how to treat grids, and KDTree
        Don't have to touch these. Just use the default settings
    bsize : integer
        The number of particles in the leaf node.
        Smaller values gives you more accurate estimation. But 1000 is enough

    Attributes
    ----------

    Examples
    ----------
    >>> from rur.sci.kinematics import sk
    >>> from numpy as np

    Allocate arrays (npart is the number of particles)
    >>> pos=np.zeros((npart,6))
    >>> mm=np.zeros(npart)

    Put the position and mass values in to the 'pos' and 'mm' array.
    In VR case,
    >>> import rur.vrload as vrload
    >>> vr       =vrload.vr_load('NH')
    >>> p=vrgal.f_rdptcl(1026, 100, horg='g')[0] # particle information of the galaxy with ID=100 @ snap=1026

    >>> pos[:,0] = p['xx']
    >>> pos[:,1] = p['yy']
    >>> pos[:,2] = p['zz']
    >>> mm       = p['mass']

    Potential
    >>> pot=sk.f_getpot(pos, mm)
    """
    ##----- Settings
    if(num_thread==None):       num_thread      = 10
    if(mesh_type==None):        mesh_type       = 'mesh'    # mesh or pm
    if(pole_type==None):        pole_type       = 'pole'    # pole or part
    if(timereport==None):       timereport      = 'on'      # on or off
    if(splitval_type==None):    splitval_type   = 'median'
    if(splitdim_type==None):    splitdim_type   = 'maxrange'
    if(bsize==None):
        bsize   = np.int32(1024)
        if(len(mm) < 100000): bsize = np.int32(100)
        if(len(mm) < 10000):  bsize = np.int32(10)

    ##-----
    pos = np.double(pos)        # should be in kpc
    mm  = np.double(mm)         # should be in Msun
    n_ptcl  = np.int32(len(mm))
    n_dim   = np.int32(len(pos[0,:]))

    if(mesh_type=='mesh'): p_type = np.int32(0)
    elif(mesh_type=='pm'): p_type = np.int32(1)
    else: p_type = np.int32(0)

    if(pole_type=='part'): e_type = np.int32(1)
    elif(pole_type=='pole'): e_type = np.int32(0)
    else: e_type = np.int32(0)

    if(timereport=='on'): tonoff = np.int32(1)
    else: tonoff = np.int32(0)

    if(splitval_type=='median'): v_type = np.int32(0)
    else: v_type = np.int32(0)       ## has not been implemented yet

    if(splitdim_type=='maxrange'): d_type = np.int32(0)
    else: d_type = np.int32(0)       ## has not been implemented yet


    Gconst  = np.double(6.67408e-11)    ## m^3 kg^-1 s^-2
    mtoKpc  = np.double(1) / np.double(3.086e19)
    kgtoMsun= np.double(1) / np.double(1.98892e30)
    Gconst  *= (mtoKpc / kgtoMsun * np.double(1e-6)) ## (km/s)^2 Kpc/Msun

    ##-----
    larr    = np.zeros(20, dtype=np.int32)
    darr    = np.zeros(20, dtype='<f8')

    larr[0] = n_ptcl
    larr[1] = n_dim
    larr[2] = num_thread
    larr[3] = p_type
    larr[4] = e_type
    larr[5] = tonoff

    larr[10]= d_type
    larr[11]= v_type
    larr[12]= bsize

    darr[0] = Gconst

    js_getpt.js_getpt_ft(larr, darr, pos, mm)
    pot = js_getpt.pot
    pot_out = np.array(pot)
    js_getpt.js_getpt_ft_free()
    return pot_out

def f_getsfe(snum, x, y, z, vx, vy, vz, dx, den, temp, num_thread=None, simtype=None):
    """ SFE Calculation by cell by cell

    This routine gives you the mass-weighted SFE-related values with given cell information
    NH SF prescription is default

    Parameters
    ----------
    snum : integer for snapshot number
    x, y, z, vx, vy, vz : numpy 1D array for position and velocity (in simulation unit)
    dx : numpy 1D array for cell size (dx) in simulation unit
    den : density of cell in simulation unit
    temp : P/rho of cell in simulation unit ( hydrovar[*,4] / hydrovar[*,0] )

    Attributes
    ----------

    Examples
    ----------
    >>> from rur.sci.kinematics import sk
    >>> from numpy as np

    """

    ##----- Settings
    if(num_thread==None): num_thread      = 10
    if(simtype==None): simtype = 'NH' # Not implemented yet for other SF recipe

    if(simtype=='NH'):
        dcrit = 10.
        vr = vl.vr_load('NH')
        vrf= vl.vr_getftns(vr)

        info= vrf.g_info(snum)

    x = np.double(x)
    y = np.double(y)
    z = np.double(z)
    vx = np.double(vx)
    vy = np.double(vy)
    vz = np.double(vz)
    dx = np.double(dx)
    den = np.double(den)
    temp= np.double(temp)
   
    mass = den * (dx**3)
    ##-----
    cs2 = (5./3. - 1.) * (1.3806200e-16) * (1e-10) / 1.6600000e-24 * temp * info['unit_T2'] / info['kms']**2

    ##-----
    larr    = np.zeros(20, dtype=np.int32)
    darr    = np.zeros(20, dtype='<f8')

    larr[0] = len(x)
    larr[1] = num_thread

    darr[0] = 10.0 / info['nH'] 
    darr[1] = 3.0/8.0 / np.pi * info['oM'] * info['aexp']
    darr[2] = np.pi
    darr[3] = info['unit_t'] / 86400. / 365. / 1e9
    darr[4] = info['unit_d']
    darr[5] = info['unit_l']

    js_getsfe.js_getsfe_ft(larr, darr, x, y, z, vx, vy, vz, dx, den, cs2)

    sfe = js_getsfe.sfe
    mach2 = js_getsfe.mach2
    alpha = js_getsfe.alpha
    t_ff = js_getsfe.t_ff
    dm_sf = js_getsfe.dum_sf
    sig_c = js_getsfe.sig_c
    sig_s = js_getsfe.sig_s
    #pot = js_getpt.pot

    ind = (sfe > 0.)
    sfe_avg = np.sum(sfe[ind] * mass[ind]) / np.sum(mass[ind])
    mach_avg = np.sum(mach2[ind]**0.5 * mass[ind]) / np.sum(mass[ind])
    alpha_avg = np.sum(alpha[ind] * mass[ind]) / np.sum(mass[ind])

    js_getsfe.js_getsfe_free()

    return sfe_avg, mach_avg, alpha_avg
    #return darr
