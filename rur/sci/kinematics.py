from rur.uri import *
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator, NearestNDInterpolator
from scipy.fft import rfft
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import gc
from multiprocessing import Process, Queue, Pool
from rur import uhmi
from numpy.lib.recfunctions import merge_arrays, append_fields

# This module contains useful functions related to galaxy kinematics.

def sig_rad(part: RamsesSnapshot.Particle, gal):
    # measure radial velocity dispersion
    vcen = get_vector(gal, 'v')
    rcen = get_vector(gal)

    vrel = part['vel', 'km/s'] - vcen
    rrel = part['pos', 'km/s'] - rcen
    vrad = np.sum(vrel * rrel, axis=-1) / utool.rss(rrel)
    return np.std(vrad)

def measure_amon(part: RamsesSnapshot.Particle, gal):
    vcen = get_vector(gal, 'v')
    rcen = get_vector(gal)

    vrel = part['vel', 'km/s'] - vcen
    rrel = (part['pos'] - rcen) / part.snap.unit['kpc']

    return np.cross(rrel, vrel) * utool.expand_shape(part['m', 'Msol'], [0], 2)

def align_axis(part: RamsesSnapshot.Particle, gal: np.recarray, center_vel=False):
    coo = get_vector(part)
    vel = get_vector(part, prefix='v')
    j = get_vector(gal, prefix='L')
    coo_gal =  get_vector(gal)
    vel_gal =  get_vector(gal, prefix='v')
    coo = utool.rotate_vector(coo - coo_gal, j)
    if(center_vel):
        vel = vel - vel_gal * part.snap.unit['km/s']
    vel = utool.rotate_vector(vel, j)

    table = utool.set_vector(part.table, coo + coo_gal, copy=True)
    utool.set_vector(table, vel, prefix='v', copy=False)
    part = RamsesSnapshot.Particle(table, part.snap)
    return part

def align_axis_cell(cell: RamsesSnapshot.Cell, gal: np.recarray, center_vel=False):
    # Experimental
    coo = get_vector(cell)
    vel = get_vector(cell, prefix='v')
    j = get_vector(gal, prefix='L')
    coo_gal =  get_vector(gal)
    vel_gal =  get_vector(gal, prefix='v')
    coo = utool.rotate_vector(coo - coo_gal, j)
    if(center_vel):
        vel = vel - vel_gal * cell.snap.unit['km/s']
    vel = utool.rotate_vector(vel, j)

    table = utool.set_vector(cell.table, coo + coo_gal, copy=True)
    utool.set_vector(table, vel, prefix='v', copy=False)
    cell = RamsesSnapshot.Cell(table, cell.snap)
    return cell

def vel_spherical(part: RamsesSnapshot.Particle, gal, pole):
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