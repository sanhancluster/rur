from rur.uri import *
import pkg_resources
from scipy.interpolate import LinearNDInterpolator

# This module contains useful functions related to galaxy analysis.

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

def circularity(part, radius_kpc=None):
    G = 6.674E-8
    x0 = np.median(part['pos', 'cm'], axis=0)
    v0 = np.average(part['vel', 'cm/s'], axis=0, weights=part['m'])

    xr = part['pos', 'cm'] - x0
    vr = part['vel', 'cm/s'] - v0
    m = part['m', 'g']

    dists = rss(xr)
    key = np.argsort(dists)

    dists = dists[key]
    xr = xr[key]
    vr = vr[key]
    m = m[key]

    if(radius_kpc is not None):
        print(dists / kpc)
        mask = dists < radius_kpc*kpc
        dists = dists[mask]
        xr = xr[mask]
        vr = vr[mask]
        m = m[mask]

    j = np.cross(xr, vr)
    jtot = np.sum(j, axis=0)
    jax = jtot/rss(jtot)

    mcum = np.cumsum(m)
    mtot = np.sum(m)
    rmax = np.max(dists)

    drs = np.diff(dists)

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

def read_YEPS_table(alpha=1):
    yeps_dir = pkg_resources.resource_filename('rur', 'YEPS/')
    if(alpha == 0):
        path = join(yeps_dir, 'wHB_ABMAG.mag')
    elif(alpha == 1):
        path = join(yeps_dir, 'wHB_ABMAGa0.mag')
    else:
        path = join(yeps_dir, 'wHB_ABMAGa4.mag')
    names = [
        'age', 'FeH', 'J_Ux', 'J_B', 'J_Bx', 'J_V', 'J_R', 'J_I', 'J_J', 'J_H', 'J_K', 'J_L', 'J_M',
        'W_C', 'W_M', 'W_T1', 'W_T2',
        'SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z',
        'WF2_f336w', 'WF2_f547m', 'WF2_f555w', 'WF2_f606w', 'WF2_f814w', 'WF2f850lp',
        'ACS_f475w', 'ACS_f555w', 'ACS_f606w', 'ACS_f814w', 'ACSf850lp',
        'WF3_f336w', 'WF3_f465w', 'WF3_f547m', 'WF3_f555w', 'WF3_f606w', 'WF3_f814w', 'WF3f850lp',
    ]
    table = np.genfromtxt(path, skip_header=1, names=names)
    return table

def measure_magnitude(stars, filter_name, alpha=1, total=True):
    table = read_YEPS_table(alpha)
    ages = stars['age', 'Gyr']
    FeHs = stars['FeH']

    ages[ages<=1.] = 1.
    FeHs[FeHs<=-2.5] = -2.5
    FeHs[FeHs>=0.5] = 0.5

    ip = LinearNDInterpolator(np.stack([table['age'], table['FeH']], axis=-1), table[filter_name], fill_value=np.nan)
    mags = ip(ages, FeHs)
    mags = mags - 2.5*np.log10(stars['m', 'Msol']/1E7)
    if(total):
        mags = mags[~np.isnan(mags)]
        return -2.5*np.log10(np.sum(10**(0.4*-mags)))
    else:
        return mags