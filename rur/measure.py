from rur.uri import *

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
