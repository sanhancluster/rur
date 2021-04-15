from rur.uri import *
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
    rrel = part['pos', 'km/s'] - rcen

    return np.cross(rrel, vrel) * part['m']

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

def measure_shell(snap, z_target=[0.25, 0.5, 1.0], nbins=20, gal_minmass=1E8, mass_cut_refine=2.4E-11, sfr_measure_Myr=100.):
    ptree = uhmi.PhantomTree.load(snap, ptree_file='ptree_stable.pkl')
    aexps = np.sort(np.unique(ptree['aexp']))
    iouts = np.sort(np.unique(ptree['timestep']))
    if(not aexps.size == iouts.size):
        raise ValueError('aexps and timesteps are not synchronized')
    aexp_target = 1./(np.array(z_target)+1)

    part_ids = uhmi.HaloMaker.load(snap, galaxy=True, load_parts=True)[1]
    max_part_size = int(np.max(part_ids) * 1.2)
    part_pool = np.full(max_part_size, -1, dtype='i4')

    profile_names = ['bin', 'm', 'sig', 'sig_rad', 'age', 'tform', 'metal', 'Lx', 'Ly', 'Lz']
    extras = ['sfr', 'r90', 'r50', 'contam', 'age', 'tform', 'metal', 'mbh', 'bh_offset']

    profile_dtype = {
        'names': profile_names,
        'formats': ['f8'] * len(profile_names),
    }
    dtype = {
        'names': ['profile'] + extras,
        'formats': [(profile_dtype, nbins)] + ['f8'] * len(extras)
    }
    dtype = np.dtype(dtype)

    iouts_target = []
    for aexp in aexp_target:
        idx = np.argmin(aexps - aexp)
        print(idx, np.min(aexps-aexp))
        iout = iouts[idx]
        iouts_target.append(iout)

    ptree_target = ptree[np.isin(ptree['timestep'], iouts_target)]
    output_table = np.zeros(ptree_target.size, dtype=dtype)

    output_table = merge_arrays([ptree_target, output_table], usemask=False, flatten=True)
    np.sort(output_table, order='id')
    print('nouts = ', len(iouts_target))
    print('aexps = ', aexp_target)

    for iout in iouts_target:
        mask = ptree_target['timestep'] == iout
        gals = ptree_target[mask]
        gals = gals[gals['m']>gal_minmass]
        print('iout = %d, ngals = %d' % (iout, gals.size))
        if(gals.size == 0):
            continue
        gals.sort(order='hmid')

        snap_now = snap.switch_iout(iout)
        halomaker, part_ids = uhmi.HaloMaker.load(snap_now, galaxy=True, load_parts=True)

        idxs = np.arange(halomaker.size)
        halomaker_idx = np.repeat(idxs, halomaker['nparts'])

        cpulist = snap_now.get_halos_cpulist(gals, radius=1.05, radius_name='r', n_divide=5)
        snap_now.get_part(cpulist=cpulist)

        part_pool[:] = -1
        part_pool[part_ids] = halomaker_idx

        for gal in gals:
            idx = np.searchsorted(output_table['id'], gal['id'])

            snap_now.set_box_halo(gal, radius=1., radius_name='r')
            snap_now.get_part(exact_box=False)
            dm = snap_now.part['dm']
            star = snap_now.part['star']
            smbh = snap_now.part['smbh']

            gal_mask = part_pool[np.abs(star['id'])] == idxs[halomaker['id'] == gal['hmid']]
            if(np.sum(gal_mask)==0):
                continue
            gal_star = star[gal_mask]
            dists = utool.get_distance(gal, gal_star)

            # first calculate r90 (may be updated if there's a subgalaxy
            r90 = utool.weighted_quantile(dists, 0.9, sample_weight=gal_star['m'])

            r50 = utool.weighted_quantile(dists, 0.5, sample_weight=gal_star['m'])
            sfr = np.sum(gal_star[gal_star['age', 'Myr']<sfr_measure_Myr]['m', 'Msol']) / (sfr_measure_Myr*1E6)

            age = np.average(gal_star['age', 'Gyr'], weights=gal_star['m'])
            tform = np.median(snap_now.age-gal_star['age', 'Gyr'])
            metal = np.average(gal_star['metal'], weights=gal_star['m'])

            dm_r90 = cut_halo(dm, gal, r90, use_halo_radius=False)
            contam = np.sum(dm_r90[dm_r90['m']>mass_cut_refine]['m'])/np.sum(dm_r90['m'])

            bh_r90 = cut_halo(smbh, gal, r90, use_halo_radius=False)

            output_table['sfr'][idx] = sfr
            output_table['r90'][idx] = r90
            output_table['r50'][idx] = r50

            output_table['nparts'][idx] = gal_star.size
            output_table['contam'][idx] = contam
            output_table['age'][idx] = age
            output_table['tform'][idx] = tform
            output_table['metal'][idx] = metal

            # Measure BHs
            if(bh_r90.size>0):
                bh_max = bh_r90[np.argmax(bh_r90['m'])]
                mbh = bh_max['m', 'Msol']
                bh_offset = utool.get_distance(gal, bh_max)

                output_table['mbh'][idx] = mbh
                output_table['bh_offset'][idx] = bh_offset

            keys = np.argsort(dists)
            gal_star = gal_star[keys]
            dists = dists[keys]
            bins = np.linspace(0, r90*1.5, nbins+1)
            dist_bin_idx = np.searchsorted(dists, bins)
            profile = output_table['profile'][idx]
            for rmin, rmax, ibin in zip(dist_bin_idx[:-1], dist_bin_idx[1:], np.arange(nbins)):
                slice_star = gal_star[np.searchsorted(dists, rmin):np.searchsorted(dists, rmax)]
                profile['m'][ibin] = np.sum(slice_star['m', 'Msol'])
                profile['sig'][ibin] = utool.rss(np.std(slice_star['vel', 'km/s'], axis=0))/np.sqrt(3.)
                profile['sig_rad'][ibin] = sig_rad(slice_star, gal)
                profile['age'][ibin] = np.average(slice_star['age', 'Gyr'], weights=slice_star['m'])
                profile['tform'][ibin] = np.median(snap.age-slice_star['age', 'Gyr'])
                profile['metal'][ibin] = np.average(slice_star['metal'], weights=slice_star['m'])
                amon = measure_amon(slice_star, gal)
                profile['Lx'][ibin] = np.sum(amon[:, 0])
                profile['Ly'][ibin] = np.sum(amon[:, 1])
                profile['Lz'][ibin] = np.sum(amon[:, 2])

    return output_table

