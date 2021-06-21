from rur.uri import *
import gc
from multiprocessing import Process, Queue, Pool
from tqdm import tqdm
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

def measure_amon(part: uri.RamsesSnapshot.Particle, gal):
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

def measure_radius_2d(ts, iout_start, ages_target=np.arange(11.624, 0, -0.25), nbins=100, gal_minmass=1E8, mass_cut_refine=2.4E-11, sfr_measure_Myr=100., nangles=100., sb_lim=26.5):

    def measure_galaxy(gal, line):
        line = line.copy()

        # load star / dm partciels
        snap_now.set_box_halo(gal, radius=1., radius_name='r')
        snap_now.get_part(exact_box=False)
        dm = snap_now.part['dm']
        star = snap_now.part['star']
        smbh = snap_now.part['smbh']

        gal_star = star
        if(star.size == 0):
            return line

        # match galaxy member stars only
        gal_mask = part_pool[np.abs(star['id'])] == idxs[halomaker['id'] == gal['hmid']][0]
        if(np.sum(gal_mask)==0):
            print('no galaxy star detected.', gal['hmid'], gal['timestep'])
            return line
        gal_star = star[gal_mask]

        # measure magnitude
        mags = phot.measure_magnitude(gal_star, filter_name='SDSS_r', total=False)
        lums = 10**(-mags/2.5)

        rholm_arr = []
        coo = gal_star['pos']-uri.get_vector(gal)
        for proj in [[0, 1], [0, 2], [1, 2]]:
            dists = utool.rss(coo[:, proj])
            dist_bins = np.linspace(0, gal['r'], nbins)
            sbarr = []
            for rmin, rmax in zip(dist_bins[:-1], dist_bins[1:]):
                mask = (rmin < dists) & (dists < rmax)
                if(np.sum(mask) == 0):
                    continue
                lums_bin = lums[mask]
                abslum = np.sum(lums_bin)
                absmag = -2.5*np.log10(abslum)
                area_pc2 = np.pi*(rmax**2-rmin**2)/snap_now.unit['pc']**2
                rbin = np.average(dists[mask], weights=lums_bin)
                sb = surface_brightness(absmag, area_pc2)
                sbarr.append([rbin, sb])
            sbarr = np.array(sbarr)

            mask = sbarr[:, 1] > sb_lim
            if(np.all(mask)): # too faint
                rholm = sbarr[0, 0]
            elif(np.sum(mask) == 0): # too bright
                rholm = sbarr[-1, 0]
            else:
                rank = np.arange(mask.size)[mask][0]
                rholm = np.interp(sb_lim, sbarr[rank-1:rank+1, 1], sbarr[rank-1:rank+1, 0])

            mask = dists < rholm
            lums_mask = lums[mask]
            dists_mask = dists[mask]

            r50 = utool.weighted_quantile(dists_mask, 0.5, sample_weight=lums_mask)
            r90 = utool.weighted_quantile(dists_mask, 0.9, sample_weight=lums_mask)

            rholm_arr.append([rholm, r90, r50])
        rholm_arr = np.array(rholm_arr)

        # set galaxy radius as median value of multiple projections
        rholm = np.mean(rholm_arr[:, 0])
        r90 = np.mean(rholm_arr[:, 1])
        r50 = np.mean(rholm_arr[:, 2])

        star_rholm = uri.cut_halo(gal_star, gal, rholm, use_halo_radius=False)
        mags_rholm = phot.measure_magnitude(star_rholm, filter_name='SDSS_r', total=False)
        lums_rholm = 10**(-2.5/mags_rholm)
        dists = utool.get_distance(gal, star_rholm)

        sfr = np.sum(star_rholm[star_rholm['age', 'Myr']<sfr_measure_Myr]['m', 'Msol']) / (sfr_measure_Myr*1E6)

        age = np.average(star_rholm['age', 'Gyr'], weights=star_rholm['m'])
        tform = np.median(snap_now.age-star_rholm['age', 'Gyr'])
        metal = np.average(star_rholm['metal'], weights=star_rholm['m'])

        dm_rholm = uri.cut_halo(dm, gal, rholm, use_halo_radius=False)
        contam = np.sum(dm_rholm[dm_rholm['m']>mass_cut_refine]['m'])/np.sum(dm_rholm['m'])

        bh_r90 = uri.cut_halo(smbh, gal, r90, use_halo_radius=False)

        line['sfr'] = sfr
        line['r90_2d'] = r90
        line['r50_2d'] = r50

        r90 = utool.weighted_quantile(dists, 0.9, sample_weight=star_rholm['m'])
        r50 = utool.weighted_quantile(dists, 0.5, sample_weight=star_rholm['m'])
        line['r90'] = r90
        line['r50'] = r50

        line['rholm'] = rholm

        line['nparts'] = star_rholm.size
        line['contam'] = contam
        line['age'] = age
        line['tform'] = tform
        line['metal'] = metal

        # Measure BHs
        if(bh_r90.size>0):
            bh_max = bh_r90[np.argmax(bh_r90['m'])]
            mbh = bh_max['m', 'Msol']
            bh_offset = utool.get_distance(gal, bh_max)

            line['mbh'] = mbh
            line['bh_offset'] = bh_offset

        keys = np.argsort(dists)
        star_rholm = star_rholm[keys]
        lums_rholm = lums_rholm[keys]
        dists = dists[keys]
        bins = np.concatenate([dists[::dists.size//nbins+1], [dists[-1]]])
        dist_bin_idx = np.searchsorted(dists, bins)
        amon_tot = np.sum(measure_amon(star_rholm, gal), axis=0)

        profile = line['profile']
        for rmin, rmax, ibin in zip(dist_bin_idx[:-1], dist_bin_idx[1:], np.arange(nbins)):
            slice_star = star_rholm[rmin:rmax]
            slice_lums = lums_rholm[rmin:rmax]
            if(slice_star.size==0):
                continue
            profile['m'][ibin] = np.sum(slice_star['m', 'Msol'])
            profile['sig'][ibin] = utool.rss(weighted_std(slice_star['vel', 'km/s'], axis=0, weights=slice_lums))/np.sqrt(3.)
            profile['rbin'][ibin] = bins[ibin+1]

            coo_sph = coo_spherical(slice_star, gal, amon_tot)

            profile['sig_rad'][ibin] = weighted_std(coo_sph[:, 0], weights=slice_lums)
            profile['sig_phi'][ibin] = weighted_std(coo_sph[:, 1], weights=slice_lums)
            profile['sig_theta'][ibin] = weighted_std(coo_sph[:, 2], weights=slice_lums)

            profile['v_rot'][ibin] = np.average(coo_sph[:, 1], weights=slice_lums)

            profile['age'][ibin] = np.average(slice_star['age', 'Gyr'], weights=slice_star['m'])
            profile['tform'][ibin] = np.median(snap_now.age-slice_star['age', 'Gyr'])
            profile['metal'][ibin] = np.average(slice_star['metal'], weights=slice_star['m'])

            amon = measure_amon(slice_star, gal)
            profile['Lx'][ibin] = np.sum(amon[:, 0])
            profile['Ly'][ibin] = np.sum(amon[:, 1])
            profile['Lz'][ibin] = np.sum(amon[:, 2])
            profile['lum'][ibin] = np.sum(slice_lums)
        line['profile'] = profile
        return line

    def measure_galaxies(gal_slice, line_slice, q):
        for gal, line, idx in zip(gal_slice, line_slice, np.arange(line_slice.size)):
            line_slice[idx] = measure_galaxy(gal, line)
        q.put(line_slice)

    if(isinstance(ts, uri.RamsesSnapshot)):
        ts = uri.TimeSeries(ts)

    # set snapshot
    snap = ts[iout_start]

    # get age-iout matching array
    ages = []
    for iout in np.arange(1, iout_start+1):
        try:
            ages.append([iout, ts[iout].age])
        except FileNotFoundError:
            continue
    ages = np.array(ages)

    # find timesteps that matches with ages_target
    iouts_target = []
    for age in ages_target:
        iouts_target.append(ages[np.argmin(np.abs(ages[:, 1]-age)), 0])
    iouts_target = np.array(iouts_target)

    # load ptree file
    ptree = uhmi.PhantomTree.load(snap, ptree_file='ptree_stable.pkl')

    # sort aexp and iouts
    iouts = np.sort(np.unique(ptree['timestep']))

    # load particle id list as sample
    part_ids = uhmi.HaloMaker.load(snap, galaxy=True, load_parts=True, path_in_repo='galaxy_local')[1]

    # set up pool (hid array) for member filtering
    max_part_size = int(np.max(part_ids) * 1.2)
    part_pool = np.full(max_part_size, -1, dtype='i4')

    # set dtypes and names
    profile_names = ['rbin', 'm', 'sig', 'sig_rad', 'sig_phi', 'sig_theta', 'age', 'tform', 'metal', 'Lx', 'Ly', 'Lz', 'lum', 'v_rot']
    extras = ['sfr', 'rholm', 'r90', 'r90_2d', 'r50', 'r50_2d', 'contam', 'age', 'tform', 'metal', 'mbh', 'bh_offset']

    profile_dtype = {
        'names': profile_names,
        'formats': ['f8'] * len(profile_names),
    }
    dtype = {
        'names': ['profile'] + extras,
        'formats': [(profile_dtype, nbins)] + ['f8'] * len(extras)
    }
    dtype = np.dtype(dtype)

    # leave target timestpes only
    ptree_target = ptree[np.isin(ptree['timestep'], iouts_target)]
    ptree_target = ptree_target[ptree_target['m']>gal_minmass]
    output_table = np.zeros(ptree_target.size, dtype=dtype)

    output_table = merge_arrays([ptree_target, output_table], usemask=False, flatten=True)
    output_table = np.sort(output_table, order='id')

    # main loop over timesteps
    for iout in iouts_target:
        uri.verbose=1

        # mask and sort snapshot gals
        mask = ptree_target['timestep'] == iout
        gals = ptree_target[mask]
        print('iout = %d, ngals = %d' % (iout, gals.size))
        if(gals.size == 0):
            continue
        gals.sort(order='hmid')

        # load halomaker data (particle id list)
        snap_now = ts[iout]
        halomaker, part_ids = uhmi.HaloMaker.load(snap_now, galaxy=True, load_parts=True, path_in_repo='galaxy_local')

        idxs = np.arange(halomaker.size)
        halomaker_idx = np.repeat(idxs, halomaker['nparts'])

        # pre-load domains from halo position
        cpulist = snap_now.get_halos_cpulist(gals, radius=1.05, radius_name='r', n_divide=5)
        snap_now.get_part(cpulist=cpulist)

        # initialize pool
        part_pool[:] = -1
        part_pool[part_ids] = halomaker_idx
        uri.timer.verbose = 0

        # loop over galaxies
        # Multiprocessing implemented
        nproc = 36 # Max. number of jobs
        nchunk = 1 # number of chunk size for jobs to be divided

        njobs = int(np.ceil(gals.size/nchunk))
        iterator = tqdm(np.arange(njobs), ncols=100)

        jobs = []
        q = Queue()
        for i in iterator:
            while True:
                for idx in np.arange(len(jobs))[::-1]:
                    if (not jobs[idx].is_alive()):
                        jobs.pop(idx)
                if (len(jobs) >= nproc):
                    sleep(0.5)
                else:
                    break

            st, ed = i * nchunk, np.minimum((i+1)*nchunk, gals.size)
            gal_slice = gals[st:ed]
            galidxs = np.searchsorted(output_table['id'], gal_slice['id'])
            line_slice = output_table[galidxs]

            p = Process(target=measure_galaxies, args=(gal_slice, line_slice, q))
            jobs.append(p)
            p.start()
            while not q.empty():
                line_slice = q.get()
                galidxs = np.searchsorted(output_table['id'], line_slice['id'])
                output_table[galidxs] = line_slice
        iterator.close()

        ok = False
        while not ok:
            ok = True
            for idx in np.arange(len(jobs)):
                if (jobs[idx].is_alive()):
                    ok = False
                if (not q.empty()):
                    line_slice = q.get()
                    galidxs = np.searchsorted(output_table['id'], line_slice['id'])
                    output_table[galidxs] = line_slice
                else:
                    sleep(0.5)


        uri.timer.verbose = 1
        snap_now.clear()
        gc.collect()
    output_table = output_table[output_table['m']>gal_minmass]

    return output_table

def surface_brightness(absmag, area_pc2):
    return absmag + 2.5*np.log10(area_pc2) - 5 + 5 * np.log10(3600*180/np.pi)
