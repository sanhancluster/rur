from rur.uri import *
import pkg_resources
from scipy.interpolate import LinearNDInterpolator
from numpy.lib.recfunctions import append_fields
from numpy.lib.recfunctions import merge_arrays
from collections import defaultdict
from multiprocessing import Pool, shared_memory

# This module contains useful functions related to galaxy photometry.
cb07_table = None

def set_boundaries(data, range):
    data = np.clip(data, range[0], range[1])
    isnan = np.isnan(data)
    if(np.any(isnan)):
        data[isnan] = range[0]
    return data
    # data[(data<=range[0]) | np.isnan(data)] = range[0]
    # data[data>range[1]] = range[1]
    # return data

def read_YEPS_table(alpha=1):
    yeps_dir = pkg_resources.resource_filename('rur', 'YEPS/')
    if(alpha == 0):
        path = join(yeps_dir, 'wHB_ABMAG.dat')
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

def read_cb07_table():
    global cb07_table
    if(cb07_table is not None):
        return cb07_table

    tags = ['1ABmag', '1color']
    mtags = [22, 32, 42, 52, 62, 72, 82]
    zarr = [0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05, 0.1]
    cb07_dir = pkg_resources.resource_filename('rur', 'cb07/')

    tables = []

    for tag in tags:
        table = []
        for mtag, z in zip(mtags, zarr):
            path = join(cb07_dir, 'cb2007_lr_BaSeL_m%02d_chab_ssp.%s' % (mtag, tag))
            with open(path, mode='r') as f:
                lines = f.readlines()

            i = 0
            while lines[i+1][0] == '#':
                i += 1
            names = lines[i][1:].split()

            subtable = np.genfromtxt(path, names=names)
            subtable = append_fields(subtable, names='metal', data=np.full(subtable.size, z), usemask=False)
            table.append(subtable)
        table = np.concatenate(table)
        tables.append(table)

    names_all = []
    tables_out = []
    for table in tables:
        names = table.dtype.names
        names_out = []
        for name in names:
            if(name not in names_all):
                names_out.append(name)
        names_all += names_out
        tables_out.append(table[names_out])
    tables = merge_arrays(tables_out, flatten=True)
    cb07_table = tables

    return tables

#    names = {}
#    names['1ABmag'] = [
#        'log_age', 'Mbol', 'Umag', 'Bmag', 'Vmag', 'Kmag',
#        '14-V', 'CFHT_g', 'CFHT_r', 'CFHT_i', 'CFHT_y', 'CFHT_z', 'CFHT_Ks',
#        'GALEX_FUV', 'GALEX_NUV', 'GALEX_F(FUV)', 'GALEX_F(NUV)', 'GALEX_F(1500A)']
#
#    names['1color'] = [
#    ]
#
#    zarr = [0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05, 0.1]
#
#    table = []
#
#    for z, mtag in zip(zarr, mtags):
#        path = join(cb07_dir, 'cb2007_lr_BaSeL_m%02d_chab_ssp.1ABmag' % mtag)
#        subtable = np.genfromtxt(path, names=names)
#        subtable = append_fields(subtable, names='metal', data=np.full(subtable.size, z), usemask=False)
#        table.append(subtable)
#    table = np.concatenate(table)
#    return table
import time
from tqdm import tqdm
def measure_magnitudes(stars, filter_names, alpha=1, total=False, model='cb07', chunk=False, nchunk = int(1e5), nthread=1, verbose=False, dev=False):
    # measures absolute magnitude from stellar ages & metallicities
    filter_aliases = alias_dict()
    filter_aliases.update({
        'SDSS_u': 'u',
        'SDSS_g': 'g',
        'SDSS_r': 'r',
        'SDSS_i': 'i',
        'SDSS_z': 'z',
        'CFHT_u': 'u_1',
        'CFHT_g': 'g_1',
        'CFHT_r': 'r_1',
        'CFHT_i': 'i_1',
        'CFHT_y': 'y',
        'CFHT_z': 'z_1',
        'CFHT_Ks': 'Ks',
    })
    # measure magnitude from star data and population synthesis model.
    nstar = len(stars)
    if verbose: print(f"Construct `{model}` table for {nstar} stars...")
    if(model == 'cb07'):
        table = read_cb07_table()

        # log_ages = np.zeros(stars.size, 'f8')
        ages = stars['age', 'yr']
        with np.errstate(invalid='ignore'):
            log_ages = np.where(ages>0, np.log10(ages), -np.inf)
        log_metals = np.log10(stars['metal'])

        grid1 = table['logageyr']
        grid2 = np.log10(table['metal'])

        log_ages = set_boundaries(log_ages, [np.min(grid1), np.max(grid1)])
        log_metals = set_boundaries(log_metals, [np.min(grid2), np.max(grid2)])

        arr1 = log_ages
        arr2 = log_metals
        m_unit = 1.

    elif(model == 'YEPS'):
        table = read_YEPS_table(alpha)
        ages = np.log10(stars['age', 'Gyr'])
        # FeHs = stars['FeH']
        FeHs = 1.024*np.log10(stars['metal']) + 1.739

        ages[ages<=5.] = 5.
        FeHs[FeHs<=-2.5] = -2.5
        FeHs[FeHs>=0.5] = 0.5

        arr1 = np.log10(ages)
        arr2 = FeHs

        grid1 = np.log10(table['age'])
        grid2 = table['FeH']

        m_unit = 1E7

    else:
        raise ValueError("Unknown model '%s'" % model)
    stacked = np.stack([grid1, grid2], axis=-1)
    # msols = stars['m', 'Msol']
    mconvert = 2.5*np.log10(stars['m', 'Msol']/m_unit)
    magdict = {}
    ip = None
    for filter_name in filter_names:
        if(ip is None):
            try:
                ip = LinearNDInterpolator(stacked, table[filter_aliases[filter_name]], fill_value=np.nan)
            except:
                ip = LinearNDInterpolator(stacked, table[filter_name], fill_value=np.nan)
        else:
            ip.values = table[filter_aliases[filter_name]].reshape(-1,1).copy() # C-contiguous
        # NO CHUNK
        if(nstar < nthread*nchunk)or(not chunk):
            mags = ip(arr1, arr2)
        # YES CHUNK
        else:
            mags = np.zeros(nstar)
            njob = nstar//nchunk + 1
            # Multi-Procesing
            if (njob>1) and (nthread>1):
                now = datetime.datetime.now()
                shmname = f"mag_chunk_{now.strftime('%Y%m%d%H%M%S_%f')}"
                shm = shared_memory.SharedMemory(name=shmname,create=True, size=mags.nbytes)
                sarr = np.ndarray(mags.shape, dtype=mags.dtype, buffer=shm.buf)
                if verbose:
                    pbar = tqdm(total=njob, desc=f"{filter_name} MagChunk{min(njob, nthread)}")
                    def update(*a): pbar.update()
                else:
                    update = None
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                with Pool(min(njob, nthread)) as pool:
                    results = [
                        pool.apply_async(_magchunk, (ip, arr1[i:i+nchunk].view(), arr2[i:i+nchunk].view(), sarr.shape, shmname, i, nchunk), callback=update) for i in range(0, nstar, nchunk)
                        ]
                    for r in results:
                        r.get()
                mags = sarr.copy()
                if verbose: pbar.close()
                shm.close()
                shm.unlink()
            # Single Processing
            else:
                iterobj = range(0, nstar, nchunk)
                if verbose: iterobj = tqdm(iterobj, desc=f"{filter_name} MagChunk")
                for i in iterobj:
                    mags[i:i+nchunk] = ip(arr1[i:i+nchunk], arr2[i:i+nchunk])
        # mags = mags - 2.5*np.log10(msols/m_unit)
        mags -= mconvert
        if(total):
            mags = mags[~np.isnan(mags)]
            magdict[filter_name] = -2.5*np.log10(np.sum(10**(0.4*-mags)))
        else:
            magdict[filter_name] = mags
    return magdict

def _magchunk(ip, ar1, ar2, shape, shmname, i, nchunk):
    exist = shared_memory.SharedMemory(name=shmname)
    mags = np.ndarray(shape, dtype=np.float64, buffer=exist.buf)
    mags[i:i+nchunk] = ip(ar1, ar2)

def measure_magnitude(stars, filter_name, alpha=1, total=False, model='cb07'):
    # measures absolute magnitude from stellar ages & metallicities
    filter_aliases = alias_dict()
    filter_aliases.update({
        'SDSS_u': 'u',
        'SDSS_g': 'g',
        'SDSS_r': 'r',
        'SDSS_i': 'i',
        'SDSS_z': 'z',
        'CFHT_u': 'u_1',
        'CFHT_g': 'g_1',
        'CFHT_r': 'r_1',
        'CFHT_i': 'i_1',
        'CFHT_y': 'y',
        'CFHT_z': 'z_1',
        'CFHT_Ks': 'Ks',
    })
    # measure magnitude from star data and population synthesis model.
    if(model == 'cb07'):
        table = read_cb07_table()

        log_ages = np.zeros(stars.size, 'f8')
        ages = stars['age', 'yr']
        log_ages = np.where(ages>0, np.log10(ages), -np.inf)
        log_metals = np.log10(stars['metal'])

        grid1 = table['logageyr']
        grid2 = np.log10(table['metal'])

        log_ages = set_boundaries(log_ages, [np.min(grid1), np.max(grid1)])
        log_metals = set_boundaries(log_metals, [np.min(grid2), np.max(grid2)])

        arr1 = log_ages
        arr2 = log_metals
        m_unit = 1.

    elif(model == 'YEPS'):
        table = read_YEPS_table(alpha)
        ages = np.log10(stars['age', 'Gyr'])
        # FeHs = stars['FeH']
        FeHs = 1.024*np.log10(stars['metal']) + 1.739

        ages[ages<=5.] = 5.
        FeHs[FeHs<=-2.5] = -2.5
        FeHs[FeHs>=0.5] = 0.5

        arr1 = np.log10(ages)
        arr2 = FeHs

        grid1 = np.log10(table['age'])
        grid2 = table['FeH']

        m_unit = 1E7

    else:
        raise ValueError("Unknown model '%s'" % model)

    try:
        ip = LinearNDInterpolator(np.stack([grid1, grid2], axis=-1), table[filter_aliases[filter_name]], fill_value=np.nan)
    except:
        ip = LinearNDInterpolator(np.stack([grid1, grid2], axis=-1), table[filter_name], fill_value=np.nan)
    mags = ip(arr1, arr2)
    mags = mags - 2.5*np.log10(stars['m', 'Msol']/m_unit)
    if(total):
        mags = mags[~np.isnan(mags)]
        return -2.5*np.log10(np.sum(10**(0.4*-mags)))
    else:
        return mags

def measure_luminosity(stars, filter_name, **kwargs):
    mags = measure_magnitude(stars, filter_name, **kwargs)
    return 10**(-mags/2.5)

def ellipse_fit(coo, weights):
    # applies an ellipse fitting on the given set of data
    # returns a, b, position angle
    x, y = coo[..., 0], coo[..., 1]

    xb, yb = np.average(x, weights=weights), np.average(y, weights=weights)
    x2b = np.average(x**2, weights=weights)
    y2b = np.average(y**2, weights=weights)
    xyb = np.average(x*y, weights=weights) - xb*yb

    a = np.sqrt((x2b + y2b)/2 + np.sqrt(((x2b-y2b)/2)**2 + xyb**2))
    b = np.sqrt((x2b + y2b)/2 - np.sqrt(((x2b-y2b)/2)**2 + xyb**2))
    phi = np.arctan2(2*xyb, x2b-y2b)/2

    return a, b, phi

def absmag_to_mass(mag, filter_name='K'):
    if(filter_name == 'K'):
        # Cappellari 2013, eq2
        return 10**(10.58 - 0.44 * (mag+23))

def surface_brightness(absmag, area_pc2, log=True):
    if(log):
        return absmag + 2.5*np.log10(area_pc2) - 5 + 5 * np.log10(3600*180/np.pi)
    else:
        return absmag / area_pc2 / 100 / (3600*180/np.pi)**2

