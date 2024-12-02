import numpy as np
from rur import uri
from rur.sci.photometry import measure_magnitude,measure_luminosity, measure_magnitudes
from multiprocessing import shared_memory
import os, time
import warnings
from scipy.optimize import OptimizeWarning
#-------------------------------------------------------------------
# Configure
#-------------------------------------------------------------------
def datdump(data, path, msg=False):
    assert isinstance(data[0], np.ndarray), "Data should be numpy.ndarray"
    assert isinstance(data[1], str)
    leng = len(data[0])
    with open(path, "wb") as f:
        f.write(leng.to_bytes(4, byteorder='little'))
        f.write(data[0].tobytes())
        f.write(data[1].encode())
    if(msg): print(f" `{path}` saved")

def datload(path, msg=False):
    with open(path, "rb") as f:
        leng = int.from_bytes(f.read(4), byteorder='little')
        try:
            data = np.frombuffer(f.read(8*leng), dtype='f8')
        except:
            f.seek(4)
            data = np.frombuffer(f.read(4*leng), dtype='i4')
        name = f.read().decode()
    if(msg): print(f" `{path}` loaded")
    return data, name



def match_sim(text):
    ncs = ['newcluster', 'newcluster2', 'nc', 'new cluster']
    nhs = ['newhorizon', 'newhorizon', 'nh', 'new horizon']
    nh2s = ['newhorizon2', 'newhorizon2', 'nh2', 'new horizon 2']
    yzicss = ['yzics', 'yz']
    if(text.lower() in ncs): return 'nc'
    if(text.lower() in nhs): return 'nh'
    if(text.lower() in nh2s): return 'nh2'
    if(text.lower() in yzicss): return 'yzics'
    return None

inhouse = {
    'nc' : '/storage7/NewCluster',
    'nh' : '/storage6/NewHorizon',
    'nh2': '/storage7/NH2',
}

default_names = {
    'mcontam'      : ('mcontam', " - mcontam: Contaminated mass in solar mass unit"),
    'r200'         : ('r200', " - r200c: Halo radius where the mean density is 200 times the critical density"),
    'm200'         : ('m200', " - m200c: Halo mass within r200c in solar mass unit"),
    'r500'         : ('r500', " - r500c: Halo radius where the mean density is 500 times the critical density"),
    'm500'         : ('m500', " - m500c: Halo mass within r500c in solar mass unit"),
    'mstar'        : ('mstar', " - mstar_r: Stellar mass within Rmax in solar mass unit"),
    'mstar_rvir'   : ('mstar_rvir', " - mstar_rvir: Stellar mass within Rvir in solar mass unit"),
    'mstar_r200'   : ('mstar_r200', " - mstar_r200: Stellar mass within R200 in solar mass unit"),
    'mstar_r500'   : ('mstar_r500', " - mstar_r500: Stellar mass within R500 in solar mass unit"),
    'mgas'         : ('mgas', " - mgas_r: Gas mass within Rmax in solar mass unit"),
    'mgas_rvir'    : ('mgas_rvir', " - mgas_rvir: Gas mass within Rvir in solar mass unit"),
    'mgas_r200'    : ('mgas_r200', " - mgas_r200: Gas mass within R200 in solar mass unit"),
    'mgas_r500'    : ('mgas_r500', " - mgas_r500: Gas mass within R500 in solar mass unit"),
    'mcold'        : ('mcold', " - mcold_r: Cold(T<1e4 K) gas mass within Rmax in solar mass unit"),
    'mcold_rvir'   : ('mcold_rvir', " - mcold_rvir: Cold(T<1e4 K) gas mass within Rvir in solar mass unit"),
    'mcold_r200'   : ('mcold_r200', " - mcold_r200: Cold(T<1e4 K) gas mass within R200 in solar mass unit"),
    'mcold_r500'   : ('mcold_r500', " - mcold_r500: Cold(T<1e4 K) gas mass within R500 in solar mass unit"),
    'mdense'       : ('mdense', " - mdense_r: Dense(T<1e4K & rho>5H/cc) gas mass within Rmax in solar mass unit"),
    'mdense_rvir'  : ('mdense_rvir', " - mdense_rvir: Dense(T<1e4K & rho>5H/cc) gas mass within Rvir in solar mass unit"),
    'mdense_r200'  : ('mdense_r200', " - mdense_r200: Dense(T<1e4K & rho>5H/cc) gas mass within R200 in solar mass unit"),
    'mdense_r500'  : ('mdense_r500', " - mdense_r500: Dense(T<1e4K & rho>5H/cc) gas mass within R500 in solar mass unit"),
    'vmaxcir'      : ('vmaxcir', " - vmaxcir: Maximum circular velocity in km/s"),
    'rmaxcir'      : ('rmaxcir', " - rmaxcir: Radius where maximum circular velocity in code unit"),
    'cNFW'         : ('cNFW', " - cNFW: NFW concentration parameter"),
    'cNFWerr'      : ('cNFWerr', " - cNFWerr: Fitting error of NFW concentration parameter"),
    'inslope'      : ('inslope', " - inslope: Inner(<Rs) slope of the density profile"),
    'inslopeerr'   : ('inslopeerr', " - inslopeerr: Error of Inner(<Rs) slope"),
}

















#-------------------------------------------------------------------
# Main functions
#-------------------------------------------------------------------
def skip_func(path, iout, names, verbose):
    path_in_repo = 'halo'
    full_path = f"{path}/{path_in_repo}/extended/{iout:05d}"
    nnames = names.copy()

    # Contamination
    fname = f"{full_path}/mcontam_{iout:05d}.dat"
    if(os.path.exists(fname)):
        del nnames[f'mcontam']

    # Virials
    radsuffixs = ['', '_rvir', '_r200', '_r500']
    fname = f"{full_path}/m500_{iout:05d}.dat"
    if(os.path.exists(fname)):
        del nnames[f'r200']
        del nnames[f'm500']
        del nnames[f'r200']
        del nnames[f'm500']
    
    # Mass
    massprefixs = ['mstar', 'mgas', 'mcold', 'mdense']
    fname = f"{full_path}/mdense_r500_{iout:05d}.dat"
    if(os.path.exists(fname)):
        for prefix in massprefixs:
            for suffix in radsuffixs:
                del nnames[f'{prefix}{suffix}']

    # Profile
    fname = f"{full_path}/rmaxcir_{iout:05d}.dat"
    if(os.path.exists(fname)):
        del nnames[f'vmaxcir']
        del nnames[f'rmaxcir']

    fname = f"{full_path}/inslopeerr_{iout:05d}.dat"
    if(os.path.exists(fname)):
        del nnames[f'cNFW']
        del nnames[f'cNFWerr']
        del nnames[f'inslope']    
        del nnames[f'inslopeerr']
    return nnames


def pre_func(keys, table, snapm, members, snap, snapstar, dm_memory, star_memory, cell_memory, full_path, nthread, verbose):
    # r200, r500 should be calculated first
    newcols = {}

    needr200s = ['mstar','mcold','mgas','mdense']
    needr200 = True in np.isin(needr200s, keys, assume_unique=True)
    if(not 'r500' in keys)&(needr200): # r200, r500 already done
        if(verbose):
            print(f" [PreFunc] > Prepare r200, r500", end='\t'); ref = time.time()
        r200s = datload(f"{full_path}/r200_{snap.iout:05d}.dat", msg=False)[0]
        if(len(r200s) != len(table)): r200s = r200s[table['id']-1] # Partitioned
        r500s = datload(f"{full_path}/r500_{snap.iout:05d}.dat", msg=False)[0]
        if(len(r500s) != len(table)): r500s = r500s[table['id']-1] # Partitioned
        odtype = table.dtype
        ndtype = odtype.descr + [('r200','f8'), ('r500','f8')]
        ntable = np.empty(len(table), dtype=ndtype)
        for name in odtype.names: ntable[name] = table[name]
        ntable['r200'] = r200s
        ntable['r500'] = r500s
        table = ntable
        if(verbose): print(f" Done ({time.time()-ref:.2f} sec)")
    
    return table, snapm, members, snap, snapstar, dm_memory, star_memory, cell_memory























debug = False
nostar = False
# This is used in multiprocessing
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
def calc_func(i, halo, shape, address, dtype, sparams, sunits, members, dm_memory, star_memory, cell_memory, cdomain):
    # Common
    global nostar
    exist = shared_memory.SharedMemory(name=address)
    result_table = np.ndarray(shape, dtype=dtype, buffer=exist.buf)

    memdist=None; memmass=None
    # Mcontam
    if('mcontam' in result_table.dtype.names):
        if('mcontam' in halo.dtype.names): mcontam = halo['mcontam']
        else:
            # 1305211 in NH,NH2,NC
            memmass = members['m'] / sunits['Msol']
            mcontam = np.sum(memmass[memmass > 1500000])
        result_table['mcontam'][i] = mcontam
    
    # Circular Velocity
    if('rmaxcir' in result_table.dtype.names):
        memdist = np.sqrt( (members['x']-halo['x'])**2 + (members['y']-halo['y'])**2 + (members['z']-halo['z'])**2 )
        if(memmass is None): memmass = members['m'] / sunits['Msol']
        argsort = np.argsort(memdist)
        memdist = memdist[argsort]; memmass = memmass[argsort]
        nonzero = memdist>0; memdist = memdist[nonzero]; memmass = memmass[nonzero]
        G = 4.30091e-3 # pc Msun^-1 (km/s)^2
        d_pc = memdist / sunits['pc']
        vcir = np.sqrt(G*np.cumsum(memmass)/d_pc)
        cumsum = np.cumsum(memmass)
        smooth = gaussian_filter1d(cumsum, 3)
        derivative = np.gradient(smooth)
        argmax = np.argmax(derivative)
        rmaxcir = memdist[argmax]*sunits['pc']
        vmaxcir = vcir[argmax]
        result_table['vmaxcir'][i] = vmaxcir
        result_table['rmaxcir'][i] = rmaxcir

    # NFW profile
    if('inslopeerr' in result_table.dtype.names):
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        def NFW(r, rs, rho0):
            x = r / rs
            return rho0 / (x * (1 + x)**2)
        if(memdist is None):
            memdist = np.sqrt( (members['x']-halo['x'])**2 + (members['y']-halo['y'])**2 + (members['z']-halo['z'])**2 )
        if(memmass is None):
            memmass = members['m'] / sunits['Msol']
            argsort = np.argsort(memdist)        
            memdist = memdist[argsort]; memmass = memmass[argsort]
            nonzero = memdist>0; memdist = memdist[nonzero]; memmass = memmass[nonzero]
        bins = np.logspace(np.log10(memdist[0]), np.log10(memdist[-1]), 2*int(np.log10(halo['m'])))
        cmas = np.histogram(memdist, bins, weights=memmass)[0]
        nums = np.histogram(memdist, bins)[0]
        r = (bins[1:] + bins[:-1]) / 2
        mask = nums > 0
        r = r[mask]; cmas = cmas[mask]; nums = nums[mask]
        poisson_error = 1/np.sqrt(nums)
        rho = cmas / (4/3*np.pi*r**3)
        rvir = halo['rvir']
        try:
            popt, pcov = curve_fit(NFW, r, rho, p0=[halo['r'], rho[0]], sigma=poisson_error, absolute_sigma=True)
            rs = popt[0]; rserr = np.sqrt(pcov[0,0])
            cNFW = rvir / rs
            cNFWerr = cNFW * np.sqrt((rserr/rs)**2)
        except:
            cNFW = np.nan; cNFWerr = np.nan
        result_table['cNFW'][i] = cNFW
        result_table['cNFWerr'][i] = cNFWerr
        
        linefit = lambda x, a, b: a*x + b
        def getsingle(x, y, n):
            poisson_error = 1 / np.sqrt(n)
            popt, pcov = curve_fit(linefit, x, y, sigma=poisson_error, absolute_sigma=False)#, bounds=([-6,2], [-np.inf, np.inf]))
            return popt[0], np.sqrt(np.diag(pcov))[0] # slope, error
        rcut = min(rs,rvir) if not np.isnan(cNFW) else rvir
        mask = memdist < rcut
        slope, error = np.nan, np.nan
        if(np.sum(mask)>0):
            try:
                slope, error = getsingle(np.log10(memdist[mask]), np.log10(memmass[mask]), np.sqrt(memmass[mask]))
            except:
                slope, error = np.nan, np.nan
        result_table['inslope'][i] = slope
        result_table['inslopeerr'][i] = error
    
    cells = None; cdist = None; cellmass=None
    def _get_cell(halo, cells, cdist, cell_memory, return_dist=False):
        if cells is None:
            cx=halo['x']; cy=halo['y']; cz=halo['z']
            cshape, caddress, cdtype, cpulist_cell, bound_cell = cell_memory
            cexist = shared_memory.SharedMemory(name=caddress)
            allcells = np.ndarray(cshape, dtype=cdtype, buffer=cexist.buf)
            domcells = uri.domain_slice(allcells, cdomain, cpulist_cell, bound_cell)
            cdist = np.sqrt( (domcells['x']-cx)**2 + (domcells['y']-cy)**2 + (domcells['z']-cz)**2 )
            rmask = cdist <= halo['r']
            if(np.sum(rmask) < 8): rmask = cdist < (halo['r'] + (1 / 2**domcells['level'])/2)
            cells = domcells[rmask]; cdist = cdist[rmask]
        if(return_dist): return cells, cdist
        return cells
    dms = None; ddist = None
    def _get_dm(halo, dms, ddist, dm_memory, return_dist=False):
        if dms is None:
            cx=halo['x']; cy=halo['y']; cz=halo['z']
            dshape, daddress, ddtype, cpulist_dm, bound_dm = dm_memory
            dexist = shared_memory.SharedMemory(name=daddress)
            alldms = np.ndarray(dshape, dtype=ddtype, buffer=dexist.buf)
            domdms = uri.domain_slice(alldms, cdomain, cpulist_dm, bound_dm)
            ddist = np.sqrt( (domdms['x']-cx)**2 + (domdms['y']-cy)**2 + (domdms['z']-cz)**2 )
            dmask = ddist <= halo['r']
            dms = domdms[dmask]; ddist = ddist[dmask]
        if(return_dist): return dms, ddist
        return dms
    stars = None; sdist = None
    def _get_star(halo, stars, sdist, star_memory, return_dist=False):
        global nostar
        if stars is None:
            sshape, saddress, sdtype, cpulist_star, bound_star = star_memory
            if sshape is None:
                nostar=True
            else:
                cx=halo['x']; cy=halo['y']; cz=halo['z']
                sexist = shared_memory.SharedMemory(name=saddress)
                allstars = np.ndarray(sshape, dtype=sdtype, buffer=sexist.buf)
                domstars = uri.domain_slice(allstars, cdomain, cpulist_star, bound_star)
                sdist = np.sqrt( (domstars['x']-cx)**2 + (domstars['y']-cy)**2 + (domstars['z']-cz)**2 )
                smask = sdist <= halo['r']
                stars = domstars[smask]; sdist = sdist[smask]
        if(return_dist): return stars, sdist
        return stars

    # R200, M200, R500, M500
    needr200s = ['r500','mstar','mcold','mgas','mdense']
    needr200 = True in np.isin(needr200s, result_table.dtype.names, assume_unique=True)
    if needr200:
        if('r500' in result_table.dtype.names):
            H0 = sparams['H0']; aexp=sparams['aexp']; kpc=sunits['kpc']
            # critical density
            H02 = (H0 * 3.24078e-20)**2 # s-2
            G = 6.6743e-11 # N m2 kg-2 = kg m s-2 m2 kg-2 = m3 s-2 kg-1
            rhoc = 3 * H02 /8 /np.pi /G # kg m-3
            rhoc *= 5.02785e-31  * (3.086e+19)**3 # Msol ckpc-3
            rhoc /= (aexp**3) # Msol pkpc-3

            # Sorting
            cells, cdist = _get_cell(halo, cells, cdist, cell_memory, return_dist=True)
            stars, sdist = _get_star(halo, stars, sdist, star_memory, return_dist=True)
            dms, ddist = _get_dm(halo, dms, ddist, dm_memory, return_dist=True)
            dx = 1 / 2**cells['level']; vol = dx**3
            cellmass = cells['rho']*vol / sunits['Msol']
            if nostar:
                dis = np.hstack((cdist,ddist))/kpc # pkpc
                mas = np.hstack((cellmass,dms['m']/sunits['Msol'])) # Msol
            else:
                dis = np.hstack((cdist,sdist,ddist))/kpc # pkpc
                mas = np.hstack((cellmass,stars['m']/sunits['Msol'],dms['m']/sunits['Msol'])) # Msol
            argsort = np.argsort(dis)
            dis = dis[argsort] # pkpc
            mas = mas[argsort] # Msol

            # Inside density
            cmas = np.cumsum(mas) # Msol
            vols = 4/3*np.pi * dis**3 # pkpc^3
            rhos = cmas / vols # Msol pkpc-3
            arg = np.argmin(np.abs(rhos - 200*rhoc))
            r200 = dis[arg] # pkpc
            if(r200 >= np.max(dis))or(r200 <= np.min(dis)): r200 = np.nan
            m200 = cmas[arg] # Msol
            result_table['r200'][i] = r200
            result_table['m200'][i] = m200
            arg = np.argmin(np.abs(rhos - 500*rhoc))
            r500 = dis[arg] # pkpc
            if(r500 >= np.max(dis))or(r500 <= np.min(dis)): r500 = np.nan
            m500 = cmas[arg] # Msol
            result_table['r500'][i] = r500
            result_table['m500'][i] = m500
        else:
            r200 = halo['r200']; r500 = halo['r500']

    # Mstar
    if('mstar_r500' in result_table.dtype.names):
        stars, sdist = _get_star(halo, stars, sdist, star_memory, return_dist=True)
        for suffix in ['','_rvir','_r200','_r500']:
            rname = suffix.replace('_','') if suffix != '' else 'r'
            radius = halo[rname] if rname in halo.dtype.names else result_table[i][rname]
            if nostar:
                result_table[f'mstar{suffix}'][i] = 0
            else:
                mask = sdist < radius
                result_table[f'mstar{suffix}'][i] = np.sum(stars['m'][mask]) / sunits['Msol']
            
    # Cell mass
    if('mgas_r500' in result_table.dtype.names):
        cells, cdist = _get_cell(halo, cells, cdist, cell_memory, return_dist=True)
        if cellmass is None:
            dx = 1 / 2**cells['level']; vol = dx**3
            cellmass = cells['rho']*vol / sunits['Msol']
        for suffix in ['','_rvir','_r200','_r500']:
            rname = suffix.replace('_','') if suffix != '' else 'r'
            radius = halo[rname] if rname in halo.dtype.names else result_table[i][rname]
            mask = cdist < radius
            result_table[f'mgas{suffix}'][i] = np.sum(cellmass[mask])

    # temperature
    if('mdense_r500' in result_table.dtype.names):
        cells, cdist = _get_cell(halo, cells, cdist, cell_memory, return_dist=True)
        if cellmass is None:
            dx = 1 / 2**cells['level']; vol = dx**3
            cellmass = cells['rho']*vol / sunits['Msol']
        T = cells['P']/cells['rho'] / sunits['K']
        cold = T < 1e4
        dense = (cells['rho'] / sunits['H/cc'] > 5) & (cold)
        for suffix in ['','_rvir','_r200','_r500']:
            rname = suffix.replace('_','') if suffix != '' else 'r'
            radius = halo[rname] if rname in halo.dtype.names else result_table[i][rname]
            mask = cdist < radius
            result_table[f'mcold{suffix}'][i] = np.sum(cellmass[mask & cold])
            result_table[f'mdense{suffix}'][i] = np.sum(cellmass[mask & dense])



def dump_func(result_table, table, full_path, iout, names, verbose, izip, partition):
    ZIP = partition>0
    nzip = 2**partition if ZIP else 1
    suffix = f"{izip:02d}p{nzip}" if ZIP else ""
    for key, val in names.items():
        title = val[0]
        desc = val[1]
        datdump((result_table[key], desc), f"{full_path}/{title}_{iout:05d}.dat{suffix}", msg=verbose)
    if(ZIP): datdump((table['id'], "IDlist"), f"{full_path}/zipID_{iout:05d}.dat{suffix}", msg=verbose)
    if(izip+1 == nzip)and(ZIP):
        if(verbose): print(f" [DumpFunc] > Zipping"); ref = time.time()
        
        # Get table IDs
        zipids={}; ntable = 0
        for jzip in range(nzip):
            suffix = f"{jzip:02d}p{nzip}"
            ids = datload(f"{full_path}/zipID_{iout:05d}.dat{suffix}", msg=False)[0]
            zipids[jzip] = ids
            ntable += len(ids)
        
        # Merge
        for key, val in names.items():
            title = val[0]
            desc = val[1]
            zipped_table = np.zeros(ntable)
            for jzip in range(nzip):
                suffix = f"{jzip:02d}p{nzip}"
                data = datload(f"{full_path}/{title}_{iout:05d}.dat{suffix}", msg=False)[0]
                ids = zipids[jzip]
                zipped_table[ids-1] = data
            datdump((zipped_table, desc), f"{full_path}/{title}_{iout:05d}.dat", msg=True)
        
        # Delete
        for jzip in range(nzip):
            suffix = f"{jzip:02d}p{nzip}"
            os.remove(f"{full_path}/zipID_{iout:05d}.dat{suffix}")
            if(verbose): print(f" [DumpFunc] > Remove `zipID_{iout:05d}.dat{suffix}`")
            for key, val in names.items():
                title = val[0]
                os.remove(f"{full_path}/{title}_{iout:05d}.dat{suffix}")
                if(verbose): print(f" [DumpFunc] > Remove `{title}_{iout:05d}.dat{suffix}`")

        if(verbose): print(f" Done ({time.time()-ref:.2f} sec)")