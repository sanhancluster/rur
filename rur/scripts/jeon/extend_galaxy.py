import numpy as np
from rur import uri
from rur.sci.photometry import measure_magnitude,measure_luminosity, measure_magnitudes
from multiprocessing import shared_memory
import os, time
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
        data = np.frombuffer(f.read(8*leng), dtype='f8')
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

global_bands = ['u', 'g', 'r', 'i', 'z']
default_names = {
    'umag'          :('umag', " - umag: SDSS u-band magnitude from Charlot & Bruzual (2007) model"),
    'gmag'          :('gmag', " - gmag: SDSS g-band magnitude from Charlot & Bruzual (2007) model"),
    'rmag'          :('rmag', " - rmag: SDSS r-band magnitude from Charlot & Bruzual (2007) model"),
    'imag'          :('imag', " - imag: SDSS i-band magnitude from Charlot & Bruzual (2007) model"),
    'zmag'          :('zmag', " - zmag: SDSS z-band magnitude from Charlot & Bruzual (2007) model"),
    'r50'           : ('r50', " - r50: 50% mass (of member particles) radius in code unit"),
    'r90'           : ('r90', " - r90: 90% mass (of member particles) radius in code unit"),
    'r50u'          :('r50u', " - r50u: 50% SDSS u-band light (of member particles) radius in code unit"),
    'r50g'          :('r50g', " - r50g: 50% SDSS g-band light (of member particles) radius in code unit"),
    'r50r'          :('r50r', " - r50r: 50% SDSS r-band light (of member particles) radius in code unit"),
    'r50i'          :('r50i', " - r50i: 50% SDSS i-band light (of member particles) radius in code unit"),
    'r50z'          :('r50z', " - r50z: 50% SDSS z-band light (of member particles) radius in code unit"),
    'r90u'          :('r90u', " - r90u: 90% SDSS u-band light (of member particles) radius in code unit"),
    'r90g'          :('r90g', " - r90g: 90% SDSS g-band light (of member particles) radius in code unit"),
    'r90r'          :('r90r', " - r90r: 90% SDSS r-band light (of member particles) radius in code unit"),
    'r90i'          :('r90i', " - r90i: 90% SDSS i-band light (of member particles) radius in code unit"),
    'r90z'          :('r90z', " - r90z: 90% SDSS z-band light (of member particles) radius in code unit"),
    'sfr'           :('SFR',          " - SFR: SFR [Msol/yr] in 100 Myr timewindows, using all member stars"),
    'sfr_r50'       :('SFR_r50',      " - SFR_r50: SFR [Msol/yr] in 100 Myr timewindows, using member stars within r50"),
    'sfr_r90'       :('SFR_r90',      " - SFR_r90: SFR [Msol/yr] in 100 Myr timewindows, using member stars within r90"),
    'sfr10'         :('SFR10',        " - SFR10: SFR [Msol/yr] in 10 Myr timewindows, using all member stars"),
    'sfr10_r50'     :('SFR10_r50',    " - SFR10_r50: SFR [Msol/yr] in 10 Myr timewindows, using member stars within r50"),
    'sfr10_r90'     :('SFR10_r90',    " - SFR10_r90: SFR [Msol/yr] in 10 Myr timewindows, using member stars within r90"),
    'age'           :('age',  " - age: mass-weighted age in Gyr"),
    'ageu'          :('ageu', " - ageu: mass-weighted age in Gyr for SDSS u-band luminosity"),
    'ageg'          :('ageg', " - ageg: mass-weighted age in Gyr for SDSS g-band luminosity"),
    'ager'          :('ager', " - ager: mass-weighted age in Gyr for SDSS r-band luminosity"),
    'agei'          :('agei', " - agei: mass-weighted age in Gyr for SDSS i-band luminosity"),
    'agez'          :('agez', " - agez: mass-weighted age in Gyr for SDSS z-band luminosity"),
    'vsig'          :('vsig',         " - vsig: (Rotational_velocity)/(Velocity_dispersion) of star (mass-weighted)",),
    'vsig_r50'      :('vsig_r50',     " - vsig_r50: (Rotational_velocity)/(Velocity_dispersion) of star (within r50) (mass-weighted)",),
    'vsig_r90'      :('vsig_r90',     " - vsig_r90: (Rotational_velocity)/(Velocity_dispersion) of star (within r50) (mass-weighted)",),
    'metal'         :('metal',        " - metal: Metallicity (Mass fraction) in star",),
    'vsig_gas'      :('vsig_gas',     " - vsig_gas: (Rotational_velocity)/(Velocity_dispersion) of gas (mass-weighted)"),
    'vsig_gas_r50'  :('vsig_gas_r50', " - vsig_gas_r50: (Rotational_velocity)/(Velocity_dispersion) of gas (within r50) (mass-weighted)"),
    'vsig_gas_r90'  :('vsig_gas_r90', " - vsig_gas_r90: (Rotational_velocity)/(Velocity_dispersion) of gas (within r50) (mass-weighted)"),
    'metal_gas'     :('metal_gas',    " - metal_gas: Metallicity (Mass fraction) in gas"),
    'mgas'          :('M_gas',            " - M_gas: Gas mass within Rmax in solar mass unit"),
    'mgas_r50'      :('M_gas_r50',        " - M_gas_r50: Gas mass within R50 in solar mass unit"),
    'mgas_r90'      :('M_gas_r90',        " - M_gas_r90: Gas mass within R90 in solar mass unit"),
    'mcold'         :('Mcold_gas',        " - Mcold_gas: Cold(T<1e4 K) gas mass within Rmax in solar mass unit"),
    'mcold_r50'     :('Mcold_gas_r50',    " - Mcold_gas_r50: Cold(T<1e4 K) gas mass within R50 in solar mass unit"),
    'mcold_r90'     :('Mcold_gas_r90',    " - Mcold_gas_r90: Cold(T<1e4 K) gas mass within R90 in solar mass unit"),
    'mdense'        :('Mdense_gas',       " - Mdense_gas: Dense(T<1e4K & rho>5H/cc) gas mass within Rmax in solar mass unit"),
    'mdense_r50'    :('Mdense_gas_r50',   " - Mdense_gas_r50: Dense(T<1e4K & rho>5H/cc) gas mass within R50 in solar mass unit"),
    'mdense_r90'    :('Mdense_gas_r90',   " - Mdense_gas_r90: Dense(T<1e4K & rho>5H/cc) gas mass within R90 in solar mass unit"),
    'H'             :('H_gas',    " - H_gas: Chem(H) mass fraction in gas"),
    'O'             :('O_gas',    " - O_gas: Chem(O) mass fraction in gas"),
    'Fe'            :('Fe_gas',   " - Fe_gas: Chem(Fe) mass fraction in gas"),
    'Mg'            :('Mg_gas',   " - Mg_gas: Chem(Mg) mass fraction in gas"),
    'C'             :('C_gas',    " - C_gas: Chem(C) mass fraction in gas"),
    'N'             :('N_gas',    " - N_gas: Chem(N) mass fraction in gas"),
    'Si'            :('Si_gas',   " - Si_gas: Chem(Si) mass fraction in gas"),
    'S'             :('S_gas',    " - S_gas: Chem(S) mass fraction in gas"),
    'D'             :('D_gas',    " - D_gas: Chem(D) mass fraction in gas"),
    'd1'            :('CDustSmall_gas',   " - CDustSmall_gas: Carbonaneous small dust(0.005μm) mass fraction"),
    'd2'            :('CDustLarge_gas',   " - CDustLarge_gas: Carbonaneous large dust(0.1μm) mass fraction"),
    'd3'            :('SiDustSmall_gas',  " - SiDustSmall_gas: Silicate small dust(0.005μm) mass fraction"),
    'd4'            :('SiDustLarge_gas',  " - SiDustLarge_gas: Silicate large dust(0.1μm) mass fraction"),
    'SBu'           :('SBu',      " - SBu: Surface Brightness [mag / arcsec2] (within Rmax) in u-band"),
    'SBu_r50'       :('SBu_r50',  " - SBu_r90: Surface Brightness [mag / arcsec2] (within R90) in u-band"),
    'SBu_r90'       :('SBu_r90',  " - SBu_r50: Surface Brightness [mag / arcsec2] (within R50) in u-band"),
    'SBg'           :('SBg',      " - SBg: Surface Brightness [mag / arcsec2] (within Rmax) in g-band"),
    'SBg_r50'       :('SBg_r50',  " - SBg_r90: Surface Brightness [mag / arcsec2] (within R90) in g-band"),
    'SBg_r90'       :('SBg_r90',  " - SBg_r50: Surface Brightness [mag / arcsec2] (within R50) in g-band"),
    'SBr'           :('SBr',      " - SBr: Surface Brightness [mag / arcsec2] (within Rmax) in r-band"),
    'SBr_r50'       :('SBr_r50',  " - SBr_r90: Surface Brightness [mag / arcsec2] (within R90) in r-band"),
    'SBr_r90'       :('SBr_r90',  " - SBr_r50: Surface Brightness [mag / arcsec2] (within R50) in r-band"),
    'SBi'           :('SBi',      " - SBi: Surface Brightness [mag / arcsec2] (within Rmax) in i-band"),
    'SBi_r50'       :('SBi_r50',  " - SBi_r90: Surface Brightness [mag / arcsec2] (within R90) in i-band"),
    'SBi_r90'       :('SBi_r90',  " - SBi_r50: Surface Brightness [mag / arcsec2] (within R50) in i-band"),
    'SBz'           :('SBz',      " - SBz: Surface Brightness [mag / arcsec2] (within Rmax) in z-band"),
    'SBz_r50'       :('SBz_r50',  " - SBz_r90: Surface Brightness [mag / arcsec2] (within R90) in z-band"),
    'SBz_r90'       :('SBz_r90',  " - SBz_r50: Surface Brightness [mag / arcsec2] (within R50) in z-band"),
    'MBH'           :('MBH', " - MBH: SMBH mass in solar mass unit"),
    'dBH'           :('dBH', " - dBH: Distance between SMBH and halo center in code unit"),
}


#-------------------------------------------------------------------------------------
# Miscellanea
zero_mags = { 'SDSS': {'u': 24.63, 'g': 25.11, 'r': 24.8, 'i': 24.36, 'z': 22.83}, }
def mag2flux(mag): return 10**(-0.4*mag)
def asinh_mag(flux, band):
    softening = {'u': 1.4e-10, 'g': 0.9e-10, 'r': 1.2e-10, 'i': 1.8e-10, 'z': 7.4e-10}
    b = softening[band]
    zeromag = zero_mags['SDSS'][band]
    f0 = mag2flux(zeromag)
    mag = -2.5/np.log(10) * ( np.arcsinh(flux/f0/2/b) + np.log(b) )
    return mag + zeromag
muMg=24.3050
muFe=55.8450
muSi=28.0855
muO =15.9990
nsilMg=1.0
nsilFe=1.0
nsilSi=1.0
nsilO=4.0
numtot=muMg*nsilMg+muFe*nsilFe+muSi*nsilSi+muO*nsilO
SioverSil=muSi*nsilSi/numtot
# small large small large (from nml)
# asize=0.005d0,0.1d0,0.005d0,0.1d0 -> grain size in microns
# sgrain=2.2d0,2.2d0,3.3d0,3.3d0 -> grain material density in g/cm^3
# d1: C_dust_small, d2: C_dust_large, d3: Si_dust_small, d4: Si_dust_large
#-------------------------------------------------------------------------------------



















#-------------------------------------------------------------------
# Main functions
#-------------------------------------------------------------------
def skip_func(path, iout, names, verbose):
    path_in_repo = 'galaxy'
    full_path = f"{path}/{path_in_repo}/extended/{iout:05d}"
    nnames = names.copy()

    # Radius
    bandsuffixs = ['', 'u', 'g', 'r', 'i', 'z']
    radsuffixs = ['', '_r50', '_r90']
    fname = f"{full_path}/r90z_{iout:05d}.dat"
    if(os.path.exists(fname)):
        for suffix in bandsuffixs:
            # if(verbose): print(f" [SkipFunc] > No need r50{suffix}, r90{suffix}")
            del nnames[f'r50{suffix}']
            del nnames[f'r90{suffix}']
    # SFR
    fname = f"{full_path}/SFR_r90_{iout:05d}.dat"
    if(os.path.exists(fname)):
        for suffix in radsuffixs:
            # if(verbose): print(f" [SkipFunc] > No need sfr{suffix}")
            del nnames[f'sfr{suffix}']
    # SFR10
    fname = f"{full_path}/SFR10_r90_{iout:05d}.dat"
    if(os.path.exists(fname)):
        for suffix in radsuffixs:
            del nnames[f'sfr10{suffix}']
    # Mag
    fname = f"{full_path}/zmag_{iout:05d}.dat"
    if(os.path.exists(fname)):
        for suffix in bandsuffixs[1:]:
            del nnames[f'{suffix}mag']
    # Age
    fname = f"{full_path}/agez_{iout:05d}.dat"
    if(os.path.exists(fname)):
        for suffix in bandsuffixs:
            del nnames[f'age{suffix}']
    # V/sigma
    fname = f"{full_path}/vsig_r90_{iout:05d}.dat"
    if(os.path.exists(fname)):
        for suffix in radsuffixs:
            del nnames[f'vsig{suffix}']
    # Metal
    fname = f"{full_path}/metal_{iout:05d}.dat"
    if(os.path.exists(fname)):
        del nnames[f'metal']
    # Hydro
    hkeys = []
    hvals = []
    clists = ['H', 'O', 'Fe', 'Mg', 'C', 'N', 'Si', 'S', 'D', 'd1','d2','d3','d4']
    for key,val in nnames.items():
        if('gas' in val[0]):
            hkeys.append(key)
            hvals.append(val[0])
    for hkey, hval in zip(hkeys, hvals):
        fname = f"{full_path}/{hval}_{iout:05d}.dat"
        if(os.path.exists(fname)):
            if hkey in clists:
                data = datload(fname, msg=False)[0]
                if(np.sum(data)>0):
                    del nnames[hkey]
            else:
                del nnames[hkey]
    # SB
    fname = f"{full_path}/SBz_r90_{iout:05d}.dat"
    if(os.path.exists(fname)):
        for bsuffix in bandsuffixs[1:]:
            for rsuffix in radsuffixs:
                del nnames[f'SB{bsuffix}{rsuffix}']
    # BH
    fname = f"{full_path}/dBH_{iout:05d}.dat"
    if(os.path.exists(fname)):
        del nnames[f'MBH']
        del nnames[f'dBH']
    
    return nnames














def pre_func(keys, table, snapm, members, snap, part_memory, cell_memory, full_path, nthread, verbose):
    # r50, r90 should be calculated first
    newcols = {}

    needr50s = ['sfr','sfr10','mgas','mcold','mdense','vsig_gas','vsig','SBz']
    needr50 = True in np.isin(needr50s, keys, assume_unique=True)
    if(not 'r90' in keys)&(needr50): # r50, r90 already done
        if(verbose):
            print(f" [PreFunc] > Prepare r50, r90", end='\t'); ref = time.time()
        r50s = datload(f"{full_path}/r50_{snap.iout:05d}.dat", msg=False)[0]
        r90s = datload(f"{full_path}/r90_{snap.iout:05d}.dat", msg=False)[0]
        odtype = table.dtype
        ndtype = odtype.descr + [('r50','f8'), ('r90','f8')]
        ntable = np.empty(len(table), dtype=ndtype)
        for name in odtype.names:
            ntable[name] = table[name]
        ntable['r50'] = r50s
        ntable['r90'] = r90s
        table = ntable
        if(verbose): print(f" Done ({time.time()-ref:.2f} sec)")
    
    needmags = ['zmag','SBz']
    needmag = True in np.isin(needmags, keys, assume_unique=True)
    umag=None; gmag=None; rmag=None; imag=None; zmag=None
    if(needmag):
        if(verbose):
            print(f" [PreFunc] > Prepare magnitudes", end='\t'); ref = time.time()
        with np.errstate(divide='ignore', invalid='ignore'):
            mags = measure_magnitudes(members, global_bands, chunk=True, nthread=nthread)
        umag = mags['u']
        gmag = mags['g']
        rmag = mags['r']
        imag = mags['i']
        zmag = mags['z']
        newcols['umag'] = (umag, ('umag', 'f8'))
        newcols['gmag'] = (gmag, ('gmag', 'f8'))
        newcols['rmag'] = (rmag, ('rmag', 'f8'))
        newcols['imag'] = (imag, ('imag', 'f8'))
        newcols['zmag'] = (zmag, ('zmag', 'f8'))
        if(verbose): print(f" Done ({time.time()-ref:.2f} sec)")
    
    needlums = ['age','r90z']
    needlum = True in np.isin(needlums, keys, assume_unique=True)
    if(needlum):
        if(verbose):
            print(f" [PreFunc] > Prepare luminosity", end='\t'); ref = time.time()
        ulum = measure_luminosity(members, 'SDSS_u') if(umag is None) else 10**(-umag/2.5)
        glum = measure_luminosity(members, 'SDSS_g') if(gmag is None) else 10**(-gmag/2.5)
        rlum = measure_luminosity(members, 'SDSS_r') if(rmag is None) else 10**(-rmag/2.5)
        ilum = measure_luminosity(members, 'SDSS_i') if(imag is None) else 10**(-imag/2.5)
        zlum = measure_luminosity(members, 'SDSS_z') if(zmag is None) else 10**(-zmag/2.5)
        newcols['ulum'] = (ulum, ('ulum', 'f8'))
        newcols['glum'] = (glum, ('glum', 'f8'))
        newcols['rlum'] = (rlum, ('rlum', 'f8'))
        newcols['ilum'] = (ilum, ('ilum', 'f8'))
        newcols['zlum'] = (zlum, ('zlum', 'f8'))
        if(verbose): print(f" Done ({time.time()-ref:.2f} sec)")


    needages = ['sfr','sfr10','age']
    needage = True in np.isin(needages, keys, assume_unique=True)
    if(needage):
        if(verbose):
            print(f" [PreFunc] > Prepare age", end='\t'); ref = time.time()
        age = members['age','Gyr']
        newcols['age'] = (age, ('age', 'f8'))
        if(verbose): print(f" Done ({time.time()-ref:.2f} sec)")
    colkeys = list(newcols.keys())
    if(len(colkeys)>0):
        if(verbose):
            print(f" [PreFunc] > New Array", end='\t'); ref = time.time()
        odtype = members.dtype
        ndtype = odtype.descr + [newcols[colkey][1] for colkey in colkeys]
        nmembers = np.empty(len(members), dtype=ndtype)
        for name in odtype.names:
            nmembers[name] = members[name]
        for colkey in colkeys:
            col = newcols[colkey]
            nmembers[colkey] = col[0]
        nmembers = uri.Particle(nmembers, snapm)
        members = nmembers
        if(verbose): print(f" Done ({time.time()-ref:.2f} sec)")
    return table, snapm, members, snap, part_memory, cell_memory













debug = True
# This is used in multiprocessing
def calc_func(i, halo, shape, address, dtype, sparams, sunits, members, part_memory, cell_memory, cdomain, send=None):
    # Common
    exist = shared_memory.SharedMemory(name=address)
    result_table = np.ndarray(shape, dtype=dtype, buffer=exist.buf)

    needdists = ['r50','r90','sfr','sfr10','r90z','vsig','SBz']
    needdist = True in np.isin(needdists, result_table.dtype.names, assume_unique=True)
    if(needdist):
        dist = np.sqrt( (members['x']-halo['x'])**2 + (members['y']-halo['y'])**2 + (members['z']-halo['z'])**2 )
        argsort = np.argsort(dist)
        dist = dist[argsort]
        members = members[argsort]
        memmass = members['m'] / sunits['Msol']
        

    # R50, R90
    needr50s = ['r90', 'sfr','sfr10','mgas','mcold','mdense','vsig_gas','vsig','SBz']
    needr50 = True in np.isin(needr50s, result_table.dtype.names, assume_unique=True)
    if needr50:
        if('r90' in result_table.dtype.names):
            cmas = np.cumsum(memmass)
            r50 = dist [np.argmin(np.abs(cmas - cmas[-1]*0.5)) ]
            result_table['r50'][i] = r50
            r90 = dist[ np.argmin(np.abs(cmas - cmas[-1]*0.9)) ]
            result_table['r90'][i] = r90
        else:
            r50 = halo['r50']; r90 = halo['r90']
    # Reff
    if('r90z' in result_table.dtype.names):
        for band in global_bands:
            light = members[f"{band}lum"]
            light = light[argsort]
            cmas = np.cumsum(light)
            result_table[f'r50{band}'][i] = dist[ np.argmin(np.abs(cmas - cmas[-1]*0.5)) ]
            result_table[f'r90{band}'][i] = dist[ np.argmin(np.abs(cmas - cmas[-1]*0.9)) ]

    # SFR100
    if('sfr' in result_table.dtype.names):
        sfr=0; sfr_r50=0; sfr_r90=0
        young = members['age'] < 0.1
        if(True in young):
            sfr = np.sum(memmass[young]) / 1e8 # Msol/yr
            young = young & (dist < r90)
            if(True in young):
                sfr_r90 = np.sum(memmass[young]) / 1e8
                young = young & (dist < r50)
                if(True in young):
                    sfr_r50 = np.sum(memmass[young]) / 1e8
        result_table['sfr'][i] = sfr
        result_table['sfr_r50'][i] = sfr_r50
        result_table['sfr_r90'][i] = sfr_r90
    # SFR10
    if('sfr10' in result_table.dtype.names):
        sfr10=0; sfr10_r50=0; sfr10_r90=0
        young = members['age'] < 0.01
        if(True in young):
            sfr10 = np.sum(memmass[young]) / 1e7 # Msol/yr
            young = young & (dist < r90)
            if(True in young):
                sfr10_r90 = np.sum(memmass[young]) / 1e7
                young = young & (dist < r50)
                if(True in young):
                    sfr10_r50 = np.sum(memmass[young]) / 1e7
        result_table['sfr10'][i] = sfr10
        result_table['sfr10_r50'][i] = sfr10_r50
        result_table['sfr10_r90'][i] = sfr10_r90
    # Magnitude
    if('zmag' in result_table.dtype.names):
        for band in global_bands:
            mags = members[f"{band}mag"]
            mags = mags[~np.isnan(mags)]
            result_table[f'{band}mag'][i] = -2.5*np.log10(np.sum(10**(-0.4*mags)))
    # Age
    if('age' in result_table.dtype.names):
        result_table['age'][i] = np.average(members['age'], weights=memmass)
        for band in global_bands:
            result_table[f'age{band}'][i] = np.average(members['age'], weights=members[f'{band}lum'])
    # Metallicity
    if('metal' in result_table.dtype.names):
        result_table['metal'][i] = np.average(members['metal'], weights=memmass)
    # V/sigma
    if('vsig_r90' in result_table.dtype.names):
        vsig=0; vsig_r50=0; vsig_r90=0
        #----- Halo property
        cx = halo['x']; cy = halo['y']; cz = halo['z']
        cvx = halo['vx']; cvy = halo['vy']; cvz = halo['vz']
        Lx = halo['Lx']; Ly = halo['Ly']; Lz = halo['Lz']
        L = np.sqrt(Lx**2 + Ly**2 + Lz**2)
        Lx /= L; Ly /= L; Lz /= L
        Lvec = np.array([Lx, Ly, Lz])
        vmean = np.sqrt(cvx**2 + cvy**2 + cvz**2)
        #----- Star property
        x = members['x'] - cx
        y = members['y'] - cy
        z = members['z'] - cz
        r = np.sqrt(x**2 + y**2 + z**2)
        argzero = r==0
        vx = members['vx']/sunits['km/s'] - cvx
        vy = members['vy']/sunits['km/s'] - cvy
        vz = members['vz']/sunits['km/s'] - cvz
        #----- Geometry
        with np.errstate(divide='ignore', invalid='ignore'):
            runit = np.vstack([x/r,y/r,z/r]).T
        runit[argzero] = 0
        rotunit = np.cross(Lvec, runit)
        vrot = np.sum(np.vstack([vx,vy,vz]).T * rotunit, axis=1)
        vnorm = np.sqrt(vx**2 + vy**2 + vz**2)
        sigma = np.sqrt( np.average((vnorm - vmean)**2, weights=memmass) )
        vsig = np.average(vrot, weights=memmass) / sigma
        result_table['vsig'][i] = vsig

        mask = r < r90
        if mask.any():
            sigma = np.sqrt( np.average((vnorm[mask] - vmean)**2, weights=memmass[mask]) )
            vsig_r90 = np.average(vrot[mask], weights=memmass[mask]) / sigma
        else:
            vsig_r90 = np.nan
        result_table['vsig_r90'][i] = vsig_r90

        mask = r < r50
        if mask.any():
            sigma = np.sqrt( np.average((vnorm[mask] - vmean)**2, weights=memmass[mask]) )
            vsig_r50 = np.average(vrot[mask], weights=memmass[mask]) / sigma
        else:
            vsig_r50 = np.nan
        result_table['vsig_r50'][i] = vsig_r50
    # SMBH
    if('dBH' in result_table.dtype.names):
        sink_dist = np.sqrt((send['x']-halo['x'])**2 + (send['y']-halo['y'])**2 + (send['z']-halo['z'])**2)
        inside = sink_dist < halo['r']
        smbhs = send[inside]
        sink_dist = sink_dist[inside]
        if(len(smbhs)==0): # No SMBH
            result_table['MBH'][i] = 0
            result_table['dBH'][i] = 0
        elif(len(smbhs)==1): # One SMBH
            result_table['MBH'][i] = smbhs['m'][0]/sunits['Msol']
            result_table['dBH'][i] = sink_dist[0]
        else:
            # Multiple SMBHs
            argclose = np.argmin(sink_dist)
            if(smbhs[argclose]['m'] == np.max(smbhs['m'])):
                result_table['MBH'][i] = smbhs[argclose]['m']/sunits['Msol']
                result_table['dBH'][i] = sink_dist[argclose]
            else:
                # calc accerelation
                accs = smbhs['m'] / sink_dist**2
                argmax = np.argmax(accs)
                result_table['MBH'][i] = smbhs[argmax]['m']/sunits['Msol']
                result_table['dBH'][i] = sink_dist[argmax]
    # Surface Brightness
    if('SBz' in result_table.dtype.names):
        # Total
        mockdist = 10 # pc, assume
        area = np.pi*(3600*180*halo['r']/sunits['pc']/mockdist/np.pi)**2 # arcsec^2
        for band in global_bands:
            appmag = members[f'{band}mag'] # + 5np.log10(mockdist) - 5
            SB = np.sum(10**(appmag / -2.5))/area
            SB = -2.5*np.log10(SB) if(SB>0) else asinh_mag(SB, band)
            result_table[f'SB{band}'][i] = SB
        # R90
        area = np.pi*(3600*180*r90/sunits['pc']/mockdist/np.pi)**2 # arcsec^2
        inner = members[dist < r90]
        if(len(inner)>0):
            for band in global_bands:
                appmag = inner[f'{band}mag']
                SB = np.sum(10**(appmag / -2.5))/area
                SB = -2.5*np.log10(SB) if(SB>0) else asinh_mag(SB, band)
                result_table[f'SB{band}_r90'][i] = SB
        else:
            for band in global_bands:
                result_table[f'SB{band}_r90'][i] = np.nan
        # R50
        area = np.pi*(3600*180*r50/sunits['pc']/mockdist/np.pi)**2 # arcsec^2
        inner = members[dist < r50]
        if(len(inner)>0):
            for band in global_bands:
                appmag = inner[f'{band}mag']
                SB = np.sum(10**(appmag / -2.5))/area
                SB = -2.5*np.log10(SB) if(SB>0) else asinh_mag(SB, band)
                result_table[f'SB{band}_r50'][i] = SB
        else:
            for band in global_bands:
                result_table[f'SB{band}_r50'][i] = np.nan
    # Hydro
    cellmass = None
    clists = ['H', 'O', 'Fe', 'Mg', 'C', 'N', 'Si', 'S', 'D', 'd1','d2','d3','d4']
    check = [clist in result_table.dtype.names for clist in clists if clist[0]!='d']
    if('metal_gas' in result_table.dtype.names)or(True in check):
        if(debug)and(i==0): print(" [CalcFunc] > Hydro")
        # halo prop
        cx = halo['x']; cy = halo['y']; cz = halo['z']
        cvx = halo['vx']; cvy = halo['vy']; cvz = halo['vz']
        vmean = np.sqrt(cvx**2 + cvy**2 + cvz**2)
        Lx = halo['Lx']; Ly = halo['Ly']; Lz = halo['Lz']
        L = np.sqrt(Lx**2 + Ly**2 + Lz**2)
        Lx /= L; Ly /= L; Lz /= L
        Lvec = np.array([Lx, Ly, Lz])

        # Load cells
        if(debug)and(i==0): print(" [CalcFunc] > Load cell")
        cshape, caddress, cdtype, cpulist_cell, bound_cell = cell_memory
        cexist = shared_memory.SharedMemory(name=caddress)
        cells = np.ndarray(cshape, dtype=cdtype, buffer=cexist.buf)
        cells = uri.domain_slice(cells, cdomain, cpulist_cell, bound_cell)
        cdist = np.sqrt( (cells['x']-cx)**2 + (cells['y']-cy)**2 + (cells['z']-cz)**2 )
        rmask = cdist < halo['r']
        if(np.sum(rmask) < 8):
            rmask = cdist < (halo['r'] + (1 / 2**cells['level'])/2)

        cells = cells[rmask]; cdist = cdist[rmask]
        dx = 1 / 2**cells['level']
        vol = dx**3
    if('metal_gas' in result_table.dtype.names):
        # Cell mass
        if(debug)and(i==0): print(" [CalcFunc] > cell mass")
        cellmass = cells['rho']*dx**3 / sunits['Msol']
        result_table['mgas'][i] = np.sum(cellmass)
        result_table['mgas_r50'][i] = np.sum(cellmass[cdist < r50])
        result_table['mgas_r90'][i] = np.sum(cellmass[cdist < r90])

        # temperature
        if(debug)and(i==0): print(" [CalcFunc] > temperature")
        T = cells['P']/cells['rho'] / sunits['K']
        cold = T < 1e4
        dense = (cells['rho'] / sunits['H/cc'] > 5) & (cold)
        result_table['mcold'][i] = np.sum(cellmass[cold])
        result_table['mcold_r50'][i] = np.sum(cellmass[cold & (cdist < r50)])
        result_table['mcold_r90'][i] = np.sum(cellmass[cold & (cdist < r90)])
        result_table['mdense'][i] = np.sum(cellmass[dense])
        result_table['mdense_r50'][i] = np.sum(cellmass[dense & (cdist < r50)])
        result_table['mdense_r90'][i] = np.sum(cellmass[dense & (cdist < r90)])


        # vsig
        if(debug)and(i==0): print(" [CalcFunc] > vsig")
        x = cells['x'] - cx
        y = cells['y'] - cy
        z = cells['z'] - cz
        r = np.sqrt(x**2 + y**2 + z**2)
        vx = cells['vx']/sunits['km/s'] - cvx
        vy = cells['vy']/sunits['km/s'] - cvy
        vz = cells['vz']/sunits['km/s'] - cvz

        runit = np.vstack([x/r,y/r,z/r]).T
        rotunit = np.cross(Lvec, runit)
        vrot = np.sum(np.vstack([vx,vy,vz]).T * rotunit, axis=1)
        vnorm = np.sqrt(vx**2 + vy**2 + vz**2)
        if(len(cellmass)>0):
            sigma = np.sqrt( np.average((vnorm - vmean)**2, weights=cellmass) )
            vsig = np.average(vrot, weights=cellmass) / sigma
        else:
            vsig = np.nan
        result_table['vsig_gas'][i] = vsig
        mask = r < r90
        if(mask.any()):
            sigma = np.sqrt( np.average((vnorm[mask] - vmean)**2, weights=cellmass[mask]) )
            vsig_r90 = np.average(vrot[mask], weights=cellmass[mask]) / sigma
        else:
            vsig_r90 = np.nan
        result_table['vsig_gas_r90'][i] = vsig_r90
        mask = r < r50
        if(mask.any()):
            sigma = np.sqrt( np.average((vnorm[mask] - vmean)**2, weights=cellmass[mask]) )
            vsig_r50 = np.average(vrot[mask], weights=cellmass[mask]) / sigma
        else:
            vsig_r50 = np.nan
        result_table['vsig_gas_r50'][i] = vsig_r50
        
        # metal
        if(debug)and(i==0): print(" [CalcFunc] > metal")
        metal = np.average(cells['metal'], weights=cellmass) if(len(cellmass)>0) else np.nan
        result_table['metal_gas'][i] = metal

    # Chemical
    if(True in check):
        if cellmass is None: cellmass = cells['rho']*dx**3 / sunits['Msol']
        for clist in clists:
            if(clist in result_table.dtype.names):
                if(debug)and(i==0): print(f" [CalcFunc] > {clist}")
                value = np.average(cells[clist], weights=cellmass) if(len(cellmass)>0) else np.nan
                result_table[clist][i] = value



def dump_func(result_table, full_path, iout, names, verbose):
    for key, val in names.items():
        title = val[0]
        desc = val[1]
        datdump((result_table[key], desc), f"{full_path}/{title}_{iout:05d}.dat", msg=verbose)