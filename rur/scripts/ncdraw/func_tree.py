import numpy as np
from time import gmtime, strftime
from multiprocessing import Pool, shared_memory
from rur.utool import dump, load, datload
import os






def _get_val(snap, gtree, htree, extended, key):
    if key=='SFMS':
        xval = gtree['m']
        yval = extended['SFR']
    elif key=='SHMR':
        xval = htree['mvir']
        yval = gtree['m']/htree['mvir']
    elif key=='CMD':
        xval = extended['rmag']
        yval = extended['gmag']-extended['rmag']
    elif key=='SB':
        xval = extended['rmag']
        yval = extended['SBr_r50']
    elif key=='Size':
        xval = gtree['m']
        yval = extended['r50'] / snap.unit['ckpc'] * gtree['aexp']
    elif key=='Metal':
        xval = gtree['m']
        yval = extended['metal']
    elif key=='MBH':
        xval = gtree['m']
        yval = extended['MBH']
    elif key=='Cold':
        xval = gtree['m']
        yval = extended['Mcold_gas_r90']/xval
    elif key=='O/H':
        xval = gtree['m']
        yval = 12 + np.log10(extended['O_gas'] / 16 / extended['H_gas'])
    elif key=='alpha':
        from rur.sci.chemistry import solar_frac
        FeH_solar = solar_frac['Fe'] / solar_frac['H']
        OFe_solar = solar_frac['O'] / solar_frac['Fe']
        MgFe_solar = solar_frac['Mg'] / solar_frac['Fe']
        SiFe_solar = solar_frac['Si'] / solar_frac['Fe']
        aFe_solar = (OFe_solar + MgFe_solar + SiFe_solar) / 3
        xval = np.log10(extended['Fe_gas'] / extended['H_gas'] / FeH_solar)
        alpha = (extended['O_gas'] + extended['Mg_gas'] + extended['Si_gas'])/3
        yval = np.log10(alpha / extended['Fe_gas'] / aFe_solar)
    elif key=='DTM':
        xval = gtree['m']
        yval = (extended['CDustLarge_gas'] + extended['CDustSmall_gas'] + extended['SiDustLarge_gas']/0.163 + extended['SiDustSmall_gas']/0.163)/extended['metal_gas']


    
    return xval, yval


def _extend(ith, ig, name, shape, dtype, gextend_path, keys, types):
    existing_memory, yvals = _get_shm(name, shape, dtype)
    iout = ig['timestep']
    iscombined = os.path.exists(f"{gextend_path}/{iout:05d}/chem/{ig['hmid']:07d}.pkl")
    if iscombined:
        oldtyp = 'none'
        for key, typ in zip(keys, types):
            if typ!= oldtyp:
                oldtyp = typ
                fname = f"{gextend_path}/{iout:05d}/{typ}/{ig['hmid']:07d}.pkl"
                loaded = load(fname, msg=False)
            val = loaded[key]
            yvals[key][ith] = val
    else:
        for key, typ in zip(keys, types):
            vals, desc = datload(f"{gextend_path}/{iout:05d}/{key}_{iout:05d}.dat", msg=False)
            val = vals[ig['hmid']-1]
            yvals[key][ith] = val
def _set_shm(leng, dtype):
    yvals = np.zeros(leng, dtype=dtype)
    now = strftime("%Y%m%d_%H%M%S", gmtime())
    memory = shared_memory.SharedMemory(name=f"gsq_tree_{now}",create=True, size=yvals.nbytes)
    arr = np.ndarray(yvals.shape, dtype=yvals.dtype, buffer=memory.buf)
    return memory, arr
def _get_shm(name, shape, dtype):
    existing_memory = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=dtype, buffer=existing_memory.buf)
    return existing_memory, arr