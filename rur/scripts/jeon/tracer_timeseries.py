from rur import uri
import numpy as np
assert np.__version__ < '2.0.0', 'numpy version should be less than 2.0.0'
import os, sys, glob
from rur.utool import *

from tqdm import tqdm
from multiprocessing import shared_memory, Pool
import argparse, time


parser = argparse.ArgumentParser(description='Fast tracer timeseries')
parser.add_argument("-r", "--repo", default='/storage7/NewCluster', required=False, help='Simulation repository', type=str)
parser.add_argument("-n", "--nthread", default=1, required=False, help='nthread', type=int)
parser.add_argument("--verbose", action='store_true')
args = parser.parse_args()
print(args)

repo = args.repo
nthread = args.nthread
path = f"{repo}/TRACER"
verbose = args.verbose
uri.timer.verbose = 1 if verbose else 0




def _dump_tracer(
    tracer_shape, tracer_name, tracer_dtype, 
    ikey, ipath, ihashs, iout, verbose, istatus):
    if istatus[4] >= iout: return
    tracer_exist = shared_memory.SharedMemory(name=tracer_name)
    tracer = np.ndarray(tracer_shape, dtype=tracer_dtype, buffer=tracer_exist.buf)
    mask = ihashs == ikey
    itracer = tracer[mask]
    argsort = np.argsort(itracer['id'])
    itracer = itracer[argsort]

    names = ['x', 'y', 'z', 'cpu', 'family']
    dtypes = ['f8', 'f8', 'f8', 'i2', 'i1']
    arrs = {}
    for jth, name, dtype in zip(range(5), names, dtypes):
        # if status[ikey, jth] < iout:
        if istatus[jth] < iout:
            fname = f"{ipath}/tracer_{name}_{ikey:03d}.pkl"
            if os.path.exists(fname):
                arr = load(fname, msg=False)
                arr = np.column_stack((arr, itracer[name]))
            else:
                arr = itracer[name].reshape(-1,1)
            if arr.dtype != dtype:
                arr.astype(dtype)
            # dump(arr, fname, msg=False)
            arrs[name] = arr
            istatus[jth] = iout
            # dump(istatus, file_status, msg=False)
    for name, arr in arrs.items():
        dump(arr, f"{ipath}/tracer_{name}_{ikey:03d}.pkl", msg=False)
    dump(istatus, file_status, msg=False)





snap = uri.RamsesSnapshot(repo, 1)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
table = snaps.iout_avail['iout']

file_header = f"{path}/header.pkl"
if not os.path.exists(file_header):
    header = {
        'mode':'nc',
        'minid':555674170,
        'desc' : '(id-minid)%1000 = file_suffix \n (id-minid)//1000 = row_number_at_each_file',
        'nout' : np.array([], dtype='i2')
    }
    dump(header, file_header, msg=verbose)
header = load(file_header, msg=verbose)
minid = header['minid']

# names = ['x', 'y', 'z', 'cpu', 'family']
for iout in table:
    # Check if this iout is already processed
    if iout in header['nout']: continue
    prefix = int(iout//100 * 100)
    ipath = f"{path}/iout_{prefix:03d}"
    if not os.path.exists(ipath): os.makedirs(ipath)

    # Check bricks
    checkout = 10000
    for ikey in range(1000):
        file_status = f"{ipath}/status_{ikey:03d}.pkl"
        if os.path.exists(file_status):
            istatus = load(file_status, msg=False)
        else:
            istatus = np.zeros(5, dtype='i2') # x, y, z, cpu, family
            dump(istatus, file_status, msg=False)

        # # Check file broken
        # for iii in range(5):
        #     fb = f"{ipath}/tracer_{names[iii]}_{ikey:03d}.pkl"
        #     if not os.path.exists(fb):
        #         istatus[iii] = 0
        # dump(istatus, file_status, msg=False)
        
        checkout = min(checkout, np.min(istatus))
    if checkout >= iout: continue

    isnap = snaps.get_snap(iout)
    isnap.get_part(pname='tracer', target_fields=['x','y','z','id','family'], nthread=24)
    tracer = isnap.part
    tracer_shmname = isnap.part_mem.name if nthread>1 else None
    ihashs, irows = (tracer['id']-minid)%1000, (tracer['id']-minid)//1000

    
    if nthread==1:
        if verbose:
            pbar = tqdm(total=1000, desc=f"iout_{iout:03d}")
            def update(*a): pbar.update()
        else:
            update = None
        # for ikey in tqdm(range(1000), desc=f"iout_{iout:03d}"):
        for ikey in range(1000):
            # This already done -> skip
            file_status = f"{ipath}/status_{ikey:03d}.pkl"
            istatus = load(file_status, msg=False)
            if istatus[4] >= iout:
                if verbose: update()
                continue
            
            mask = ihashs == ikey
            itracer = tracer[mask]
            argsort = np.argsort(itracer['id'])
            itracer = itracer[argsort]

            names = ['x', 'y', 'z', 'cpu', 'family']
            dtypes = ['f8', 'f8', 'f8', 'i2', 'i1']
            arrs = {}
            for jth, name, dtype in zip(range(5), names, dtypes):
                if istatus[jth] < iout:
                    fname = f"{ipath}/tracer_{name}_{ikey:03d}.pkl"
                    if os.path.exists(fname):
                        arr = load(fname, msg=False)
                        arr = np.column_stack((arr, itracer[name]))
                    else:
                        arr = itracer[name].reshape(-1,1)
                    if arr.dtype != dtype:
                        arr.astype(dtype)
                    # dump(arr, fname, msg=False)
                    arrs[name] = arr
                    istatus[jth] = iout
                    # dump(istatus, file_status, msg=False)
            for name, arr in arrs.items():
                dump(arr, f"{ipath}/tracer_{name}_{ikey:03d}.pkl", msg=False)
            dump(istatus, file_status, msg=False)
            if verbose: update()
        pbar.close()
    else:
        raise NotImplementedError("MP is much slower than SP")
        ikeys = []
        status = [0]*1000
        for ikey in range(1000):
            istatus = load(f"{ipath}/status_{ikey:03d}.pkl", msg=False)
            if istatus[4] < iout:
                ikeys.append(ikey)
                status[ikey] = istatus
        if verbose:
            pbar = tqdm(total=len(ikeys), desc=f"iout_{iout:03d}")
            def update(*a): pbar.update()
        else:
            update = None

        with Pool(processes=nthread) as pool:
            async_result = [pool.apply_async(
                _dump_tracer, (
                    tracer.shape, tracer_shmname, tracer.dtype, 
                    ikey, ipath, ihashs, iout, verbose, status[ikey]),
                callback=update) for ikey in ikeys]
            for r in async_result: r.get()

    header['nout'] = np.sort(np.append(header['nout'], iout).astype('i2'))
    dump(header, file_header, msg=False)
    isnap.clear()

