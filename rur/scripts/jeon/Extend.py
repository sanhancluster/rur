import numpy as np
from rur import uri, uhmi
from rur.utool import load
import os, glob
# import read_bricks
from multiprocessing import Pool, shared_memory
from tqdm import tqdm
import argparse
import time
import datetime
import signal

print("ex: python3 Extend.py --mode nc --title R50 --verbose")
parser = argparse.ArgumentParser(description='Extend HaloMaker (syj3514@yonsei.ac.kr)')
parser.add_argument("-m", "--mode", default='nc', required=False, help='Simulation mode', type=str)
parser.add_argument("-n", "--nthread", default=8, required=False, help='Ncore', type=int)
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--low", action='store_true')
args = parser.parse_args()
print(args)
mode = args.mode
nthread = args.nthread
verbose = args.verbose
low = args.low
uri.timer.verbose = 1 if verbose else 0

def check_chems(result_dtype, hvars):
    checklist = ['H', 'O', 'Fe', 'Mg', 'C', 'N',
    'Si', 'S', 'D', 'd1','d2','d3','d4']

    dtype = np.dtype(result_dtype)
    ndtype = [descr for descr in dtype.descr if(descr[0] not in checklist)or(descr[0] in hvars)]
    return ndtype

def getmem(members, cparts, i):
    if(members is None):
        return None
    else:
        return members.table[cparts[i]:cparts[i+1]]

# --------------------------------------------------------------
# Main Function
# --------------------------------------------------------------
def calc_extended(
    path, galaxy, iout, name_dicts, pre_func, calc_func, dump_func,
    nthread=8, verbose=False,
    need_member=False, mtarget_fields=None,
    need_part=False, ptarget_fields=None,
    need_cell=False, ctarget_fields=None, 
    get_additional=None, func_additional=None,
    **kwargs):
    global mode

    # Setting database
    path_in_repo = 'galaxy' if galaxy else 'halo'
    names = list(name_dicts.keys())
    prefix = "GAL" if galaxy else "HAL"
    full_path = f"{path}/{path_in_repo}/extended/{iout:05d}"
    if(not os.path.exists(full_path)):
        os.makedirs(full_path)
    snap = uri.RamsesSnapshot(path, iout)

    # Check names
    hnames = snap.hydro_names
    for name in names:
        if(name in ['H','O','Fe','Mg','C','Si','N','S','D','d1','d3','d2','d4']):
            if(name not in hnames)or(low):
                del name_dicts[name]
    if(len(name_dicts)==0):
        print(f"Skip {iout}")
        return
    names = list(name_dicts.keys())
    result_dtype = [(name, 'f8') for name in names]
    if(verbose): print(f"\nExtended: {names} of {path_in_repo}\n")
    if('dBH' in names):
        def _f1(table, snapm, members, snap, part_memory, cell_memory):
            snap.read_sink()
            return snap.sink_data
        get_additional = _f1
        def _f2(dat):
            ndtype = np.dtype([('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('m', 'f8')])
            ndat = np.empty(len(dat), dtype=ndtype)
            for name in ndtype.names:
                ndat[name] = dat[name]
            return ndat
        func_additional = _f2
    need_members = {
        'r50z':['x', 'y', 'z', 'm', 'epoch', 'metal'],
        'r50':['x', 'y', 'z', 'm'],
        'SBz':['x','y','z','epoch','metal'],
        'metal':['m','metal'],
        'vsig':['x', 'y', 'z', 'vx','vy','vz', 'm'],
        'sfr':['x', 'y', 'z', 'm', 'epoch'],
        'sfr10':['x', 'y', 'z', 'm', 'epoch'],
        'age':['m', 'epoch','metal'],
        'zmag':['m', 'epoch', 'metal'],
    }
    fields = []
    for name in names:
        if(name in need_members):
            fields += need_members[name]
    if(len(fields)>0):
        fields = list(set(fields))
        mtarget_fields = fields
        need_member = True
        if(verbose): print(f" > Member fields: {fields}")
    need_cell = 'metal_gas' in names
    if(need_cell)&(verbose): print(f" > Need to load cell data")
    if(need_cell)&(low): ctarget_fields = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'rho','P','level','metal','cpu']


    
    # Load HaloMaker
    sparams = snap.params; sunits = snap.unit
    table = uhmi.HaloMaker.load(snap,galaxy=galaxy, extend=False)
    if(verbose): print(f" > Calculate for {len(table)} {path_in_repo}s")

    # Load Member particles
    snapm = None
    members=None; cparts=None
    part_memory = (None, None, None)
    cell_memory = (None, None, None)
    if(need_member):
        if(verbose): print(f" > Read member particles")
        snapm = uri.RamsesSnapshot(path, iout)
        members = uhmi.HaloMaker.read_member_parts(snapm, table, galaxy=galaxy, nthread=nthread, copy=True, target_fields=mtarget_fields)
        nparts = table['nparts']
        cparts = np.cumsum(nparts); cparts = np.insert(cparts, 0, 0)

    # Load Raw data
    if(need_part)or(need_cell):
        domain = f"{path}/{path_in_repo}/{prefix}_{iout:05d}/domain_{iout:05d}.pkl"
        if(os.path.exists(domain)):
            if(verbose): print(f" > Load domain")
            domain = load(domain, msg=verbose)
            cpulist = np.unique(np.concatenate(domain))
        else:
            if(verbose): print(f" > Get halos cpu list")
            cpulist = snap.get_halos_cpulist(table, nthread=nthread)
        if(need_part):
            if(verbose): print(f" > Get part")
            pname = 'star' if galaxy else 'dm'
            snap.get_part(pname, nthread=nthread, target_fields=ptarget_fields, cpulist=cpulist)
            pshape = snap.part.shape; paddress = snap.part_mem.name; pdtype = snap.part.dtype
            part_memory = (pshape, paddress, pdtype)        
        if(need_cell):
            if(verbose): print(f" > Get cell")
            snap.get_cell(target_fields=ctarget_fields, nthread=nthread, cpulist=cpulist)
            cshape = snap.cell.shape; caddress = snap.cell_mem.name; cdtype = snap.cell.dtype
            cell_memory = (cshape, caddress, cdtype)
            result_dtype = check_chems(result_dtype, snap.hydro_names)
    
    # Preprocess
    if(pre_func is not None):
        if(verbose): print(f" > Preprocess datasets")
        table, snapm, members, snap, part_memory, cell_memory = pre_func(names, table, snapm, members, snap, part_memory, cell_memory, full_path, verbose)

    # Additional data
    send = None
    if(func_additional is not None):
        if(verbose): print(f" > Get additional data")
        dat = get_additional(table, snapm, members, snap, part_memory, cell_memory)
        send = func_additional(dat)

    # Assign shared memory
    if(verbose): print(f" > Make shared memory")
    shmname = f"extend_{mode}_{path_in_repo}_{snap.iout:05d}"
    if(os.path.exists(f"/dev/shm/{shmname}")):
        os.remove(f"/dev/shm/{shmname}")
    result_table = np.empty(len(table), dtype=result_dtype)
    memory = shared_memory.SharedMemory(name=shmname, create=True, size=result_table.nbytes)
    result_table = np.ndarray(result_table.shape, dtype=result_dtype, buffer=memory.buf)
    shape = result_table.shape; address = memory.name; dtype = result_dtype

    # Main Calculation
    if(verbose): print(f" > Start Calculation")
    if(verbose):
        pbar = tqdm(total=len(table), desc=f"Nthread={min(len(table), nthread)}")
        def update(*a):
            pbar.update()
    else:
        update = None
    if(snap is not None):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    with Pool(processes=min(len(table),nthread)) as pool:
        async_result = [pool.apply_async(calc_func, args=(i, table[i], shape, address, dtype, sparams, sunits, getmem(members, cparts, i), part_memory, cell_memory, send), callback=update) for i in range(len(table))]
        # iterobj = tqdm(async_result, total=len(async_result), desc=f"Nthread={nthread}") if(verbose) else async_result
        iterobj = async_result
        for result in iterobj:
            result.get()
    if(snap is not None):
        signal.signal(signal.SIGTERM, snap.terminate)
    
    # Dump and relase memory
    if(verbose): print(f" > Dumping")
    dump_func(result_table, full_path, iout, name_dicts, verbose)
    memory.close()
    memory.unlink()
    if(snap is not None):
        snap.clear()
    if(snapm is not None):
        snapm.clear()
    if(verbose): print(f" Done\n")
    # raise ValueError("Done")
# --------------------------------------------------------------


# --------------------------------------------------------------
# Execution
# --------------------------------------------------------------
if __name__ == "__main__":
    # Import calculation functions
    from extend import default_names, pre_func, calc_func, dump_func, skip_func, inhouse

    # Set path
    path = inhouse[mode]
    galaxy = True
    path_in_repo = 'galaxy' if galaxy else 'halo'

    # Load nout
    full_path = f"{path}/{path_in_repo}/"
    bricks = glob.glob(f"{full_path}/tree_bricks*")
    nout = [int(ib[-5:]) for ib in bricks]
    nout.sort()
    
    # Run!
    iterator = nout# if verbose else tqdm(nout)
    for iout in iterator:
        names = skip_func(path, galaxy, iout, default_names, verbose)
        skip = len(names)==0
        if(skip):
            print(f"\n=================\nSkip {iout}\n=================")
            continue
        now = datetime.datetime.now() 
        now = f"V{now}V" if verbose else now
        print(f"\n=================\nStart {iout} {now}\n================="); ref = time.time()
        calc_extended(
            path, galaxy, iout, names, 
            pre_func, calc_func, dump_func, 
            verbose=verbose, nthread=nthread,)
        print(f"Done ({time.time()-ref:.2f} sec)")
