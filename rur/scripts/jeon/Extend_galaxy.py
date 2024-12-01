print("ex: python3 Extend.py --mode nc --nthread 24 --verbose --chem")

import numpy as np
from rur import uri, uhmi
from rur.utool import load
import os, glob, sys
from multiprocessing import Pool, shared_memory
from tqdm import tqdm
import argparse, time, datetime, signal

"""
Extend list:
(galaxy)
['age','ageg','agei','ager','ageu','agez','umag','gmag','rmag','imag','zmag','metal','r50','r50g','r50i','r50r','r50u','r50z','r90','r90g','r90i','r90r','r90u','r90z','SBg','SBg_r50','SBg_r90','SBi','SBi_r50','SBi_r90','SBr','SBr_r50','SBr_r90','SBu','SBu_r50','SBu_r90','SBz','SBz_r50','SBz_r90','SFR','SFR10','SFR10_r50','SFR10_r90','SFR_r50','SFR_r90','vsig','vsig_r50','vsig_r90','MBH','dBH','vsig_gas','vsig_gas_r50','vsig_gas_r90','Mcold_gas','Mcold_gas_r50','Mcold_gas_r90','Mdense_gas','Mdense_gas_r50','Mdense_gas_r90','metal_gas','M_gas','M_gas_r50','M_gas_r90','H_gas','O_gas','Fe_gas','Mg_gas','C_gas','N_gas','Si_gas','S_gas','D_gas','CDustLarge_gas','CDustSmall_gas','SiDustLarge_gas','SiDustSmall_gas']
"""


def delprint(n=1):
    """Delete the last line in the STDOUT."""
    for _ in range(n):
        sys.stdout.write("\x1b[1A")  # cursor up one line
        sys.stdout.write("\x1b[2K")  # delete the last line

parser = argparse.ArgumentParser(description='Extend HaloMaker (syj3514@yonsei.ac.kr)')
parser.add_argument("-m", "--mode", default='nc', required=False, help='Simulation mode', type=str)
parser.add_argument("-n", "--nthread", default=8, required=False, help='Ncore', type=int)
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--nocell", action='store_true')
parser.add_argument("--chem", action='store_true')
args = parser.parse_args()
print(args)
mode = args.mode
nthread = args.nthread
verbose = args.verbose
nocell = args.nocell
chem = args.chem
galaxy = True
if(nocell): chem = False
uri.timer.verbose = 1 if verbose else 0

def check_chems(result_dtype, hvars):
    global chem
    checklist = ['H', 'O', 'Fe', 'Mg', 'C', 'N',
    'Si', 'S', 'D', 'd1','d2','d3','d4']

    dtype = np.dtype(result_dtype)
    if(chem):
        ndtype = [descr for descr in dtype.descr if(descr[0] not in checklist)or(descr[0] in hvars)]
    else:
        ndtype = [descr for descr in dtype.descr if(descr[0] not in checklist)]
    return ndtype

def getmem(members, cparts, i):
    if(members is None):
        return None
    else:
        return members.table[cparts[i]:cparts[i+1]]

def domsave(fname, domain):
    if os.path.exists(fname): return None
    domain_16 = [dom.astype(np.int16) for dom in domain]
    bdomain = [dom.tobytes() for dom in domain_16]
    with open(fname, "wb") as f:
        f.write(len(bdomain).to_bytes(4, byteorder='little'))
        for i in range(len(bdomain)):
            f.write(bdomain[i])
            f.write("\n".encode())
    assert os.path.exists(fname)

def domload(path, msg=False):
    with open(path, "rb") as f:
        leng = int.from_bytes(f.read(4), byteorder='little')
        domain = [None]*leng
        oldv = None
        cursor = 0
        for i in range(leng):
            v=f.readline()
            if(len(v)%2 == 0):
                v = oldv + v
                cursor -= 1
            domain[cursor] = np.frombuffer(v[:-1], dtype='i2')
            oldv = v
            cursor += 1

        while cursor < leng:
            v=f.readline()
            if(len(v)%2 == 0):
                v = oldv + v
                cursor -= 1
            domain[cursor] = np.frombuffer(v[:-1], dtype='i2')
            cursor += 1
            
    if(msg): print(f" `{path}` loaded")
    return domain

# --------------------------------------------------------------
# Main Function
# --------------------------------------------------------------
def calc_extended(
    path, iout, name_dicts, pre_func, calc_func, dump_func,
    nthread=8, verbose=False,
    need_member=False, mtarget_fields=None,
    need_part=False, ptarget_fields=None,
    need_cell=False, ctarget_fields=None, 
    get_additional=None, func_additional=None,
    **kwargs):
    global mode
    walltimes = []
    ref = time.time()

    # Setting database
    path_in_repo = 'galaxy'
    names = list(name_dicts.keys())
    prefix = "GAL"
    full_path = f"{path}/{path_in_repo}/extended/{iout:05d}"
    if(not os.path.exists(full_path)): os.makedirs(full_path)
    uri.timer.verbose=0
    snap = uri.RamsesSnapshot(path, iout); snap.shmprefix = "extendgalaxy"
    uri.timer.verbose = 1 if verbose else 0

    # Check names
    hnames = snap.hydro_names
    for name in names:
        delete = False
        if(name in ['H','O','Fe','Mg','C','Si','N','S','D','d1','d3','d2','d4']):
            if(name not in hnames)or(nocell)or(not chem):
                delete = True
        if('gas' in name_dicts[name][0])&(nocell):
            delete = True
        if(delete): del name_dicts[name]
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
            for name in ndtype.names: ndat[name] = dat[name]
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
        'zmag':['m', 'epoch', 'metal']}
    ftmp = []
    fields = ['x', 'y', 'z', 'vx','vy','vz', 'm', 'epoch', 'metal']
    for name in names:
        if(name in need_members): ftmp += need_members[name]
    if(len(ftmp)>0):
        mtarget_fields = [field for field in fields if field in list(set(ftmp))]
        need_member = True
        if(verbose): print(f" > Member fields: {fields}")
    
    need_cells = {
        'mgas_r90':['x', 'y', 'z', 'rho', 'level'],
        'mcold_r90':['x', 'y', 'z', 'rho', 'P', 'level'],
        'mdense_r90':['x', 'y', 'z', 'rho', 'P', 'level'],
        'vsig_gas_r90':['x', 'y', 'z', 'vx','vy','vz', 'rho', 'level'],
        'metal_gas':['x','y','z','rho','level','metal'],
        'H':['x','y','z','rho','level','H','O','Fe','Mg','C','Si','N','S','D'] if chem else ['x','y','z','rho','level'],
        'd1':['x','y','z','rho','level','d1','d2','d3','d4'] if chem else ['x','y','z','rho','level']}
    ctmp = []; ctarget_fields = [
        'x', 'y', 'z', 'vx','vy','vz', 'rho','P','level','metal',
        'H', 'O', 'Fe', 'Mg', 'C', 'Si', 'N', 'S', 'D', 'd1', 'd2', 'd3', 'd4']
    for name in names:
        if(name in need_cells): ctmp += need_cells[name]
    if(len(ctmp)>0):
        ctarget_fields = [field for field in ctarget_fields if field in list(set(ctmp))]
        need_cell = True
        if(verbose): print(f" > Cell fields: {ctarget_fields}")
       
    # need_cell = len([value[0] for value in name_dicts.values() if('gas' in value[0])]) >0
    if(nocell): need_cell = False
    if(need_cell)&(verbose): print(f" > Need to load cell data")
    # if(need_cell)&(not chem): ctarget_fields = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'rho','P','level','metal','cpu']


    
    # Load HaloMaker
    sparams = snap.params; sunits = snap.unit
    table = uhmi.HaloMaker.load(snap,galaxy=True, extend=False)
    if(verbose): print(f" > Calculate for {len(table)} {path_in_repo}s")
    domain = [None for _ in range(len(table))]
    walltime = ("Preparation", time.time()-ref); walltimes.append(walltime); ref = time.time()

    # Load Member particles
    snapm = None
    members=None; cparts=None
    part_memory = (None, None, None, None, None)
    cell_memory = (None, None, None, None, None)
    if(need_member):
        if(verbose): print(f" > Read member particles ({np.sum(table['nparts'])})")
        snapm = uri.RamsesSnapshot(path, iout); snapm.shmprefix = "extendgalaxy"
        members = uhmi.HaloMaker.read_member_parts(snapm, table, galaxy=galaxy, nthread=nthread, copy=True, target_fields=mtarget_fields)
        nparts = table['nparts']
        cparts = np.cumsum(nparts); cparts = np.insert(cparts, 0, 0)
        delprint(2)
    walltime = ("Read member", time.time()-ref); walltimes.append(walltime); ref = time.time()

    # Load Raw data
    if(need_part)or(need_cell):
        fdomain = f"{path}/{path_in_repo}/{prefix}_{iout:05d}/domain_{iout:05d}.dat"
        if(os.path.exists(fdomain)):
            if(verbose): print(f" > Load domain")
            domain = domload(fdomain, msg=verbose)
            cpulist = np.unique(np.concatenate(domain))
        else:
            if(verbose): print(f" > Get halos cpu list")
            cpulist, domain = snap.get_halos_cpulist(table, nthread=nthread, full=True)
            domsave(fdomain, domain)
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
            cpulist_cell = snap.cpulist_cell; bound_cell = snap.bound_cell
            cell_memory = (cshape, caddress, cdtype, cpulist_cell, bound_cell)
            result_dtype = check_chems(result_dtype, snap.hydro_names)
        walltime = ("Read raw", time.time()-ref); walltimes.append(walltime); ref = time.time()
    
    # Preprocess
    if(pre_func is not None):
        if(verbose): print(f" > Preprocess datasets")
        table, snapm, members, snap, part_memory, cell_memory = pre_func(names, table, snapm, members, snap, part_memory, cell_memory, full_path, nthread, verbose)
        walltime = ("Preprocess", time.time()-ref); walltimes.append(walltime); ref = time.time()

    # Additional data
    send = None
    if(func_additional is not None):
        if(verbose): print(f" > Get additional data")
        dat = get_additional(table, snapm, members, snap, part_memory, cell_memory)
        delprint(2)
        send = func_additional(dat)
        walltime = ("Additional data", time.time()-ref); walltimes.append(walltime); ref = time.time()

    # Assign shared memory
    if(verbose): print(f" > Make shared memory")
    shmname = f"extendgalaxy_{mode}_{path_in_repo}_{snap.iout:05d}"
    if(os.path.exists(f"/dev/shm/{shmname}")): os.remove(f"/dev/shm/{shmname}")
    result_table = np.empty(len(table), dtype=result_dtype)
    memory = shared_memory.SharedMemory(name=shmname, create=True, size=result_table.nbytes)
    result_table = np.ndarray(result_table.shape, dtype=result_dtype, buffer=memory.buf)
    shape = result_table.shape; address = memory.name; dtype = result_dtype

    # Main Calculation
    if(verbose): print(f" > Start Calculation")
    if(verbose):
        reft = time.time()
        pbar = tqdm(total=len(table), desc=f"Nthread={min(len(table), nthread)}")
        def update(*a): pbar.update()
    else: update = None
    if(snap is not None): signal.signal(signal.SIGTERM, signal.SIG_DFL)
    with Pool(processes=min(len(table),nthread)) as pool:
        async_result = [pool.apply_async(calc_func, args=(i, table[i], shape, address, dtype, sparams, sunits, getmem(members, cparts, i), part_memory, cell_memory, domain[i], send), callback=update) for i in range(len(table))]
        iterobj = async_result
        for result in iterobj: result.get()
    if(snap is not None): signal.signal(signal.SIGTERM, snap.terminate)
    if(verbose):
        pbar.close(); delprint(1)
    walltime = ("Get results", time.time()-ref); walltimes.append(walltime); ref = time.time()

    # Dump and relase memory
    if(verbose): print(f" > Dumping")
    dump_func(result_table, full_path, iout, name_dicts, verbose)
    memory.close(); memory.unlink()
    if(snap is not None): snap.clear()
    if(snapm is not None): snapm.clear()
    if(verbose): print(f" Done\n")
    walltime = ("Dump", time.time()-ref); walltimes.append(walltime); ref = time.time()
    if(verbose):
        for name, walltime in walltimes: print(f" > {name}: {walltime:.2f} sec")
# --------------------------------------------------------------


# --------------------------------------------------------------
# Execution
# --------------------------------------------------------------
if __name__ == "__main__":
    # Import calculation functions
    from extend_galaxy import default_names, pre_func, calc_func, dump_func, skip_func, inhouse

    # Set path
    path = inhouse[mode]
    path_in_repo = 'galaxy'

    # Load nout
    full_path = f"{path}/{path_in_repo}/"
    bricks = glob.glob(f"{full_path}/tree_bricks*")
    nout = [int(ib[-5:]) for ib in bricks]
    nout.sort()
    
    # Run!
    iterator = nout[::-1]# if verbose else tqdm(nout)
    for iout in iterator:
        # if(iout>411): continue
        names = skip_func(path, iout, default_names, verbose)
        skip = len(names)==0
        if(skip):
            print(f"\n=================\nSkip {iout}\n=================")
            continue
        now = datetime.datetime.now() 
        now = f"V{now}V" if verbose else now
        print(f"\n=================\nStart {iout} {now}\n================="); ref = time.time()
        calc_extended(
            path, iout, names, 
            pre_func, calc_func, dump_func, 
            verbose=verbose, nthread=nthread,)
        print(f"Done ({time.time()-ref:.2f} sec)")
