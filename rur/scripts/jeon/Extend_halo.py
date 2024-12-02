print("ex: python3 Extend.py --mode nc --nthread 24 --verbose")

import numpy as np
from rur import uri, uhmi
from rur.utool import load
import os, glob, sys
from multiprocessing import Pool, shared_memory
from tqdm import tqdm
import argparse, time, datetime, signal

"""
Extend list:
(halo)
['mcontam', 'r200c','m200c','r500c','m500c', 
'mstar_r', 'mstar_rvir', 'mstar_r200', 'mgas_r', 'mgas_rvir', 'mgas_r200',
'mcold_r', 'mcold_rvir', 'mcold_r200', 'mdense_r', 'mdense_rvir', 'mdense_r200',
'vmaxcir, 'rmaxcir, 'cNFW', 'cNFWerr', 'inslope', 'inslopeerr']
"""


def delprint(n=1):
    """Delete the last line in the STDOUT."""
    for _ in range(n):
        sys.stdout.write("\x1b[1A")  # cursor up one line
        sys.stdout.write("\x1b[2K")  # delete the last line

parser = argparse.ArgumentParser(description='Extend HaloMaker (syj3514@yonsei.ac.kr)')
parser.add_argument("-m", "--mode", default='nc', required=False, help='Simulation mode', type=str)
parser.add_argument("-n", "--nthread", default=8, required=False, help='Ncore', type=int)
parser.add_argument("-s", "--sep", default=-1, required=False, help='Separation', type=int)
parser.add_argument("-p", "--partition", default=0, required=False, help='Divide halo domain (1=x, 2=xy, 3=xyz)', type=int)
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--onlymem", action='store_true')
args = parser.parse_args()
print(args)
# mode:
#   Currently ['nc','nh','nh2'] are available.
#   See `inhouse` dictionary in `extend_galaxy.py` for the path.
mode = args.mode
# nthread:
#   Number of cores for multiprocessing.
nthread = args.nthread
# sep:
#   If sep>=0, only the iout%4==sep will be calculated.
#   Recommend to use this option when you want to use multi tardis nodes.
sep = args.sep
# partition:
#   If partition>0, the galaxy domain will be divided.
#   The number of divided domains is 2^partition. (1=x, 2=xy, 3=xyz)
#   Recommend to use this option when you lack memory.
partition = args.partition
ZIP = partition>0
# verbose:
#   If True, the progress bar will be shown and the verbose mode will be activated.
verbose = args.verbose
# onlymem:
#   If True, only the member particles will be calculated.
onlymem = args.onlymem
galaxy = False
uri.timer.verbose = 1 if verbose else 0

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
    need_dm=False, dtarget_fields=None,
    need_star=False, starget_fields=None,
    need_cell=False, ctarget_fields=None, 
    get_additional=None, func_additional=None,
    izip=None, partition=-1, 
    **kwargs):
    global mode
    walltimes = []
    ref = time.time()
    ZIP = partition>0
    nzip = 2**partition if ZIP else 1
    if(ZIP)and(verbose):
        print(f"--- ZIP mode: {izip}/{nzip} ---")

    # Setting database
    path_in_repo = 'halo'
    names = list(name_dicts.keys())
    prefix = "HAL"
    full_path = f"{path}/{path_in_repo}/extended/{iout:05d}"
    if(not os.path.exists(full_path)): os.makedirs(full_path)
    uri.timer.verbose=0
    snap = uri.RamsesSnapshot(path, iout); snap.shmprefix = "extendhalo"
    uri.timer.verbose = 1 if verbose else 0

    names = list(name_dicts.keys())
    if onlymem:
        for name in names:
            if 'star' in name: del name_dicts[name]
            elif 'gas' in name: del name_dicts[name]
            elif 'cold' in name: del name_dicts[name]
            elif 'dense' in name: del name_dicts[name]
            elif '200' in name: del name_dicts[name]
            elif '500' in name: del name_dicts[name]
        names = list(name_dicts.keys())
    if(len(names)==0):
        print(f"\n=================\nSkip {iout}\n=================")
        return True
    print(f"Extend this: {names}")


    result_dtype = [(name, 'f8') for name in names]

    # Member need?
    need_members = {
        'mcontam':['m'],
        'vmaxcir':['x','y','z','m'],
        'cNFW':['x','y','z','m'],
        'inslope':['x','y','z','m']}
    ftmp = []; fields = ['x', 'y', 'z', 'vx','vy','vz', 'm', 'epoch', 'metal']
    for name in names:
        if(name in need_members): ftmp += need_members[name]
    if(len(ftmp)>0):
        mtarget_fields = [field for field in fields if field in list(set(ftmp))]
        need_member = True
        if(verbose): print(f" > Member fields: {mtarget_fields}")
    # Cell need?
    need_cells = {
        'm500':['x', 'y', 'z', 'rho', 'level'],
        'mgas_r500':['x', 'y', 'z', 'rho', 'level'],
        'mdense_r500':['x', 'y', 'z', 'rho', 'P', 'level']}
    ctmp = []; ctarget_fields = [
        'x', 'y', 'z', 'vx','vy','vz', 'rho','P','level','metal',
        'H', 'O', 'Fe', 'Mg', 'C', 'Si', 'N', 'S', 'D', 'd1', 'd2', 'd3', 'd4']
    for name in names:
        if(name in need_cells): ctmp += need_cells[name]
    if(len(ctmp)>0):
        ctarget_fields = [field for field in ctarget_fields if field in list(set(ctmp))]
        need_cell = True
        if(verbose): print(f" > Cell fields: {ctarget_fields}")
    # DM need?
    need_dms = {
        'm500':['x','y','z','m']}
    dtmp = []; dtarget_fields = ['x','y','z','vx','vy','vz','m']
    for name in names:
        if(name in need_dms): dtmp += need_dms[name]
    if(len(dtmp)>0):
        dtarget_fields = [field for field in dtarget_fields if field in list(set(dtmp))]
        need_dm = True
        if(verbose): print(f" > DM fields: {dtarget_fields}")
    # Star need?
    snapstar = None
    need_stars = {
        'm500':['x','y','z','m'],
        'mstar_r500':['x','y','z','m']}
    stmp = []; starget_fields = ['x','y','z','vx','vy','vz','m']
    for name in names:
        if(name in need_stars): stmp += need_stars[name]
    if(len(stmp)>0):
        starget_fields = [field for field in starget_fields if field in list(set(stmp))]
        need_star = True
        snapstar = uri.RamsesSnapshot(path, iout); snapstar.shmprefix = "extendhalo"
        if(verbose): delprint(1)
        if(verbose): print(f" > Star fields: {starget_fields}")

    # Load HaloMaker
    sparams = snap.params; sunits = snap.unit
    table = uhmi.HaloMaker.load(snap, galaxy=False, extend=False)
    if(ZIP):
        ntable = len(table)
        if(partition >= 1): # nzip=2, 4, 8
            medx = np.median(table['x']); checkx = izip%2
            table = table[table['x']<medx] if checkx==0 else table[table['x']>=medx]
        if(partition >= 2): # nzip=4, 8
            medy = np.median(table['y']); checky = (izip//2)%2
            table = table[table['y']<medy] if checky==0 else table[table['y']>=medy]
        if(partition == 3): # nzip=8
            medz = np.median(table['z']); checkz = (izip//4)%2
            table = table[table['z']<medz] if checkz else table[table['z']>=medz]
        if(partition>3):
            raise ValueError("Partition should be 1, 2, or 3")
        if(verbose):
            print(f" > Partition Level: {partition}, ({izip}/{nzip})")
            print(f" > Table: {ntable}->{len(table)}")

    if(verbose): print(f" > Calculate for {len(table)} {path_in_repo}s")
    domain = [None for _ in range(len(table))]
    walltime = ("Preparation", time.time()-ref); walltimes.append(walltime); ref = time.time()

    # Load Member particles
    snapm = None
    members=None; cparts=None
    dm_memory = (None, None, None, None, None)
    star_memory = (None, None, None, None, None)
    cell_memory = (None, None, None, None, None)
    if(need_member):
        if(verbose): print(f" > Read member particles ({np.sum(table['nparts'])})")
        snapm = uri.RamsesSnapshot(path, iout); snapm.shmprefix = "extendhalo"
        members = uhmi.HaloMaker.read_member_parts(snapm, table, galaxy=False, nthread=nthread, copy=True, target_fields=mtarget_fields)
        nparts = table['nparts']
        cparts = np.cumsum(nparts); cparts = np.insert(cparts, 0, 0)
        if(verbose): delprint(2)
    walltime = ("Read member", time.time()-ref); walltimes.append(walltime); ref = time.time()

    # Load Raw data
    if(need_dm)or(need_cell)or(need_star):
        fdomain = f"{path}/{path_in_repo}/{prefix}_{iout:05d}/domain_{iout:05d}.dat"
        if(os.path.exists(fdomain)):
            if(verbose): print(f" > Load domain")
            domain = domload(fdomain, msg=verbose)
            if(ZIP): domain = [domain[ith-1] for ith in table['id']]
            cpulist = np.unique( np.concatenate( domain ) )
        else:
            if(verbose): print(f" > Get halos cpu list")
            cpulist, domain = snap.get_halos_cpulist(table, nthread=nthread, full=True)
            if(not ZIP): domsave(fdomain, domain)
        if(need_dm):
            if(verbose): print(f" > Get DM")
            snap.get_part(pname='dm', nthread=nthread, target_fields=dtarget_fields, cpulist=cpulist)
            dshape = snap.part.shape; daddress = snap.part_mem.name; ddtype = snap.part.dtype
            cpulist_dm = snap.cpulist_part; bound_dm = snap.bound_part
            dm_memory = (dshape, daddress, ddtype, cpulist_dm, bound_dm)     
        if(need_star):
            if(verbose): print(f" > Get Star")
            if snapstar.star[0]:
                snapstar.get_part(pname='star', nthread=nthread, target_fields=starget_fields, cpulist=cpulist)
                sshape = snapstar.part.shape; saddress = snapstar.part_mem.name; sdtype = snapstar.part.dtype
                cpulist_star = snapstar.cpulist_part; bound_star = snapstar.bound_part
                star_memory = (sshape, saddress, sdtype, cpulist_star, bound_star)
            else:
                star_memory = (None, None, None, None, None)
        if(need_cell):
            if(verbose): print(f" > Get Cell")
            snap.get_cell(target_fields=ctarget_fields, nthread=nthread, cpulist=cpulist)
            cshape = snap.cell.shape; caddress = snap.cell_mem.name; cdtype = snap.cell.dtype
            cpulist_cell = snap.cpulist_cell; bound_cell = snap.bound_cell
            cell_memory = (cshape, caddress, cdtype, cpulist_cell, bound_cell)
        walltime = ("Read raw", time.time()-ref); walltimes.append(walltime); ref = time.time()

    # Preprocess
    if(pre_func is not None):
        if(verbose): print(f" > Preprocess datasets")
        table, snapm, members, snap, snapstar, dm_memory, star_memory, cell_memory = pre_func(names, table, snapm, members, snap, snapstar, dm_memory, star_memory, cell_memory, full_path, nthread, verbose)
        walltime = ("Preprocess", time.time()-ref); walltimes.append(walltime); ref = time.time()


    # Assign shared memory
    if(verbose): print(f" > Make shared memory")
    shmname = f"extendhalo_{mode}_{path_in_repo}_{snap.iout:05d}"
    if(os.path.exists(f"/dev/shm/{shmname}")): os.remove(f"/dev/shm/{shmname}")
    result_table = np.empty(len(table), dtype=result_dtype)
    memory = shared_memory.SharedMemory(name=shmname, create=True, size=result_table.nbytes)
    result_table = np.ndarray(result_table.shape, dtype=result_dtype, buffer=memory.buf)
    shape = result_table.shape; address = memory.name; dtype = result_dtype

    # Main Calculation
    if(verbose): print(f" > Start Calculation")
    if(verbose):
        pbar = tqdm(total=len(table), desc=f"Nthread={min(len(table), nthread)}")
        def update(*a): pbar.update()
    else: update = None
    if(snap is not None): signal.signal(signal.SIGTERM, signal.SIG_DFL)
    with Pool(processes=min(len(table),nthread)) as pool:
        async_result = [pool.apply_async(calc_func, args=(i, table[i], shape, address, dtype, sparams, sunits, getmem(members, cparts, i), dm_memory, star_memory, cell_memory, domain[i]), callback=update) for i in range(len(table))]
        iterobj = async_result
        for result in iterobj: result.get()
    if(snap is not None): signal.signal(signal.SIGTERM, snap.terminate)
    if(verbose):
        pbar.close(); delprint(1)
    walltime = ("Get results", time.time()-ref); walltimes.append(walltime); ref = time.time()
    if('cNFW' in result_table.dtype.names)and('inslope' in result_table.dtype.names):
        NnanNFW = np.sum(np.isnan(result_table['cNFW']))
        Nnanslope = np.sum(np.isnan(result_table['inslope']))
        print(f"#### {NnanNFW}, {Nnanslope} of {len(result_table)}failed fitting ####")

    # Dump and relase memory
    if(verbose): print(f" > Dumping")
    dump_func(result_table, table, full_path, iout, name_dicts, verbose, izip, partition)
    memory.close(); memory.unlink()
    if(snap is not None): snap.clear()
    if(snapstar is not None): snapstar.clear()
    if(snapm is not None): snapm.clear()
    if(verbose): print(f" Done\n")
    walltime = ("Dump", time.time()-ref); walltimes.append(walltime); ref = time.time()
    if(verbose):
        for name, walltime in walltimes: print(f" > {name}: {walltime:.2f} sec")
    return False
# --------------------------------------------------------------


# --------------------------------------------------------------
# Execution
# --------------------------------------------------------------
if __name__ == "__main__":
    # Import calculation functions
    from extend_halo import default_names, pre_func, calc_func, dump_func, skip_func, inhouse

    # Set path
    path = inhouse[mode]
    path_in_repo = 'halo'

    # Load nout
    full_path = f"{path}/{path_in_repo}/"
    bricks = glob.glob(f"{full_path}/tree_bricks*")
    nout = [int(ib[-5:]) for ib in bricks]
    nout.sort()
    
    # Run!
    iterator = nout[::-1]# if verbose else tqdm(nout)
    for iout in iterator:
        # if(iout>100): continue
        if sep>=0:
            if iout%4 != sep: continue
        names = skip_func(path, iout, default_names, verbose)
        skip = len(names)==0
        if(skip):
            print(f"\n=================\nSkip {iout}\n=================")
            continue
        now = datetime.datetime.now() 
        now = f"--{now}--" if verbose else now
        print(f"\n=================\nStart {iout} {now}\n================="); ref = time.time()
        nzip = 2**partition if ZIP else 1
        for izip in range(nzip):
            skipped = calc_extended(
                path, iout, names, 
                pre_func, calc_func, dump_func, 
                verbose=verbose, nthread=nthread,izip=izip, partition=partition)
            if skipped: break
        if(not skipped): print(f"Done ({time.time()-ref:.2f} sec)")
