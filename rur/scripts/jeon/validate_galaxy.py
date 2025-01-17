print("ex: python3 Extend.py --mode nc --nthread 24 --verbose")

import numpy as np
from rur import uri, uhmi
from rur.utool import load, domload, domsave, domload_legacy
import os, glob, sys
from multiprocessing import Pool, shared_memory
from tqdm import tqdm
import argparse, time, datetime, signal
from extend_galaxy import inhouse, datdump, datload, match_sim


def delprint(n=1):
    """Delete the last line in the STDOUT."""
    for _ in range(n):
        sys.stdout.write("\x1b[1A")  # cursor up one line
        sys.stdout.write("\x1b[2K")  # delete the last line

parser = argparse.ArgumentParser(description='Extend HaloMaker (syj3514@yonsei.ac.kr)')
parser.add_argument("-m", "--mode", default='nc', required=False, help='Simulation mode', type=str)
parser.add_argument("-n", "--nthread", default=8, required=False, help='Ncore', type=int)
parser.add_argument("-s", "--sep", default=-1, required=False, help='Separation iout (s%4)', type=int)
parser.add_argument("-N", "--nsep", default=4, required=False, help="Nsep", type=int)
parser.add_argument("-p", "--partition", default=0, required=False, help='Divide galaxy domain (1=x, 2=xy, 3=xyz)', type=int)
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--nocell", action='store_true')
parser.add_argument("--chem", action='store_true')
parser.add_argument("--debug", action='store_true')
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
nsep = args.nsep
# partition:
#   If partition>0, the galaxy domain will be divided.
#   The number of divided domains is 2^partition. (1=x, 2=xy, 3=xyz)
#   Recommend to use this option when you lack memory.
partition = args.partition
ZIP = partition>0
# verbose:
#   If True, the progress bar will be shown and the verbose mode will be activated.
verbose = args.verbose
# nocell:
#   If True, the cell data will not be loaded.
nocell = args.nocell
# chem:
#   If True, the chemical data will be loaded.
chem = args.chem
galaxy = True
DEBUG = args.debug
if(nocell): chem = False
uri.timer.verbose = 1 if verbose else 0


def verify(path, iout,
            verbose=False, nthread=8,izip=None, partition=-1, DEBUG=False):
    
    walltimes = []
    ref = time.time()
    ZIP = partition>0
    nzip = 2**partition if ZIP else 1
    if(ZIP)and(verbose):
        print(f"--- ZIP mode: {izip}/{nzip} ---")

    # Setting database
    path_in_repo = 'galaxy'
    prefix = "GAL"
    full_path = f"{path}/{path_in_repo}/extended/{iout:05d}"
    if(not os.path.exists(full_path)): os.makedirs(full_path)
    if os.path.exists(f"{full_path}/wrong_verified.txt"): return True
    if os.path.exists(f"{full_path}/good_verified.txt"): return True
    uri.timer.verbose=0
    snap = uri.RamsesSnapshot(path, iout); snap.shmprefix = "extendgalaxy"
    uri.timer.verbose = 1 if verbose else 0

    # Load HaloMaker
    sparams = snap.params; sunits = snap.unit
    table = uhmi.HaloMaker.load(snap,galaxy=True, extend=False)
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

    fdomain = f"{path}/{path_in_repo}/{prefix}_{iout:05d}/domain_{iout:05d}.dat"
    if(os.path.exists(fdomain)):
        if(verbose): print(f" > Load domain")
        domain = domload(fdomain, msg=verbose)
        if(ZIP): domain = [domain[ith-1] for ith in table['id']]
        cpulist = np.unique( np.concatenate( domain ) )
    else:
        if(verbose): print(f" > Get halos cpu list")
        cpulist, domain = snap.get_halos_cpulist(table, nthread=nthread, full=True, manual=True)
        if(not ZIP): domsave(fdomain, domain)
        
    ctarget_fields = ['x', 'y', 'z', 'rho', 'level']
    snap.get_cell(target_fields=ctarget_fields, nthread=nthread, cpulist=cpulist)
    cshape = snap.cell.shape; caddress = snap.cell_mem.name; cdtype = snap.cell.dtype
    cpulist_cell = snap.cpulist_cell; bound_cell = snap.bound_cell
    cell_memory = (cshape, caddress, cdtype, cpulist_cell, bound_cell)
    walltime = ("Read raw", time.time()-ref); walltimes.append(walltime); ref = time.time()

    # Assign shared memory
    if(verbose): print(f" > Make shared memory")
    shmname = f"extendgalaxy_{mode}_{path_in_repo}_{snap.iout:05d}"
    if(os.path.exists(f"/dev/shm/{shmname}")): os.remove(f"/dev/shm/{shmname}")
    result_table = np.empty(len(table), dtype='f8')
    memory = shared_memory.SharedMemory(name=shmname, create=True, size=result_table.nbytes)
    result_table = np.ndarray(result_table.shape, dtype='f8', buffer=memory.buf)
    shape = result_table.shape; address = memory.name; dtype = 'f8'

    # Main Calculation
    if(verbose): print(f" > Start Calculation")
    if(verbose):
        pbar = tqdm(total=len(table), desc=f"Nthread={min(len(table), nthread)}")
        def update(*a): pbar.update()
    else: update = None
    if(snap is not None): signal.signal(signal.SIGTERM, signal.SIG_DFL)
    with Pool(processes=min(len(table),nthread)) as pool:
        async_result = [pool.apply_async(_verify, args=(i, table[i], shape, address, dtype, sparams, sunits, cell_memory, domain[i]), callback=update) for i in range(len(table))]
        iterobj = async_result
        for result in iterobj: result.get()
    if(snap is not None): signal.signal(signal.SIGTERM, snap.terminate)
    if(verbose):
        pbar.close(); delprint(1)
    walltime = ("Get results", time.time()-ref); walltimes.append(walltime); ref = time.time()

    preexist = datload(f"{full_path}/M_gas_{iout:05d}.dat", msg=False)[0]
    allclose = np.allclose(result_table, preexist)
    if not allclose:
        print()
        print("Different!!!!!")
        print()
        np.savetxt(f"{full_path}/wrong_verified.txt", np.vstack((result_table, preexist)))
    else:
        print()
        print("Safe!!")
        print()
        np.savetxt(f"{full_path}/good_verified.txt", np.vstack((result_table, preexist)))
    memory.close(); memory.unlink(); snap.clear()
    if(verbose): print(f" Done\n")
    walltime = ("Dump", time.time()-ref); walltimes.append(walltime); ref = time.time()
    if(verbose):
        for name, walltime in walltimes: print(f" > {name}: {walltime:.2f} sec")
    return allclose

# This is used in multiprocessing
def _verify(i, halo, shape, address, dtype, sparams, sunits, cell_memory, cdomain):
    debug=False
    # Common
    exist = shared_memory.SharedMemory(name=address)
    result_table = np.ndarray(shape, dtype=dtype, buffer=exist.buf)

    # Hydro
    cellmass = None

    if(debug)and(i==0): print(" [CalcFunc] > Hydro")
    # halo prop
    cx = halo['x']; cy = halo['y']; cz = halo['z']

    # Load cells
    if(debug)and(i==0): print(" [CalcFunc] > Load cell")
    cshape, caddress, cdtype, cpulist_cell, bound_cell = cell_memory
    cexist = shared_memory.SharedMemory(name=caddress)
    allcells = np.ndarray(cshape, dtype=cdtype, buffer=cexist.buf)
    cells = uri.domain_slice(allcells, cdomain, cpulist_cell, bound_cell)
    cdist = np.sqrt( (cells['x']-cx)**2 + (cells['y']-cy)**2 + (cells['z']-cz)**2 )
    rmask = cdist <= halo['r']
    if(np.sum(rmask) < 8):
        rmask = cdist <= (halo['r'] + (1 / 2**cells['level'])/2)

    cells = cells[rmask]; cdist = cdist[rmask]
    dx = 1 / 2**cells['level']
    vol = dx**3

    # Cell mass
    if(debug)and(i==0): print(" [CalcFunc] > cell mass")
    cellmass = cells['rho']*dx**3 / sunits['Msol']
    result_table[i] = np.sum(cellmass)


# --------------------------------------------------------------
# Execution
# --------------------------------------------------------------
if __name__ == "__main__":
    # Set path
    path = inhouse[mode]
    path_in_repo = 'galaxy'

    # Load nout
    full_path = f"{path}/{path_in_repo}/"
    bricks = glob.glob(f"{full_path}/tree_bricks*")
    nout = [int(ib[-5:]) for ib in bricks]
    nout.sort()
    
    # Run!
    iterator = nout# if verbose else tqdm(nout)
    for iout in iterator:
        #if(iout>576): continue
        if sep>=0:
            if iout%nsep != sep: continue

        now = datetime.datetime.now() 
        now = f"--{now}--" if verbose else now
        print(f"\n=================\nStart {iout} {now}\n================="); ref = time.time()

        nzip = 2**partition if ZIP else 1
        for izip in range(nzip):
            allclose = verify(
                path, iout,
                verbose=verbose, nthread=nthread,izip=izip, partition=partition, DEBUG=DEBUG)
