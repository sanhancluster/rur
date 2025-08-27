import tqdm
import numpy as np
import matplotlib.pyplot as plt
from rur import uri, uhmi
import time, os, h5py, argparse
import socket
host = socket.gethostname()

# Arguments
parser = argparse.ArgumentParser(description='Validate HDF5 files for RAMSES snapshots')
parser.add_argument('repo', type=str, default='/storage7/NewCluster', help='Repository path')
parser.add_argument('--nthread', type=int, default=16, help='Number of threads to use')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
args = parser.parse_args()
nthread = args.nthread
repo = args.repo
verbose = args.verbose

# Snapshots
uri.timer.verbose = int(verbose)
snap1 = uri.RamsesSnapshot(repo, 1); snap1s = uri.TimeSeries(snap1, auto=True)
snap2 = uri.RamsesSnapshot(repo, 1); snap2s = uri.TimeSeries(snap2, auto=True)
iout_table = snap1s.read_iout_avail()
iouts = iout_table['iout']

# Test sets
testlist = ['leaf', 'branch', 'dm', 'star', 'tracer', 'sink']
oldtxt = "\t".join([f"told{ith+1}" for ith in range(6)])
newtxt = "\t".join([f"tnew{ith+1}" for ith in range(6)])
dtype1 = [('iout', 'i4'), ('cell', 'i1'), ('part', 'i1'), ('valid', 'i1')]
dtype2 = [(f'told_{testlist[ith]}', float) for ith in range(6)]
dtype3 = [(f'tnew_{testlist[ith]}', float) for ith in range(6)]
dtype = dtype1 + dtype2 + dtype3

# Initialize
comments = [
    f"# ({host}) `{repo}`",
    f"# 0) iout: Snapshot ith output",
    f"# 1) cell: Leaf Cell HDF5 file exists",
    f"# 2) part: Particle HDF5 file exists",
    f"# 3) valid: Validation result",
    f"# 4) told1: (Raw) Leaf Cell time (s)",
    f"# 5) told2: (Raw) Branch Cell time (s)",
    f"# 6) told3: (Raw) DM Particle time (s)",
    f"# 7) told4: (Raw) Star Particle time (s)",
    f"# 8) told5: (Raw) Tracer Particle time (s)",
    f"# 9) told6: (Raw) Sink Particle time (s)",
    f"# 10) tnew1: (HDF) Leaf Cell time (s)",
    f"# 11) tnew2: (HDF) Branch Cell time (s)",
    f"# 12) tnew3: (HDF) DM Particle time (s)",
    f"# 13) tnew4: (HDF) Star Particle time (s)",
    f"# 14) tnew5: (HDF) Tracer Particle time (s)",
    f"# 15) tnew6: (HDF) Sink Particle time (s)",
    f"# iout\tcell\tpart\tvalid\t{oldtxt}\t{newtxt}",
]
comment = "\n".join(comments)
# Initialize
FILE_VALID = f"{snap2.repo}/hdf/valid.txt"
if not os.path.exists(FILE_VALID):
    with open(FILE_VALID, 'w') as f:
        # Write header
        for line in comments:
            f.write(f"{line}\n")
        # f.write(f"# iout\tcell\tpart\tvalid\t{oldtxt}\t{newtxt}\n")
valid = np.genfromtxt(FILE_VALID, skip_header=len(comments)-1, dtype=dtype, delimiter='\t')

# Misc functions
def exist_hdf(path):
    try:
        if not os.path.exists(path):
            return False
        with h5py.File(path, 'r') as fl:
            return 'version' in fl.attrs.keys()
    except:
        return False
    
def _valid_cell(snap, **kwargs):
    verbose = kwargs.pop('verbose', False)
    ref = time.time()
    cell = snap.get_cell(**kwargs)
    leng = len(cell)
    x = np.median(cell['x'])
    snap.clear(verbose=int(verbose))
    return leng, x, time.time() - ref

def _valid_part(snap, **kwargs):
    verbose = kwargs.pop('verbose', False)
    ref = time.time()
    pname = kwargs.get('pname', 'dm')
    if pname == 'sink':
        part = snap.get_sink(all=True)
    else:
        part = snap.get_part(**kwargs)
    leng = len(part)
    x = np.median(part['x'])
    snap.clear(verbose=int(verbose))
    return leng, x, time.time() - ref

def _identical(leng1, leng2, x1, x2):
    msg = f"({leng1} vs {leng2}) & ({x1:.6f} vs {x2:.6f})"
    if leng1 != leng2:
        return False, msg
    if not np.allclose(x1, x2):
        return False, msg
    return True, msg

# Validation Function
def validation(snap1:uri.RamsesSnapshot, snap2:uri.RamsesSnapshot, nthread=16, verbose=False):
    snap1.box = snap1.default_box
    snap2.box = snap2.default_box
    gals = uhmi.HaloMaker.load(snap1, galaxy=True)
    target = None
    if len(gals)>0: target = gals[np.argmax(gals['m'])]
    times1 = [-1,-1,-1,-1,-1,-1]
    times2 = [-1,-1,-1,-1,-1,-1]

    # Check Leaf Cell
    if verbose: print(f"Validating Leaf Cell")
    ith = 0
    leng1, x1, times1[ith] = _valid_cell(snap1, nthread=nthread, hdf=False, read_branch=False, verbose=verbose)
    leng2, x2, times2[ith] = _valid_cell(snap2, nthread=nthread, hdf=True, read_branch=False, verbose=verbose)
    same, msg = _identical(leng1, leng2, x1, x2)
    if verbose: print(f"{msg}\n")
    if not same:
        return False, times1, times2
    if target is not None:
        snap1.set_box_halo(target, radius_name='r'); snap2.set_box_halo(target, radius_name='r')
        leng1, x1, _ = _valid_cell(snap1, nthread=nthread, hdf=False, read_branch=False, verbose=verbose)
        leng2, x2, _ = _valid_cell(snap2, nthread=nthread, hdf=True, read_branch=False, verbose=verbose)
        same, msg = _identical(leng1, leng2, x1, x2)
        if verbose: print(f"{msg}\n")
        if not same:
            return False, times1, times2

    # Check Branch Cell
    if verbose: print(f"Validating Branch Cell")
    ith += 1
    leng1, x1, times1[ith] = _valid_cell(snap1, nthread=nthread, hdf=False, read_branch=True, verbose=verbose)
    leng2, x2, times2[ith] = _valid_cell(snap2, nthread=nthread, hdf=True, read_branch=True, verbose=verbose)
    same, msg = _identical(leng1, leng2, x1, x2)
    if verbose: print(f"{msg}\n")
    if not same:
        return False, times1, times2
    if target is not None:
        snap1.set_box_halo(target, radius_name='r'); snap2.set_box_halo(target, radius_name='r')
        leng1, x1, _ = _valid_cell(snap1, nthread=nthread, hdf=False, read_branch=True, verbose=verbose)
        leng2, x2, _ = _valid_cell(snap2, nthread=nthread, hdf=True, read_branch=True, verbose=verbose)
        same, msg = _identical(leng1, leng2, x1, x2)
        if verbose: print(f"{msg}\n")
        if not same:
            return False, times1, times2

    header = snap1.extract_header()
    # Check DMs
    if verbose: print(f"Validating DMs")
    ith += 1
    leng1, x1, times1[ith] = _valid_part(snap1, pname='dm', nthread=nthread, hdf=False, verbose=verbose)
    leng2, x2, times2[ith] = _valid_part(snap2, pname='dm', nthread=nthread, hdf=True, verbose=verbose)
    same, msg = _identical(leng1, leng2, x1, x2)
    if verbose: print(f"{msg}\n")
    if not same:
        if verbose: print(f"Length mismatch: {leng1} vs {leng2}\n")
        return False, times1, times2
    if target is not None:
        snap1.set_box_halo(target, radius_name='r'); snap2.set_box_halo(target, radius_name='r')
        leng1, x1, _ = _valid_part(snap1, pname='dm', nthread=nthread, hdf=False, verbose=verbose)
        leng2, x2, _ = _valid_part(snap2, pname='dm', nthread=nthread, hdf=True, verbose=verbose)
        same, msg = _identical(leng1, leng2, x1, x2)
        if verbose: print(f"{msg}\n")
        if not same:
            if verbose: print(f"Length mismatch: {leng1} vs {leng2}\n")
            return False, times1, times2

    if header['star'] > 0:
        # Check Stars
        if verbose: print(f"Validating Stars")
        ith += 1
        leng1, x1, times1[ith] = _valid_part(snap1, pname='star', nthread=nthread, hdf=False, verbose=verbose)
        leng2, x2, times2[ith] = _valid_part(snap2, pname='star', nthread=nthread, hdf=True, verbose=verbose)
        same, msg = _identical(leng1, leng2, x1, x2)
        if verbose: print(f"{msg}\n")
        if not same:
            if verbose: print(f"Length mismatch: {leng1} vs {leng2}\n")
            return False, times1, times2
        if target is not None:
            snap1.set_box_halo(target, radius_name='r'); snap2.set_box_halo(target, radius_name='r')
            leng1, x1, _ = _valid_part(snap1, pname='star', nthread=nthread, hdf=False, verbose=verbose)
            leng2, x2, _ = _valid_part(snap2, pname='star', nthread=nthread, hdf=True, verbose=verbose)
            same, msg = _identical(leng1, leng2, x1, x2)
            if verbose: print(f"{msg}\n")
            if not same:
                if verbose: print(f"Length mismatch: {leng1} vs {leng2}\n")
                return False, times1, times2

    if header['tracer'] > 0:
        # Check Tracer
        if verbose: print(f"Validating Tracer")
        ith += 1
        leng1, x1, times1[ith] = _valid_part(snap1, pname='tracer', nthread=nthread, hdf=False, verbose=verbose)
        leng2, x2, times2[ith] = _valid_part(snap2, pname='tracer', nthread=nthread, hdf=True, verbose=verbose)
        same, msg = _identical(leng1, leng2, x1, x2)
        if verbose: print(f"{msg}\n")
        if not same:
            if verbose: print(f"Length mismatch: {leng1} vs {leng2}\n")
            return False, times1, times2
        if target is not None:
            snap1.set_box_halo(target, radius_name='r'); snap2.set_box_halo(target, radius_name='r')
            leng1, x1, _ = _valid_part(snap1, pname='tracer', nthread=nthread, hdf=False, verbose=verbose)
            leng2, x2, _ = _valid_part(snap2, pname='tracer', nthread=nthread, hdf=True, verbose=verbose)
            same, msg = _identical(leng1, leng2, x1, x2)
            if verbose: print(f"{msg}\n")
            if not same:
                if verbose: print(f"Length mismatch: {leng1} vs {leng2}\n")
                return False, times1, times2

    if header['sink'] > 0:
        # Check Sinks
        if verbose: print(f"Validating Sinks")
        ith += 1
        leng1, x1, times1[ith] = _valid_part(snap1, pname='sink', nthread=nthread, hdf=False, verbose=verbose)
        leng2, x2, times2[ith] = _valid_part(snap2, pname='sink', nthread=nthread, hdf=True, verbose=verbose)
        same, msg = _identical(leng1, leng2, x1, x2)
        if verbose: print(f"{msg}\n")
        if not same:
            if verbose: print(f"Length mismatch: {leng1} vs {leng2}\n")
            return False, times1, times2
        if target is not None:
            snap1.set_box_halo(target, radius_name='r'); snap2.set_box_halo(target, radius_name='r')
            leng1, x1, _ = _valid_part(snap1, pname='sink', nthread=nthread, hdf=False, verbose=verbose)
            leng2, x2, _ = _valid_part(snap2, pname='sink', nthread=nthread, hdf=True, verbose=verbose)
            same, msg = _identical(leng1, leng2, x1, x2)
            if verbose: print(f"{msg}\n")
            if not same:
                if verbose: print(f"Length mismatch: {leng1} vs {leng2}\n")
                return False, times1, times2

    return True, times1, times2


# Validation Loop if main
if __name__ == "__main__":
    print()
    print()
    if len(valid)>0:
        print(f"Continue from previous validation results:")
        print(valid)
        print()
        print()
    print(f"See `{FILE_VALID}` for validation results.")
    print()
    print()
    for iout in tqdm.tqdm(iouts, desc="Validating HDFs"):
        if iout in valid['iout']: continue
        snap1 = snap1s.get_snap(iout)
        snap2 = snap2s.get_snap(iout)
        iscell = int( exist_hdf(snap1.get_path('hdf_cell')) )
        ispart = int( exist_hdf(snap2.get_path('hdf_part')) )
        ivalid = False
        times1 = [0,0,0,0,0,0]
        times2 = [0,0,0,0,0,0]
        if iscell and ispart:
            ivalid, times1, times2 = validation(snap1, snap2, nthread=nthread, verbose=False)
        timestr1 = "\t".join([f"{t:.2f}" for t in times1])
        timestr2 = "\t".join([f"{t:.2f}" for t in times2])

        valid_append = np.empty(1, dtype=dtype)
        valid_append['iout'] = iout
        valid_append['cell'] = iscell
        valid_append['part'] = ispart
        valid_append['valid'] = ivalid
        for ith, test in enumerate(testlist):
            valid_append[f'told_{test}'] = times1[ith]
            valid_append[f'tnew_{test}'] = times2[ith]
        valid = np.append(valid, valid_append)
        argsort = np.argsort(valid['iout'])
        valid = valid[argsort]
        # savetxt = f"{iout:05d}\t{iscell}\t{ispart}\t{int(ivalid)}\t{timestr1}\t{timestr2}"

        # with open(FILE_VALID, 'a') as f:
        #     f.write(f"{savetxt}\n")
        fmt = ['%d', '%d', '%d', '%d'] + ['%.2f'] * 12
        np.savetxt(FILE_VALID, valid, fmt=fmt, header=comment, comments='', delimiter='\t')