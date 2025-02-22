from rur import uri
import numpy as np
# assert np.__version__ < '2.0.0', 'numpy version should be less than 2.0.0'
import os, sys, glob
from rur.utool import *

from tqdm import tqdm
from multiprocessing import shared_memory, Pool
import argparse, time


parser = argparse.ArgumentParser(description='Fast tracer timeseries')
parser.add_argument("-r", "--repo", default='/storage7/NewCluster', required=False, help='Simulation repository', type=str)
parser.add_argument("-n", "--nthread", default=1, required=False, help='nthread', type=int)
parser.add_argument("-m", "--mod", default=-1, required=False, help='ioutmod', type=int)
parser.add_argument("-N", "--nmod", default=0, required=False, help='ioutNmod', type=int)
parser.add_argument("--verbose", action='store_true')
args = parser.parse_args()
print(args)

repo = args.repo
nthread = args.nthread
mod = args.mod
nmod = args.nmod
path = f"{repo}/TRACER/parking"
verbose = args.verbose
uri.timer.verbose = 1 if verbose else 0


snap = uri.RamsesSnapshot(repo, 1)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
table = snaps.iout_avail['iout']

def dump_as_dat(data1d, path, msg=False):
    leng = len(data1d)
    with open(path, "wb") as f:
        f.write(leng.to_bytes(4, byteorder='little'))
        f.write(data1d.tobytes())
    if(msg): print(f" `{path}` saved")

def load_from_dat(path, dtype='f8', msg=False):
    bsize = int( dtype[-1] )
    with open(path, "rb") as f:
        leng = int.from_bytes(f.read(4), byteorder='little')
        data = np.frombuffer(f.read(bsize*leng), dtype=dtype)
    if(msg): print(f" `{path}` loaded")
    return data

for iout in table:
    if nmod>0:
        if iout%nmod != mod: continue
    file_tracer = f"{path}/tracer_family_{iout:03d}.pkl"
    if os.path.exists(file_tracer): continue
    isnap = snaps.get_snap(iout)
    isnap.get_part(pname='tracer', target_fields=['x','y','z','id','family'], nthread=nthread)
    argsort = np.argsort(isnap.part['id'])
    tracer = isnap.part[argsort]

    # Save ID
    file_ids = f"{path}/tracer_ids.dat"
    if not os.path.exists(file_ids):
        dump_as_dat(tracer['id'], file_ids, msg=False)

    # Save other columns
    names = ['x','y','z','cpu','family']
    dtypes = ['f8','f8','f8','i2','i1']
    for name, dtype in zip(names, dtypes):
        file_name = f"{path}/tracer_{name}_{iout:03d}.dat"
        if not os.path.exists(file_name):
            dump_as_dat(tracer[name], file_name, msg=False)
    isnap.clear()
