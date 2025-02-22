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

for iout in table:
    if nmod>0:
        if iout%nmod != mod: continue
    file_tracer = f"{path}/tracer_{iout:03d}.pkl"
    if os.path.exists(file_tracer): continue
    isnap = snaps.get_snap(iout)
    isnap.get_part(pname='tracer', target_fields=['x','y','z','id','family'], nthread=nthread)
    argsort = np.argsort(isnap.part['id'])
    tracer = isnap.part[argsort]
    dump(tracer, file_tracer, msg=False)
    isnap.clear()
