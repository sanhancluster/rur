from rur import uri
import numpy as np
assert np.__version__ < '2.0.0', "This script is not compatible with numpy 2.0.0 or later."
import os, sys, glob
from rur.utool import *

from tqdm import tqdm
from multiprocessing import shared_memory, Pool
import argparse, time

# python3 tracer_dump2.py --mod 0
parser = argparse.ArgumentParser(description='Fast tracer timeseries')
parser.add_argument("-m", "--mod", default=-1, required=False, help='ioutmod', type=int)
parser.add_argument("-n", "--nthread", default=1, required=False, help='nthread', type=int)
args = parser.parse_args()
mod = args.mod
nthread = args.nthread

path = f"/storage5/TRACER"
parking = f"{path}/parking"
repo = '/storage7/NewCluster'
snap = uri.RamsesSnapshot(repo, 1)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
table = snaps.iout_avail['iout'] # type: ignore


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

def dump_as_dats(data1d, path, first, clear_col=None, msg=False):
    if first:
        leng = len(data1d)
        with open(path, "wb") as f:
            f.write(leng.to_bytes(4, byteorder='little'))
            f.write(data1d.tobytes())
    else:
        if clear_col is not None:
            bsize = 8
            if "cpu" in path: bsize = 2
            if "family" in path: bsize = 1
            with open(path, "ab+") as f:
                f.seek(0)
                leng = int.from_bytes(f.read(4), byteorder='little')
                f.seek(4 + leng*bsize*clear_col)
                f.truncate()
        with open(path, "ab+") as f:
            f.write(data1d.tobytes())
    if(msg): print(f" `{path}` saved")

def load_from_dats(path, icols=None, dtype='f8', msg=False):
    bsize = int( dtype[-1] )
    with open(path, "rb") as f:
        f.seek(0, 2); 
        fsize = f.tell() - 4
        f.seek(0)
        leng = int.from_bytes(f.read(4), byteorder='little')
        nline = fsize//(leng*bsize)
        data = np.zeros((nline, leng), dtype=dtype)
        if icols is None:
            icols = np.arange(nline)
        for i in range(nline):
            if i in icols:
                data[i] = np.frombuffer(f.read(bsize*leng), dtype=dtype)
            else:
                f.read(bsize*leng)
        return data[icols]
    if(msg): print(f" `{path}` loaded")
    return data

ids = load_from_dat(f"{parking}/tracer_ids.dat", dtype='i4')
Ntracer = len(ids)
minid = 555674170
Nrow = 100000
Nhash = len(ids)//Nrow + 1
Nout = len(table)//100 + 1
def get_hash(iids):
    global Nrow, minid
    
    prefixs = (iids-minid)//Nrow
    irows = (iids-minid)%Nrow
    return prefixs, irows

names = {'x':'f8', 'y':'f8', 'z':'f8', 'family':'i1', 'tag':'i1', 'partp':'i4', 'cpu':'i4'}


header_name = f"{path}/header.pkl"
if os.path.exists(header_name):
    header = load(header_name)
else:
    header = dict(
        mode='nc', 
        minid=minid, 
        desc='(id-minid)%1000 = file_suffix \n (id-minid)//1000 = row_number_at_each_file', 
        nout=np.array([1]))
nout = header['nout']


for iout in tqdm(table):
    mynumber = iout//100
    if mod >= 0:
        if mynumber != mod: continue

    # Ith output check
    dirname = f"{path}/{mynumber}00"
    if not os.path.exists(dirname): os.makedirs(dirname)
    file_progress = f"{dirname}/progress.pkl"
    iout_mask = (table >= (mynumber)*100)&(table < (mynumber + 1)*100)
    icol = np.where(table[iout_mask] == iout)[0][0]

    # Check in progress
    calced_iout, done_icol = load(file_progress, msg=False) if os.path.exists(file_progress) else (iout, -1)
    if calced_iout > iout:
        if not iout in nout:
            nout = np.append(nout, iout)
            nout = np.unique(nout)
            header['nout'] = nout
            dump(header, header_name, msg=False)
        continue # Already done
    elif calced_iout == iout:
        if done_icol == icol:
            if not iout in nout:
                nout = np.append(nout, iout)
                nout = np.unique(nout)
                header['nout'] = nout
                dump(header, header_name, msg=False)
            continue # Already done
        else: clear_col = done_icol # Terminated in the middle
    else: clear_col = None

    # Parking
    isnap = snaps.get_snap(iout)
    isnap.get_part(pname='tracer', target_fields=['x','y','z','id','family','tag','partp'], nthread=nthread)
    assert isinstance(isnap.part, uri.Particle)
    argsort = np.argsort(isnap.part['id'])
    tracer = isnap.part.table[argsort]

    lenchunk = np.sum(iout_mask)
    dump(np.array([iout, done_icol]), file_progress, msg=False)
    for name, dtype in names.items():
        # parked = load_from_dat(f"{parking}/tracer_{name}_{iout:03d}.dat", dtype=dtype)
        parked = tracer[name]
        if name=='cpu': parked = parked.astype('i2')
        cursor = 0
        # for ihash in tqdm(range(Nhash), desc=f"[{iout}] ({name})"):
        for ihash in range(Nhash):
            npart = Nrow if ihash < Nhash-1 else Ntracer - Nrow*(Nhash-1)
            fname = f"{dirname}/tracer_{name}_{ihash:04d}.dat"
            if not os.path.exists(fname):
                first = True
            else:
                first = False
            arr = parked[cursor : cursor+npart]
            dump_as_dats(arr, fname, first, msg=False, clear_col=clear_col)
            cursor += npart
    dump(np.array([iout, icol]), file_progress, msg=False)
    nout = np.append(nout, iout)
    nout = np.unique(nout)
    header['nout'] = nout
    dump(header, header_name, msg=False)
    isnap.clear()
