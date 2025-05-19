import numpy as np; print(np.__version__)
assert (np.__version__ < '2.0.0'), "numpy version is too high, don't use pickle in Numpy2"

from rur import uri
uri.timer.verbose=0
from rur.utool import load, dump, datload
import os
from tqdm import tqdm
import shutil




# ---------------------------------------------
# Edit this
snap = uri.RamsesSnapshot('/storage7/NewCluster', 623)
skip = 100000 # No skip
# skip = 623 # No check iout<skip
CHEMs = ['H','O','Fe','Mg','C','N', 'S','Si','D']
# CHEMs = [] # if no chem
DUSTs = ['CDustLarge','CDustSmall','SiDustLarge','SiDustSmall']
# DUSTs = [] # if no dust
# ----------------------------------------------








# ----------------------------------------------
# Load Snapshots
# ----------------------------------------------
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
print()
print(snaps.iout_avail['iout'])
nout = snaps.iout_avail['iout']

# ----------------------------------------------
# What to be extended
# ----------------------------------------------
pnames=[
    'umag','gmag','rmag','imag','zmag',
    'r50','r50g','r50i','r50r','r50u','r50z',
    'r90','r90g','r90i','r90r','r90u','r90z',
    'age','ageg','agei','ager','ageu','agez',
    'SFR','SFR_r50','SFR_r90','SFR10','SFR10_r50','SFR10_r90',
    'vsig','vsig_r50','vsig_r90',
    'SBu','SBu_r50','SBu_r90',
    'SBg','SBg_r50','SBg_r90',
    'SBr','SBr_r50','SBr_r90',
    'SBi','SBi_r50','SBi_r90',
    'SBz','SBz_r50','SBz_r90',
    'metal','MBH','dBH',
    ]
pdtype = [(f, 'f8') for f in pnames]

gnames=[
    'M_gas','M_gas_r50','M_gas_r90',
    'Mcold_gas','Mcold_gas_r50','Mcold_gas_r90',
    'Mdense_gas','Mdense_gas_r50','Mdense_gas_r90',
    'vsig_gas','vsig_gas_r50','vsig_gas_r90',
    'metal_gas',
    ]
gdtype = [(f, 'f8') for f in gnames]

cnames=[f"{chem}_gas" for chem in CHEMs] + [f"{dust}_gas" for dust in DUSTs]
cdtype = [(f, 'f8') for f in cnames]

names = pnames + gnames + cnames

# ----------------------------------------------
# Current status
# ----------------------------------------------
avail = np.full((len(nout), len(names)), False, dtype=bool)
if(os.path.exists(f"{snap.repo}/galaxy/extended/avail.pkl2")):
    avail = load(f"{snap.repo}/galaxy/extended/avail.pkl2")['avail']
if(os.path.exists(f"{snap.repo}/galaxy/extended/avail.pkl")):
    avail = load(f"{snap.repo}/galaxy/extended/avail.pkl")['avail']

# ----------------------------------------------
# Main run
# ----------------------------------------------
pdict = {}; gdict = {}; cdict = {}
for iout in nout:
    # Set path
    if iout < skip: continue
    snap = snaps.get_snap(iout)
    path = f"{snap.repo}/galaxy/extended/{iout:05d}"
    if(not os.path.exists(path)): continue # Not yet extended
    ppath = f"{path}/part"; gpath = f"{path}/gas"; cpath = f"{path}/chem"
    if(not os.path.exists(ppath)): os.makedirs(ppath)
    if(not os.path.exists(gpath)): os.makedirs(gpath)
    if(not os.path.exists(cpath)): os.makedirs(cpath)
    
    # Check if already done
    descs = {}; rerun = False
    if(os.path.exists(f"{path}/desc.pkl")):
        pextra = []; gextra = []; cextra = []
        descs = load(f"{path}/desc.pkl", msg=False)
        for pname in pnames:
            if(descs[pname]=='Not ready'):
                pextra.append(pname)
                rerun = True
        for gname in gnames:
            if(descs[gname]=='Not ready'):
                gextra.append(gname)
                rerun = True
        for cname in cnames:
            if(descs[cname]=='Not ready'):
                cextra.append(cname)
                rerun = True
        if(not rerun):
            print(f"{iout:05d} is already done")
            continue # already done
        print(f"[{iout:05d}] Extra: {pextra}, {gextra}, {cextra}")

    # First time
    if(not rerun):
        # Load data
        ngal = 0
        for pname in pnames:
            fname = f"{path}/{pname}_{iout:05d}.dat"
            data = None; desc = 'Not ready'
            if(os.path.exists(fname)):
                data, desc = datload(fname)
                if(ngal==0): ngal=len(data)
            descs[pname] = desc; pdict[pname] = data
        if(ngal==0): continue # No galaxies
        for gname in gnames:
            fname = f"{path}/{gname}_{iout:05d}.dat"
            data = None; desc = 'Not ready'
            if(os.path.exists(fname)):
                data, desc = datload(fname)
                if(ngal==0): ngal=len(data)
            descs[gname] = desc; gdict[gname] = data
        for cname in cnames:
            fname = f"{path}/{cname}_{iout:05d}.dat"
            data = None; desc = 'Not ready'
            if(os.path.exists(fname)):
                data, desc = datload(fname)
                if(ngal==0): ngal=len(data)
            descs[cname] = desc; cdict[cname] = data


        parr = np.empty(1, dtype=pdtype)[0]
        garr = np.empty(1, dtype=gdtype)[0]
        carr = np.empty(1, dtype=cdtype)[0]
        for i in tqdm( range(ngal), desc=f"{iout}" ):
            for pname in pnames:
                parr[pname] = np.nan if(pdict[pname] is None) else pdict[pname][i]
            dump(parr, f"{ppath}/{i+1:07d}.pkl", msg=False)
            for gname in gnames:
                garr[gname] = np.nan if(gdict[gname] is None) else gdict[gname][i]
            dump(garr, f"{gpath}/{i+1:07d}.pkl", msg=False)
            for cname in cnames:
                carr[cname] = np.nan if(cdict[cname] is None) else cdict[cname][i]
            dump(carr, f"{cpath}/{i+1:07d}.pkl", msg=False)
        dump(descs, f"{path}/desc.pkl", msg=False)
    # Rerun
    else:
        # Load data
        ngal = 0
        for pname in pnames:
            if(pname in pextra):
                fname = f"{path}/{pname}_{iout:05d}.dat"
                data = None; desc = 'Not ready'
                if(os.path.exists(fname)):
                    data, desc = datload(fname)
                    if(ngal==0): ngal=len(data)
                descs[pname] = desc; pdict[pname] = data
        for gname in gnames:
            if(gname in gextra):
                fname = f"{path}/{gname}_{iout:05d}.dat"
                data = None; desc = 'Not ready'
                if(os.path.exists(fname)):
                    data, desc = datload(fname)
                    if(ngal==0): ngal=len(data)
                descs[gname] = desc; gdict[gname] = data
        for cname in cnames:
            if(cname in cextra):
                fname = f"{path}/{cname}_{iout:05d}.dat"
                data = None; desc = 'Not ready'
                if(os.path.exists(fname)):
                    data, desc = datload(fname)
                    if(ngal==0): ngal=len(data)
                descs[cname] = desc; cdict[cname] = data
        if(ngal==0): continue # No galaxies
        
        parr = np.empty(1, dtype=pdtype)[0]
        garr = np.empty(1, dtype=gdtype)[0]
        carr = np.empty(1, dtype=cdtype)[0]
        for i in tqdm( range(ngal), desc=f"{iout}" ):
            tmp = load(f"{ppath}/{i+1:07d}.pkl", msg=False)
            for pname in pnames:
                if(pname in pextra):
                    parr[pname] = np.nan if(pdict[pname] is None) else pdict[pname][i]
                else:
                    parr[pname] = tmp[pname]
            dump(parr, f"{ppath}/{i+1:07d}.pkl", msg=False)
            tmp = load(f"{gpath}/{i+1:07d}.pkl", msg=False)
            for gname in gnames:
                if(gname in gextra):
                    garr[gname] = np.nan if(gdict[gname] is None) else gdict[gname][i]
                else:
                    garr[gname] = tmp[gname]
            dump(garr, f"{gpath}/{i+1:07d}.pkl", msg=False)
            tmp = load(f"{cpath}/{i+1:07d}.pkl", msg=False)
            for cname in cnames:
                if(cname in cextra):
                    carr[cname] = np.nan if(cdict[cname] is None) else cdict[cname][i]
                else:
                    carr[cname] = tmp[cname]
            dump(carr, f"{cpath}/{i+1:07d}.pkl", msg=False)
        dump(descs, f"{path}/desc.pkl", msg=False)

# ----------------------------------------------
# Record availability
# ----------------------------------------------
for i, iout in tqdm(enumerate(nout), total=len(nout)):
    if(avail[i].all()): continue
    path = f"{snap.repo}/galaxy/extended/{iout:05d}"
    for j, name in enumerate(names):
        if(avail[i, j]): continue
        if os.path.exists(f"{path}/{name}_{iout:05d}.dat"):
            avail[i, j] = True

# ----------------------------------------------
# Dump updated avail
# ----------------------------------------------
extend_avail = dict(
    nout = nout,
    names = names,
    avail = avail,
)
dump(extend_avail, f"{snap.repo}/galaxy/extended/avail.pkl")