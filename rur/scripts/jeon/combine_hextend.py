import numpy as np; print(np.__version__)
assert (np.__version__ < '2.0.0'), "numpy version is too high, don't use pickle in Numpy2"

from rur import uri
from rur.utool import load, dump, datload
from rur.sim import NewCluster, NewHorizon2
import os
from tqdm import tqdm
import shutil





NC = NewCluster(623)
NCs = uri.TimeSeries(NC)
NCs.read_iout_avail()
print()
print(NCs.iout_avail['iout'])




hnames_nc = ['mcontam','vmaxcir','rmaxcir',
    'r200','m200','r500','m500',
    'mdm_r200','mdm_r500','mdm','mdm_rvir',
    'mstar_r200','mstar_r500','mstar','mstar_rvir',
    'mgas_r500','mgas','mgas_rvir','mgas_r200',
    'mcold_r500','mcold','mcold_rvir','mcold_r200',
    'mdense','mdense_rvir','mdense_r200','mdense_r500',
    'cNFW','cNFWerr','inslope','inslopeerr',
]
hdtype_nc = np.dtype([(f, 'f8') for f in hnames_nc])
names = hnames_nc




skip = 623




uri.timer.verbose=0
nout = NCs.iout_avail['iout']
hdict = {}
for iout in nout:
    # Set path
    if iout < skip: continue
    NC = NCs.get_snap(iout)
    path = f"{NC.repo}/halo/extended/{iout:05d}"
    if(not os.path.exists(path)): continue # Not yet extended
    hpath = f"{path}/halo"
    if(not os.path.exists(hpath)): os.makedirs(hpath)
    
    descs = {}
    rerun = False
    if(os.path.exists(f"{path}/desc.pkl")):
        hextra = []
        descs = load(f"{path}/desc.pkl", msg=False)
        for hname in hnames_nc:
            if not hname in descs:
                descs[hname] = 'Not ready'
                hextra.append(hname)
                rerun = True
            elif(descs[hname]=='Not ready'):
                hextra.append(hname)
                rerun = True
        if(not rerun):
            print(f"{iout:05d} is already done")
            continue # already done
        print(f"[{iout:05d}] Extra: {hextra}")

    if(not rerun):
        # Load data
        nhal = 0
        for hname in hnames_nc:
            fname = f"{path}/{hname}_{iout:05d}.dat"
            data = None; desc = 'Not ready'
            if(os.path.exists(fname)):
                data, desc = datload(fname)
                if(nhal==0): nhal=len(data)
            descs[hname] = desc; hdict[hname] = data
        if(nhal==0): continue # No halos


        harr = np.empty(1, dtype=hdtype_nc)[0]
        for i in tqdm( range(nhal), desc=f"{iout}" ):
            for hname in hnames_nc:
                harr[hname] = np.nan if(hdict[hname] is None) else hdict[hname][i]
            dump(harr, f"{hpath}/{i+1:07d}.pkl", msg=False)
        dump(descs, f"{path}/desc.pkl", msg=False)
    else:
        # Load data
        nhal = 0
        for hname in hnames_nc:
            if(hname in hextra):
                fname = f"{path}/{hname}_{iout:05d}.dat"
                data = None; desc = 'Not ready'
                if(os.path.exists(fname)):
                    data, desc = datload(fname)
                    if(nhal==0): nhal=len(data)
                descs[hname] = desc; hdict[hname] = data
        if(nhal==0): continue # No halos
        
        harr = np.empty(1, dtype=hdtype_nc)[0]
        for i in tqdm( range(nhal), desc=f"{iout}" ):
            tmp = load(f"{hpath}/{i+1:07d}.pkl", msg=False)
            for hname in hnames_nc:
                if(hname in hextra):
                    harr[hname] = np.nan if(hdict[hname] is None) else hdict[hname][i]
                else:
                    harr[hname] = tmp[hname]
            dump(harr, f"{hpath}/{i+1:07d}.pkl", msg=False)
        dump(descs, f"{path}/desc.pkl", msg=False)





avail = np.full((len(nout), len(names)), False, dtype=bool)
if(os.path.exists(f"{NC.repo}/halo/extended/avail.pkl2")):
    avail = load(f"{NC.repo}/halo/extended/avail.pkl2")['avail']
if(os.path.exists(f"{NC.repo}/halo/extended/avail.pkl")):
    avail = load(f"{NC.repo}/halo/extended/avail.pkl")['avail']

for i, iout in tqdm(enumerate(nout), total=len(nout)):
    if(avail[i].all()): continue
    path = f"{NC.repo}/halo/extended/{iout:05d}"
    for j, name in enumerate(names):
        if(avail[i, j]): continue
        if os.path.exists(f"{path}/{name}_{iout:05d}.dat"):
            avail[i, j] = True

extend_avail = dict(
    nout = nout,
    names = names,
    avail = avail,
)




dump(extend_avail, f"{NC.repo}/halo/extended/avail.pkl")