from rur import uri
import numpy as np
import numba as nb
from rur.utool import *


@nb.jit(parallel=True)
def large_isin(a, b):
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in nb.prange(n):
        if a[i] in set_b:
            result[i] = True
    return result

repo = '/storage7/NewCluster'
snap = uri.RamsesSnapshot(repo, 1)
snaps = uri.TimeSeries(snap)
nout = snap.get_iout_avail()

ref_iout = 500
nout = nout[nout<=ref_iout]
print(nout)

fout = nout[-1]
print(f"Reference snapshot: {fout}")
snap = snaps.get_snap(fout)
ftracer = snap.get_part(pname='tracer', nthread=24, target_fields=['id','family','cpu'])
minid = np.min(ftracer['id'])
uri.timer.verbose=0

start_on_middle=True

# dict_keys(['mode', 'minid', 'desc', 'nout'])

path = "/storage8/NC/TRACER"
if(start_on_middle):
    fname = f"{path}/tracer_{999:03d}.pkl"
    tdict = load(fname, msg=False)
    header = load(f"{path}/header.pkl", msg=False)
    assert header['minid'] == minid
    os.rename(f"{path}/header.pkl", f"{path}/old/header.pkl")
    saved_nout = tdict['nout']
    print(f"Saved: {saved_nout[0]}~{saved_nout[-1]}")
    saved_fout = saved_nout[0]
    for iout in nout:
        if(iout <= saved_fout): continue
        isnap = snaps.get_snap(iout)
        tracer = isnap.get_part(pname='tracer', nthread=24, target_fields=['id','family','cpu'])
        thash = np.mod(tracer['id']-minid,1000)
        for ihash in tqdm(range(1000), desc=f'[{iout}] Add column...'):
            fname = f"{path}/tracer_{ihash:03d}.pkl"
            tdict = load(fname, msg=False)
            itracer = tracer[thash == ihash]
            argsort = np.argsort( (itracer['id']-minid)//1000 )
            newcol = itracer[argsort]['cpu'].reshape(-1,1)
            tdict['cpumap'] = np.hstack((newcol, tdict['cpumap']))
            tdict['nout'] = np.insert(tdict['nout'], 0, iout)
            os.rename(fname, f"{path}/old/tracer_{ihash:03d}.pkl")
            dump(tdict, fname, msg=False)
        header['nout'] = np.insert(header['nout'], 0, iout)
        isnap.clear()
        dump(header, f"{path}/header.pkl", msg=False)

else:
    header = dict(mode='nc', minid=minid, desc='(id-minid)%100 = file_suffix \n (id-minid)//100 = row_number_at_each_file', nout=np.array([fout]))
    fhash = np.mod(ftracer['id']-minid,1000)
    # Initialize tracer dictionary
    for ihash in tqdm(range(1000), desc='Initialize...'):
        fname = f"{path}/tracer_{ihash:03d}.pkl"
        if(os.path.exists(fname)):
            raise FileExistsError(f"`{fname}` already exists!")
        else:
            tdict = {}
            tdict['nout'] = nout[::-1]
            tdict['cpumap']=np.array([])

            itracer = ftracer[fhash == ihash]
            argsort = np.argsort( (itracer['id']-minid)//1000 )
            tdict['cpumap'] = itracer[argsort]['cpu'].reshape(-1,1)
        dump(tdict, fname, msg=False)
    dump(header, f"{path}/header.pkl", msg=False)

    for iout in nout[::-1][1:]:
        isnap = snaps.get_snap(iout)
        tracer = isnap.get_part(pname='tracer', nthread=24, target_fields=['id','family','cpu'])
        thash = np.mod(tracer['id']-minid,1000)
        for ihash in tqdm(range(1000), desc=f'[{iout}] Add column...'):
            fname = f"{path}/tracer_{ihash:03d}.pkl"
            tdict = load(fname)
            itracer = tracer[thash == ihash]
            argsort = np.argsort( (itracer['id']-minid)//1000 )
            tdict['cpumap'] = np.hstack((tdict['cpumap'], itracer[argsort]['cpu'].reshape(-1,1)))
            dump(tdict, fname, msg=False)
        isnap.clear()
        header['nout'] = np.append(header['nout'], iout)
        dump(header, f"{path}/header.pkl", msg=False)

'''
How to read?

For example,
you have a tracer ID=555819556.
Keep in mind that minID=555674170.
then, the hash of it is (ID-minID)%1000 = 86.

555819556-555674170 = 145386

Ok, now open the `tracer_086.pkl` file.
This is dictionary file(`nout` and `cpumap`)
You will see the `cpumap` array.
the row number is: (ID-minID)//1000 = 1453.
Then, the CPU number is `cpumap[1453]`.

So combine, `nout` abd `cpumap[1453]`,
You can get the CPU number of all snapshots(in nout)
'''