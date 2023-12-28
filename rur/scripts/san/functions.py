import numpy as np
from tqdm import tqdm
import os
from rur import uri, painter, uhmi
import matplotlib.pyplot as plt

aexp = 0.
def get_times_from_log(log_path):
    steps = []
    timers = []
    for pp in log_path:
        f = open(pp, 'r')
        lines = f.readlines()
        for line in lines:
            if 'Fine step=' in line:
                aexp = float(line[50:61])
                if(not np.isfinite(aexp)):
                    print(line)
            if 'Time elapsed since last' in line:
                steps.append([float(line[37:46]), aexp])
            if 'TIMER' in line:
                timers.append()

    return steps


def draw_bh_sequence(ts, sp, list_sinkid, overwrite=False, base_path='/home/hansan/images/NC/sink_trace/massive',
                     viewer_kw=dict(show_smbh=[False, True], smbh_minmass=1E6, radius=7.5, mode=['phot', 'gas'])):
    dir_format = 'sink_%05d'
    img_format = '%05d.png'
    #iout_avail = ts.read_iout_avail()

    for sinkid in tqdm(list_sinkid):
        if (sinkid <= 0):
            continue
        tl = sp[sp['id'] == sinkid]
        iout_avail = ts.iout_avail[ts.iout_avail['icoarse'] > np.min(tl['icoarse'])]
        for av in tqdm(iout_avail[::-1]):
            snap = ts[av['iout']]
            if np.isscalar(base_path):
                base_path = [base_path]
                viewer_kw = [viewer_kw]

            for bpath, kw in zip(base_path, viewer_kw):
                path_format = os.path.join(bpath, dir_format, img_format)
                path = path_format % (sinkid, snap.iout)
                if (os.path.exists(path) and not overwrite):
                    continue
                sink = snap.get_sink(all=True)
                target = sink[sink['id'] == sinkid]
                if (target.size > 0):
                    target = target[0]
                else:
                    continue

                painter.viewer(snap, target=target, source=None, savefile=path, **kw)
                plt.clf()
                plt.close()

            snap.clear()


def get_timeline(sp, sp_base=None, rank=1, target_id=None, icoarse_max=None, icoarse_min=None, icoarse_rank=None):
    if icoarse_max is None:
        icoarse_max = np.max(sp['icoarse'])
    if icoarse_min is None:
        icoarse_min = np.min(sp['icoarse'])
    if icoarse_rank is None:
        icoarse_rank = icoarse_max

    sp_last = sp[sp['icoarse'] == icoarse_rank]
    if target_id is None:
        sp_last.sort(order='m')
        target_id = sp_last[-rank]['id']
    tl = sp[sp['id'] == target_id]

    mask = (tl['icoarse'] >= icoarse_min) & (tl['icoarse'] <= icoarse_max)
    tl = tl[mask]

    if sp_base is not None:
        tl_base = sp_base[sp_base['id'] == target_id]
        mask = ~np.isin(tl_base['icoarse'], tl['icoarse'])
        tl = np.concatenate([tl, tl_base[mask]])

    tl.sort(order='icoarse')
    return tl


