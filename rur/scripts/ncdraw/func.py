from rur import uri, uhmi, painter
from rur.utool import datload, load, dump
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.colors as mcolors
import colorsys, os
import matplotlib as mpl

def detect_auto_refout(snap, snaps=None):
    print(" > Auto-detecting latest output (refout)...")
    if snaps is None: snaps = uri.TimeSeries(snap)

    latest = []

    # 1. Check in `iout_avail.txt`
    iout_avail = snaps.read_iout_avail()
    fout = iout_avail['iout'].max()
    latest.append(fout)
    print(f"    From `iout_avail.txt`: {fout}")

    # 2. Check HaloMaker output
    idir = f"{snap.repo}/halo"
    fnames = os.listdir(idir)
    fnames = [int(fname[-5:]) for fname in fnames if fname.startswith('tree_bricks')]
    fout = np.max(fnames)
    latest.append(fout)
    print(f"    From HaloMaker output: {fout}")

    # 3. Check GalaxyMaker output
    idir = f"{snap.repo}/galaxy"
    fnames = os.listdir(idir)
    fnames = [int(fname[-5:]) for fname in fnames if fname.startswith('tree_bricks')]
    fout = np.max(fnames)
    latest.append(fout)
    print(f"    From GalaxyMaker output: {fout}")

    # 4. Check Halo Extend output
    idir = f"{snap.repo}/halo/extended"
    avail = load(f"{idir}/avail.pkl", msg=False)['nout']
    fout = np.max(avail)
    latest.append(fout)
    print(f"    From Halo Extend output: {fout}")

    # 5. Check Galaxy Extend output
    idir = f"{snap.repo}/galaxy/extended"
    avail = load(f"{idir}/avail.pkl", msg=False)['nout']
    fout = np.max(avail)
    latest.append(fout)
    print(f"    From Galaxy Extend output: {fout}")

    # 6. Check ptree
    idir = f"{snap.repo}/ptree"
    fnames = os.listdir(idir)
    fnames = [int(fname[-9:-4]) for fname in fnames if fname.startswith('ptree_0') and fname.endswith('.pkl')]
    fout = np.max(fnames)
    latest.append(fout)
    print(f"    From ptree output: {fout}")

    # 7. Check ptree_dm
    idir = f"{snap.repo}/ptree_dm"
    fnames = os.listdir(idir)
    fnames = [int(fname[-9:-4]) for fname in fnames if fname.startswith('ptree_0') and fname.endswith('.pkl')]
    fout = np.max(fnames)
    latest.append(fout)
    print(f"    From ptree_dm output: {fout}")

    refout = min(latest)
    print(f" > Auto-detected latest output (refout): {refout}")
    return refout



def get_mbh(snap, gals):
    print(" > [Get MBH] Read sink...")
    snap.read_sink()
    sinks = snap.sink_data
    if len(sinks)==0:
        print(" > [Get MBH] Fail. Read Sink_Table")
        snap.mode = 'nh'
        sinks = snap.read_sink_table()
    msol = snap.unit['Msol']

    mtable = np.zeros(np.max(sinks['id'])+1, dtype=int)-1
    mtable[sinks['id']] = np.arange(len(sinks))
    sink_ids = match_smbh(gals, sinks)
    sink_index = mtable[sink_ids]
    sink_sorted = sinks[sink_index]
    gals['MBH'] = np.where(sink_ids>=0, sink_sorted['m']/msol, 0)
    gals['dBH'] = np.where(sink_ids>=0, np.sqrt((sink_sorted['x']-gals['x'])**2 + (sink_sorted['y']-gals['y'])**2 + (sink_sorted['z']-gals['z'])**2), 0)
    return gals

def match_smbh(gals, sinks):
    argsort = np.argsort(-gals['m'])
    notyet = np.full(len(sinks), True, dtype=bool)
    gals = gals[argsort]
    sink_ids = np.zeros(len(gals), dtype=int)
    for i in tqdm(range(len(gals)), desc='Matching Sink-Galaxy'):
        igal = gals[i]
        isinks = sinks[notyet]
        sink_dist = np.sqrt((isinks['x']-igal['x'])**2 + (isinks['y']-igal['y'])**2 + (isinks['z']-igal['z'])**2)
        inside = sink_dist < igal['r']
        smbhs = isinks[inside]
        sink_dist = sink_dist[inside]

        if(len(smbhs)==0): # No SMBH
            sink_ids[i] = -1
        elif(len(smbhs)==1): # One SMBH
            iwhere = np.where(inside)[0][0]
            notyet[iwhere] = False
            sink_ids[i] = smbhs['id'][0]
        else:
            # Multiple SMBHs
            iwheres = np.where(inside)[0]
            argclose = np.argmin(sink_dist)
            if(smbhs[argclose]['m'] == np.max(smbhs['m'])):
                iwhere = iwheres[argclose]
                notyet[iwhere] = False
                sink_ids[i] = smbhs[argclose]['id']
            else:
                # calc accerelation
                accs = smbhs['m'] / sink_dist**2
                argmax = np.argmax(accs)
                iwhere = iwheres[argmax]
                notyet[iwhere] = False
                sink_ids[i] = smbhs[argmax]['id']
        if not notyet.any(): break
    argsort = np.argsort(gals['id'])
    return sink_ids[argsort]

def get_contam(hals):
    try:
        mcontam = hals['mcontam']
    except:
        mdm_min = np.min(hals['m']/hals['nparts'])
        mcontam = hals['m'] - hals['nparts']*mdm_min
    return mcontam

def match_shmr(gals, hals, Hmin, Matchlvl):
    # Remove contaminated
    mcontam = get_contam(hals)
    hals1 = hals[((mcontam/hals['m']) < 0.01) & (hals['mvir']>Hmin)]

    # Matching
    argsort = np.argsort(-hals1['mvir'])
    hals1 = hals1[argsort]
    occupied = np.full(len(gals), True, dtype=bool)
    gids = np.zeros(len(hals1), dtype=int)
    for i, hal in tqdm(enumerate(hals1), total=len(hals1), desc=" > Matching Halo-Galaxy"):
        cands = gals[occupied]
        cands = cands[cands['m'] < hal['mvir']]
        dist = np.sqrt((cands['x']-hal['x'])**2 + (cands['y']-hal['y'])**2 + (cands['z']-hal['z'])**2)
        rh = hal['rvir']
        rg = cands['r']

        _cands = cands[dist<rg]
        if len(_cands)>0:
            argmax = np.argmax(_cands['m'])
            gid = _cands['id'][argmax]
            gids[i] = gid
            occupied[gid-1] = False
        else:
            _cands = cands[dist<rh]
            if len(_cands)>0 and Matchlvl>=1:
                argmax = np.argmax(_cands['m'])
                gid = _cands['id'][argmax]
                gids[i] = gid
                occupied[gid-1] = False
            else:
                _cands = cands[dist < (rh+rg)]
                if len(_cands)>0 and Matchlvl>=2:
                    argmax = np.argmax(_cands['m'])
                    gid = _cands['id'][argmax]
                    gids[i] = gid
                    occupied[gid-1] = False
                else:
                    # Fail
                    gids[i] = -1
    return hals1, gids

def darken_color(hex_color, amount=0.3):
    rgb = mcolors.to_rgb(hex_color)
    hls = colorsys.rgb_to_hls(*rgb)
    dark_hls = (hls[0], max(0, hls[1]-amount), hls[2])
    dark_rgb = colorsys.hls_to_rgb(*dark_hls)
    return dark_rgb

def show_redshift(ax, snaps, zlist = [0.55, 0.7, 1, 1.5, 2, 3, 4, 6, 10], label='Redshift', **tick_params_dict):
    ax2 = ax.twiny()
    # top xtick as redshift
    ax2.set_xlim(ax.get_xlim())
    as_tobe_shown = 1/(1+np.array(zlist))
    ts = snaps.interpolate_iout_table(as_tobe_shown, 'aexp', 'age')
    ax2.set_xticks(ts)
    ax2.set_xticklabels(zlist)
    ax2.set_xlabel(label, fontsize=9)
    # locate top
    ax2.xaxis.set_ticks_position('top')
    ax2.tick_params(**tick_params_dict)

def add_colorbar(fig,ax,cloc,cnorm,snaps):
    cax = ax.inset_axes(cloc)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=cnorm, cmap='berlin_r'), cax=cax,
        aspect=40, pad=0.01, orientation='horizontal'
        )
    cbar.set_label('Gyr', labelpad=0, fontsize=9)
    tick_params_dict = dict(axis='x', colors='w', labelcolor='k',labelsize=9, direction='in', pad=2)
    show_redshift(cax, snaps, label='z', **tick_params_dict)
    cax.tick_params(**tick_params_dict)