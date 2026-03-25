import filecmp
import itertools
import time
import datetime
now = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
print(now)
from rur import uri, uhmi, painter, drawer
from rur.config import sink_prop_dtype_drag_y2
from rur.utool import datload, load, dump
from rur.sci import smbh
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from multiprocessing import Pool, shared_memory
from time import gmtime, strftime
from importlib import reload
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import colorsys
import os, sys


from func import *
import params as P0
import params_smbh as P
from func_tree import _set_shm, _extend, _get_shm

# Font find
try:
    # Load Font
    fname = P0.FONT
    fontprop = font_manager.FontProperties(fname=fname)

    # Apply to normal text
    mpl.rcParams['font.family'] = fontprop.get_name()

    # Apply to math text
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = fontprop.get_name()
    mpl.rcParams['mathtext.it'] = fontprop.get_name() + ':italic'
    mpl.rcParams['mathtext.bf'] = fontprop.get_name() + ':bold'

    # Override generic families to prevent warnings
    for key in ['sans-serif', 'serif', 'cursive', 'monospace', 'fantasy']:
        mpl.rcParams[f'font.{key}'] = [fontprop.get_name()]
except Exception as e:
    print(f"Error loading font: {e}")
    print("Falling back to default font.")





#------------------------------
# Load Data
#------------------------------
print(f" > Load Data...")
# Snapshot
if P.REFOUT is None:
    snap = uri.RamsesSnapshot('/storage7/NewCluster', 1, mode='nc')
    P.REFOUT = detect_auto_refout(snap)
snap = uri.RamsesSnapshot('/storage7/NewCluster', P.REFOUT, mode='nc')
snaps = uri.TimeSeries(snap)
snaps.read_icoarse_avail()
snaps.read_iout_avail()

# Sink Timeline
sink_timeline = snap.read_sinkprops(use_cache=True)
unis, inds, counts = np.unique(sink_timeline['id'], return_index=True, return_counts=True)
# Sink snapshot
snap.read_sink()
smbhs = snap.sink_data

# Catalog
gals = uhmi.HaloMaker.load(snap, galaxy=True)
hals = uhmi.HaloMaker.load(snap, galaxy=False)
hals = hals[get_contam(hals)/hals['m'] < P.FCONTAM]
hals = hals[hals['level'] <= P.HLVL]

# Tree
ptree = load('/storage7/NewCluster/ptree/ptree_stable.pkl')
ptree_dm = load('/storage7/NewCluster/ptree_dm/ptree_stable.pkl')
itree_dm = ptree_dm[ptree_dm['timestep'] == snap.iout]
argsort = np.argsort(itree_dm['hmid'])
itree_dm = itree_dm[argsort]
itree = ptree[ptree['timestep'] == snap.iout]
argsort = np.argsort(itree['hmid'])
itree = itree[argsort]
def get_tree(gal, galaxy=True):
    global itree, itree_dm, ptree, ptree_dm
    iitree = itree if galaxy else itree_dm
    iptree = ptree if galaxy else ptree_dm
    igal = iitree[gal['id']-1]
    mytree = iptree[iptree['last'] == igal['last']]
    argsort = np.argsort(mytree['timestep'])
    return mytree[argsort]



#------------------------------
# Match Halo and Galaxy
#------------------------------
print(f" > Match Halo and Galaxy...")
mhals, mgids = match_shmr(gals, hals, P.HMIN, P.MATCHLVL)
mask = mgids>0
mhals, mgids = mhals[mask], mgids[mask]
mgals = gals[mgids-1]
mask = mgals['level'] <= P.GLVL
mhals, mgals = mhals[mask], mgals[mask]

#------------------------------
# Match Galaxy and Sink using Tree
#------------------------------
print(f" > Match Galaxy and Sink...")
sink_ids = np.zeros(len(mgals), dtype=int)-1
for i, gal in tqdm(enumerate(mgals), total=len(mgals)):
    myhal = mhals[i]
    dist = np.sqrt((smbhs['x']-myhal['x'])**2 + (smbhs['y']-myhal['y'])**2 + (smbhs['z']-myhal['z'])**2)
    cands = np.where(dist < myhal['rvir'])[0]
    if len(cands) == 0: continue
    gtree = get_tree(gal, galaxy=True)
    gcoarse = snaps.interpolate_iout_table(gtree['timestep'], 'iout', 'icoarse')

    cand_ids = smbhs['id'][cands]
    isin = np.isin(unis, cand_ids)
    _unis, _inds, _counts = unis[isin], inds[isin], counts[isin]
    dist = np.inf
    cand_id = -1
    for _uni, _ind, _count in zip(_unis, _inds, _counts):
        _sink = sink_timeline[_ind:_ind+_count]
        mask = np.isin(_sink['icoarse'], gcoarse)
        _sink = _sink[mask]
        mask = np.isin(gcoarse, _sink['icoarse'])
        gtree_masked = gtree[mask]
        _dist = np.mean(np.sqrt(( _sink['x']-gtree_masked['x'])**2 + ( _sink['y']-gtree_masked['y'])**2 + ( _sink['z']-gtree_masked['z'])**2))
        if _dist < dist:
            dist = _dist
            cand_id = _uni
    sink_ids[i] = cand_id

mask = sink_ids > 0
print(f" > {mask.sum()} / {len(mgals)} galaxies matched with SMBHs")
mhals, mgals, sink_ids = mhals[mask], mgals[mask], sink_ids[mask]
sorter = np.argsort(sink_ids)
isin = np.isin(smbhs['id'], sink_ids)
indices_in_sink = sorter[np.searchsorted(sink_ids, smbhs['id'][isin], sorter=sorter)]
sort_idx = np.argsort(indices_in_sink)
msmbhs = smbhs[isin][sort_idx]
mtable = np.zeros(unis.max()+1, dtype=int)-1
mtable[msmbhs['id']] = np.arange(len(msmbhs))

need_extend = (P.GRADIUS not in gtree.dtype.names)
if need_extend:
    gextend_path = f"{snap.repo}/galaxy/extended"
    pkeys = [P.GRADIUS]
    gkeys = []
    ckeys = []
    keys = pkeys + gkeys + ckeys
    types = ['part']*len(pkeys) + ['gas']*len(gkeys) + ['chem']*len(ckeys)
    dtype = [(key, 'f8') for key in keys]

for uni, ind, count in zip(unis, inds, counts):
    if mtable[uni]<0: continue
    print(f" > Process SMBH ID={uni}...")
    hmind = mtable[uni]
    mhal, mgal = mhals[hmind], mgals[hmind]
    isink_timeline = sink_timeline[ind:ind+count]
    htree = get_tree(mhal, galaxy=False)
    gtree = get_tree(mgal, galaxy=True)
    
    if need_extend:
        # --------------------------
        # Extend Gtree
        # --------------------------
        memory, extended = _set_shm(len(gtree), dtype)
        pbar = tqdm(total=len(gtree), desc='   Extend')
        def update(*a): pbar.update()
        with Pool(processes=24) as pool:
            for ith, ig in enumerate(gtree):
                pool.apply_async(_extend, args=(ith, ig, memory.name, extended.shape, extended.dtype, gextend_path, keys, types), callback=update)
            pool.close()
            pool.join()
        pbar.close()
    else:
        extended = gtree

    fig = plt.figure(figsize=(15,5), dpi=300)
    outer_gs = gridspec.GridSpec(1, 2, wspace=0.2, width_ratios=[1,3])
    gs1 = outer_gs[0].subgridspec(1,1)
    # ---------------------------
    # Draw mass evolution
    # ---------------------------
    ax0 = fig.add_subplot(gs1[0,0])
    ax0.set_facecolor('none')
    labels = ['Mass', 
              r'V$_{\rm rel}$', 'Density', 'Accretion Rate', 'Eddington Ratio',
              'Spin', 'Efficiency', 'Feedback', 'Total Released E']
    axes = [ax0]
    gs2 = outer_gs[1].subgridspec(4,2, wspace=0.6)
    for icol in range(2):
        for irow in range(4):
            axes.append(fig.add_subplot(gs2[irow,icol]))
    xgyr = snaps.interpolate_icoarse_table(isink_timeline['icoarse'], 'icoarse', 'age')
    ax0.plot(snaps.interpolate_iout_table(htree['timestep'], 'iout', 'age'), np.log10(htree['mvir']), label='M$_{\\rm Halo}$', color='grey')
    ax0.plot(snaps.interpolate_iout_table(gtree['timestep'], 'iout', 'age'), np.log10(gtree['m']), label='M$_{\\rm *}$', color='salmon')
    
    # ---------------------------
    # Draw others
    # ---------------------------
    smbh.draw_sink_timeline(snap, isink_timeline, show_macc=False, axes=axes, xmode='Age [Gyr]', xarr=xgyr, plot_params=dict(color='k'))

    # ---------------------------
    # Draw track
    # ---------------------------
    ax = ax0.inset_axes((0.5, 0.02, 0.48, 0.48))
    ax.set_zorder(-1)
    ax.plot(isink_timeline['x'], isink_timeline['y'], color='k', zorder=5, lw=0.5)
    for ihal in htree:
        cir = plt.Circle((ihal['x'], ihal['y']), ihal['rvir'], ec='none', fc='grey', fill=True, zorder=0)
        ax.add_patch(cir)
    for igal, iradius in zip(gtree, extended[P.GRADIUS]):
        cir = plt.Circle((igal['x'], igal['y']), P.GRADII*iradius, ec='none', fc='salmon', fill=True, zorder=1)
        ax.add_patch(cir)
    ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax = axes[0]
    leg = ax.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((0.05, 0.85))
        leg.set_loc('upper left')

    ax = axes[-1]
    ax.set_ylabel("Etot\n(10$^{61}$ erg)")

    for ax in [axes[0], axes[1], axes[5]]:
        show_redshift(ax, snaps, label='z')

    for i, ax in enumerate(axes):
        ax.text(0.05, 0.95, labels[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    if need_extend:
        memory.close()
        memory.unlink()

    fig.savefig(f"{P0.FIGOUTDIR}/fig_smbh_{uni:05d}_at_{P.REFOUT}.png", dpi=300, facecolor='w', bbox_inches='tight', pad_inches=0.10, transparent=False)
    print(f" > Saved `{P0.FIGOUTDIR}/fig_smbh_{uni:05d}_at_{P.REFOUT}.png`")
    plt.close()