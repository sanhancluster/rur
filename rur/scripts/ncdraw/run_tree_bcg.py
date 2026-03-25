import datetime
now = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
print(now)
from rur import uri, uhmi, painter, drawer
from rur.utool import dump, load
from rur.utool import datload
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from multiprocessing import Pool, shared_memory
from time import gmtime, strftime

from importlib import reload
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import colorsys
import os

from func import *
import params as P0
import params_tree as P
from func_tree import _get_val, _set_shm, _extend, _get_shm

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


print("\n > Load Data...")
uri.timer.verbose=0
snap = uri.RamsesSnapshot('/storage7/NewCluster', 1)
cMpc = snap.unit['cMpc']
snaps = uri.TimeSeries(snap)
max_out = detect_auto_refout(snap, snaps)
iout_avail = snaps.read_iout_avail()
ptree = load('/storage7/NewCluster/ptree/ptree_stable.pkl')
ptree_dm = load('/storage7/NewCluster/ptree_dm/ptree_stable.pkl')
# max_out = min(np.max(ptree['timestep']), np.max(ptree_dm['timestep']))
print(f" > Max iout = {max_out}")
max_snap = snaps.get_snap(max_out)



refiout = P.REFOUT
if refiout is None:
    refiout = max_out
NC = snaps.get_snap(refiout)
NCgals = uhmi.HaloMaker.load(NC, galaxy=True, extend=True)
NChals = uhmi.HaloMaker.load(NC, galaxy=False, extend=True)
mcontam = get_contam(NChals)
NCgroups = NChals[(mcontam/NChals['m']) < 0.01]
NCgroups = NCgroups[NCgroups['mvir'] > P.MAXMASS_CUT/10]
NCgroups, gids = match_shmr(NCgals, NCgroups, 0, 1)
mask = gids>0
NCgroups = NCgroups[mask]; gids = gids[mask]
print(f" > {len(NCgroups)} halos will be considered...")
NCBCGs = NCgals[gids-1]
primaryh = NCgroups[np.argmax(NCgroups['mvir'])]
primaryg = NCBCGs[np.argmax(NCgroups['mvir'])]


print(f" > Sorting trees...")
itree_dm = ptree_dm[ptree_dm['timestep'] == refiout]
argsort = np.argsort(itree_dm['hmid'])
itree_dm = itree_dm[argsort]
itree = ptree[ptree['timestep'] == refiout]
argsort = np.argsort(itree['hmid'])
itree = itree[argsort]

data = {
    'htree':{},
    'gtree':{},
    'color':{},
    'dcolor':{}
}

fig, ax = plt.subplots(figsize=(4,3))
mmask = np.full(len(NCgroups), True, dtype=bool)
for i in tqdm(range(len(NCgroups)), desc=' > Check maximum Mvir'):
    BCH = NCgroups[i]
    BCG = NCBCGs[i]
    iBCH = itree_dm[BCH['id']-1]
    iBCG = itree[BCG['id']-1]
    htree = ptree_dm[ptree_dm['last'] == iBCH['last']]
    if np.max(htree['mvir']) < P.MAXMASS_CUT:
        mmask[i] = False; continue
    argsort = np.argsort(htree['timestep']); htree = htree[argsort]
    gtree = ptree[ptree['last'] == iBCG['last']]
    argsort = np.argsort(gtree['timestep']); gtree = gtree[argsort]
    hmask = np.isin(htree['timestep'], gtree['timestep'])
    htree = htree[hmask]
    gmask = np.isin(gtree['timestep'], htree['timestep'])
    gtree = gtree[gmask]
    data['htree'][BCH['id']] = htree
    data['gtree'][BCG['id']] = gtree

    pl = ax.plot(htree['timestep'], htree['mvir'], lw=1)
    color = pl[0].get_color()
    data['color'][BCH['id']] = color
    data['dcolor'][BCH['id']] = darken_color(color, 0.2)
NCgroups = NCgroups[mmask]
NCBCGs = NCBCGs[mmask]
print(f" > {len(NCgroups)} halos are selected")
plt.close()



# --------------------------
# Main
# --------------------------

print(f" > Draw Mass...")
fig, axes = plt.subplots(3,4, figsize=(16*1.3, 12), dpi=150)
'''
Mass&Dist   SHMR    CMD     SB          (halo and photometry)
SFMS        Size    Metal   MBH         (Galactic)
Cold        O/H     aFe/FeH DTM         (Gas)
'''

# For Massive Halos
_ax = axes[0,0]; _ax.set_facecolor('none'); _ax.axis('off')
ax = _ax.inset_axes([0,0,0.9,1]); ax.set_facecolor('none')
ax2 = ax.twinx()
ax2.set_zorder(-1)
leg_elements = [
    Line2D([0], [0], linestyle='-', lw=2, label='Primary Halo Mass', color=data['color'][primaryh['id']]),
    Line2D([0], [0], linestyle='--', lw=2, label='BCG Stellar Mass', color=data['dcolor'][primaryh['id']]),
    Patch(facecolor=data['color'][primaryh['id']], edgecolor='none', alpha=0.5, label=r'$R_{\rm vir}$'),
    ]

detail = P.DETAIL['Dist']
yscale, ymin, ymax, ylabel = detail['yscale'], detail['ymin'], detail['ymax'], detail['ylabel']
for i in range(len(NCgroups)):
    BCH = NCgroups[i]; BCG = NCBCGs[i]; htree = data['htree'][BCH['id']]; gtree = data['gtree'][BCG['id']]; color = data['color'][BCH['id']]
    hgyr = snaps.interpolate_iout_table(htree['timestep'], 'iout', 'age')
    if BCH['id'] == primaryh['id']:
        ax2.fill_between(hgyr, 0, htree['rvir']/cMpc*htree['aexp'], ec='none', fc=color, alpha=0.5, zorder=-1)
    else:
        pritree = data['htree'][primaryh['id']]
        mask = np.isin(pritree['timestep'], htree['timestep']); pritree = pritree[mask]
        mask = np.isin(htree['timestep'], pritree['timestep']); htree = htree[mask]
        dist = np.sqrt((htree['x']-pritree['x'])**2 + (htree['y']-pritree['y'])**2 + (htree['z']-pritree['z'])**2)/cMpc*htree['aexp']
        ax2.fill_between(hgyr, dist-htree['rvir']/cMpc*htree['aexp'], dist+htree['rvir']/cMpc*htree['aexp'], ec='none', fc=color, alpha=0.3, zorder=-1)
        # ax2.plot(hgyr, dist, lw=1, color=color, ls='-')
ax2.set_yscale(yscale)
ax2.set_ylim(ymin, ymax)
ax2.set_ylabel(ylabel)

detail = P.DETAIL['Mass']
xscale, xlabel, xmin, xmax = detail['xscale'], detail['xlabel'], detail['xmin'], detail['xmax']
yscale, ymin, ymax, ylabel = detail['yscale'], detail['ymin'], detail['ymax'], detail['ylabel']
tlabel, tloc = detail['tlabel'], detail['tloc']
for i in range(len(NCgroups)):
    BCH = NCgroups[i]; BCG = NCBCGs[i]; htree = data['htree'][BCH['id']]; gtree = data['gtree'][BCG['id']]; color = data['color'][BCH['id']]
    hgyr = snaps.interpolate_iout_table(htree['timestep'], 'iout', 'age')
    ggyr = snaps.interpolate_iout_table(gtree['timestep'], 'iout', 'age')

    lw = 2 if BCH['id']==primaryh['id'] else 1
    zorder = 2 if BCH['id']==primaryh['id'] else 1
    pl = ax.plot(hgyr, htree['mvir'], lw=lw, color=color, zorder=zorder)
    dcolor = data['dcolor'][BCH['id']]
    ax.plot(ggyr, gtree['m'], lw=lw, color=dcolor, ls='--', zorder=zorder)

ax.legend(handles=leg_elements, loc='lower center', fontsize=9, frameon=False)

ax.set_xscale(xscale)
ax.set_xlim(0, max_snap.age)
ax.set_xlabel(xlabel)
ax.set_yscale(yscale)
ax.set_ylabel(ylabel)
ax.text(tloc[0], tloc[2], tlabel, transform=ax.transAxes, ha=tloc[1], va=tloc[3], color='k', weight='bold')
show_redshift(ax, snaps)


# ---------------------------------------------
# From here, BCG&BCH only
# ---------------------------------------------
draw_gids = [primaryg['id']]
draw_hids = [primaryh['id']]
subfix = 'BCG'
draw_colorline = True


firstdraw = True
for draw_gid, draw_hid in zip(draw_gids, draw_hids):
    gtree = data['gtree'][draw_gid]
    htree = data['htree'][draw_hid]
    zorder = 2 if draw_hid==primaryh['id'] else 1

    print(f"   Merge extended values to tree...")
    gextend_path = f"{snap.repo}/galaxy/extended"
    pkeys = ['SFR', 'r50', 'gmag', 'rmag', 'SBr_r50', 'metal', 'MBH', 'dBH']
    gkeys = ['Mcold_gas_r90', 'metal_gas']
    ckeys = ['O_gas', 'H_gas', 'Mg_gas', 'Si_gas', 'Fe_gas', 'CDustLarge_gas','CDustSmall_gas','SiDustLarge_gas','SiDustSmall_gas']
    keys = pkeys + gkeys + ckeys
    types = ['part']*len(pkeys) + ['gas']*len(gkeys) + ['chem']*len(ckeys)
    dtype = [(key, 'f8') for key in keys]


    memory, extended = _set_shm(len(gtree), dtype)
    pbar = tqdm(total=len(gtree), desc='   Extend')
    def update(*a): pbar.update()
    with Pool(processes=24) as pool:
        for ith, ig in enumerate(gtree):
            pool.apply_async(_extend, args=(ith, ig, memory.name, extended.shape, extended.dtype, gextend_path, keys, types), callback=update)
        pool.close()
        pool.join()
    pbar.close()

    if draw_colorline:
        redshifts = 1/gtree['aexp']-1
        cnorm_by_z = plt.Normalize(redshifts.min(), redshifts.max())
        gyrs = snaps.interpolate_iout_table(gtree['timestep'], 'iout', 'age')
        cnorm_by_gyr = plt.Normalize(gyrs.min(), gyrs.max())

    axkeys = ['SHMR', 'CMD', 'SB', 'SFMS', 'Size', 'Metal', 'MBH', 'Cold', 'O/H', 'alpha', 'DTM']
    for j in range(11):
        print(f" > Draw {axkeys[j]}...")
        ax = axes.flatten()[j+1]
        ax.set_facecolor('none')
        axkey = axkeys[j]
        xval, yval = _get_val(snap, gtree, htree, extended, axkey)
        detail = P.DETAIL[axkey]
        xscale, xlabel, xmin, xmax, invertx = detail['xscale'], detail['xlabel'], detail['xmin'], detail['xmax'], detail['invertx']
        yscale, ylabel, ymin, ymax, inverty = detail['yscale'], detail['ylabel'], detail['ymin'], detail['ymax'], detail['inverty']
        tlabel, tloc = detail['tlabel'], detail['tloc']
        cloc = detail['cloc']

        ax.scatter(xval, yval, s=0)
        if draw_colorline:
            drawer.colorline(
                xval, yval, 
                z=gyrs ,cmap=P.CMAP, norm=cnorm_by_gyr,
                linewidth=2, ax=ax, zorder=zorder)
        else:
            ax.plot(xval, yval, color=data['color'][draw_hid], lw=2, zorder=zorder)
        if firstdraw:
            ax.set_xscale(xscale)
            if xmin is not None or xmax is not None: ax.set_xlim(xmin, xmax)
            if invertx: ax.invert_xaxis()
            ax.set_xlabel(xlabel)
            if ymin is not None or ymax is not None: ax.set_ylim(ymin, ymax)
            if inverty: ax.invert_yaxis()
            ax.set_yscale(yscale)
            ax.set_ylabel(ylabel, color='k')
            ax.text(tloc[0], tloc[2], tlabel, transform=ax.transAxes, ha=tloc[1], va=tloc[3], color='k', weight='bold', fontsize=11)

            if draw_colorline: add_colorbar(fig,ax,cloc,cnorm_by_gyr,snaps)

    memory.close()
    memory.unlink()
    firstdraw = False
plt.subplots_adjust(wspace=0.2)
plt.savefig(f'{P0.FIGOUTDIR}/fig_tree_{subfix}_at_{max_out}.png', dpi=300, facecolor='w', bbox_inches='tight', pad_inches=0.10, transparent=False)
print(f" > Saved `{P0.FIGOUTDIR}/fig_tree_{subfix}_at_{max_out}.png`")
plt.close()