from rur import uri, uhmi, painter
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.lines import Line2D

import params_nc as P
import params as P0
from func_nc import *
from func import *


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





snap = uri.RamsesSnapshot('/storage7/NewCluster', 1)
snaps = uri.TimeSeries(snap)
iout_avail = snaps.read_iout_avail()
P.ZLIST = []
if P.ADD_FINAL:
    fout = detect_auto_refout(snap, snaps=snaps)
    faexp = snaps.interpolate_iout_table(fout, 'iout', 'aexp')
    fz = float(1/faexp - 1)
    P.ZLIST.append(fz)
    P.MULTI = False
    P.SHOW_SCATTER = not P.MULTI
print("Draw for z-list:")
print(P.ZLIST)
lenZ = len(P.ZLIST)
cmap = plt.get_cmap(P.CMAP)
if P.MULTI: colors = [cmap(i/(lenZ-1)) for i in range(lenZ)]
fig, axes = plt.subplots(3,4, figsize=(16*1.2, 12), dpi=150)
'''
Massfunc    SHMR    CMD     SB          (halo and photometry)
SFMS        Size    Metal   MBH         (Galactic)
Cold        O/H     aFe/FeH DTM         (Gas)
'''
leg_elements = [Line2D([0], [0], linestyle='none', marker=r'$\mathbf{NC }$', mec='none', mfc=P.COLOR1, markersize=12, label='')]
tax = [[0,0,'txt','top','left'] for i in range(12)]
maxout = 0
for ith, iz in enumerate(P.ZLIST):
    iaexp = 1/(1+iz)
    snap = snaps.get_from_aexp(iaexp)
    maxout = max(maxout, snap.iout)
    if P.MULTI:
        icolor1 = colors[ith]
        P.COLOR1 = icolor1
        P.COLOR2 = icolor1
        P.COLOR3 = icolor1
    else:
        icolor1 = P.COLOR1
    label = f"z={iz:.2f}" if (iz%1!=0) else f"z={int(iz)}"
    leg_elements.append(Line2D([0],[0], lw=2, ls='-', color=icolor1, label=label))
    
    gals = uhmi.HaloMaker.load(snap, galaxy=True, extend=True)
    hals = uhmi.HaloMaker.load(snap, galaxy=False, extend=False)
    mcontam = get_contam(hals)
    hmask = (hals['level']<=P.LEVELCUT)&(mcontam/hals['m'] < 0.01)
    hals1 = hals[hmask]

    gmask = (gals['level']<=P.LEVELCUT)|((gals['nparts']>P.NPARTSCUT)&(gals['age']>P.AGECUT))
    gals1 = gals[gmask]
    gals2 = gals[~gmask]
    kwargs = dict(gals1=gals1, gals2=gals2, snap=snap)

    ax = axes[0,0]
    draw_massfunc(ax, gals1, hals1)
    tax[0][0] = 0.98; tax[0][1] = 0.98; tax[0][2] = "Mass Functions"; tax[0][3] = 'top'; tax[0][4] = 'right'

    ax = axes[0,1]
    pair_hals, pair_gids = match_shmr(gals, hals, P.DETAIL['SHMR']['Hmin'], P.DETAIL['SHMR']['Matchlvl'])
    draw_shmr(ax, pair_hals, pair_gids, gals=gals)
    tax[1][0] = 0.98; tax[1][1] = 0.02; tax[1][2] = "SHMR"; tax[1][3] = 'bottom'; tax[1][4] = 'right'

    ax = axes[0,2]
    draw_CMD(ax, **kwargs)
    tax[2][0] = 0.98; tax[2][1] = 0.02; tax[2][2] = "CMD"; tax[2][3] = 'bottom'; tax[2][4] = 'right'

    ax = axes[0,3]
    draw_SB(ax, **kwargs)
    tax[3][0] = 0.98; tax[3][1] = 0.02; tax[3][2] = "Surface Brightness"; tax[3][3] = 'bottom'; tax[3][4] = 'right'

    ax = axes[1,0]
    draw_sfms(ax, **kwargs)
    tax[4][0] = 0.98; tax[4][1] = 0.02; tax[4][2] = "SFMS"; tax[4][3] = 'bottom'; tax[4][4] = 'right'

    ax = axes[1,1]
    draw_size(ax, **kwargs)
    tax[5][0] = 0.98; tax[5][1] = 0.02; tax[5][2] = "Size-Mass"; tax[5][3] = 'bottom'; tax[5][4] = 'right'

    ax = axes[1,2]
    draw_metal(ax, **kwargs)
    tax[6][0] = 0.98; tax[6][1] = 0.02; tax[6][2] = "Metallicity"; tax[6][3] = 'bottom'; tax[6][4] = 'right'

    ax = axes[1,3]
    draw_mbh(ax, **kwargs)
    tax[7][0] = 0.02; tax[7][1] = 0.98; tax[7][2] = "MBH"; tax[7][3] = 'top'; tax[7][4] = 'left'

    ax = axes[2,0]
    draw_cold(ax, **kwargs)
    tax[8][0] = 0.02; tax[8][1] = 0.02; tax[8][2] = "Cold Gas"; tax[8][3] = 'bottom'; tax[8][4] = 'left'

    ax = axes[2,1]
    draw_metal(ax, key='OH', **kwargs)
    tax[9][0] = 0.98; tax[9][1] = 0.02; tax[9][2] = "O/H"; tax[9][3] = 'bottom'; tax[9][4] = 'right'

    ax = axes[2,2]
    draw_alpha(ax, **kwargs)
    tax[10][0] = 0.02; tax[10][1] = 0.02; tax[10][2] = r"[$\alpha$/Fe]-[Fe/H]"; tax[10][3] = 'bottom'; tax[10][4] = 'left'

    ax = axes[2,3]
    draw_DTM(ax, **kwargs)
    tax[11][0] = 0.98; tax[11][1] = 0.02; tax[11][2] = "DTM"; tax[11][3] = 'bottom'; tax[11][4] = 'right'

    snap.clear()


leg_ncols = np.ceil(len(P.ZLIST)/4).astype(int)
for i in range(12):
    ax = axes.flatten()[i]
    # ax.text(tax[i][0], tax[i][1], tax[i][2], transform=ax.transAxes, va=tax[i][3], ha=tax[i][4], weight='bold', zorder=10)
    loc1 = 'upper' if tax[i][3]=='top' else 'lower'
    loc2 = 'left' if tax[i][4]=='left' else 'right'
    loc = f"{loc1} {loc2}"
    anchorX = 0.02 if tax[i][4]=='left' else 0.98
    anchorY = 0.98 if tax[i][3]=='top' else 0.02
    bbox_to_anchor = (anchorX, anchorY)
    ax.legend(
        handles=leg_elements, loc=loc, bbox_to_anchor=bbox_to_anchor,
        fontsize=8, frameon=False,
        title = tax[i][2], title_fontproperties=font_manager.FontProperties(weight='bold', size=11),
        labelspacing=0.3, handletextpad=1.2, borderpad=0.0, borderaxespad=0.0,
        ncol=leg_ncols, columnspacing=0.5)

if P.MULTI:
    plt.savefig(f'{P0.FIGOUTDIR}/fig_NC_evol_at_{maxout}.png', dpi=300, facecolor='w', bbox_inches='tight', pad_inches=0.10, transparent=False)
    print(f" >  Saved `{P0.FIGOUTDIR}/fig_NC_evol_at_{maxout}.png`")
else:
    plt.savefig(f'{P0.FIGOUTDIR}/fig_NC_last_at_{maxout}.png', dpi=300, facecolor='w', bbox_inches='tight', pad_inches=0.10, transparent=False)
    print(f" >  Saved `{P0.FIGOUTDIR}/fig_NC_last_at_{maxout}.png`")