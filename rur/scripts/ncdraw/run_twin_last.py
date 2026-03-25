from rur import uri, uhmi, painter
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.lines import Line2D

import params_twin as P
import params as P0
from func_twin import *
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



NC = uri.RamsesSnapshot('/storage7/NewCluster', 1)
NCs = uri.TimeSeries(NC)
NH = uri.RamsesSnapshot('/storage5/NH2', 1)
NHs = uri.TimeSeries(NH)
P.ZLIST = []
if P.ADD_FINAL:
    fout = detect_auto_refout(NC, NCs)
    faexp = NCs.interpolate_iout_table(fout, 'iout', 'aexp')
    fz = float(1/faexp - 1)
    P.ZLIST.append(fz)
    P.MULTI = False
    P.SHOW_SCATTER = not P.MULTI
print("Draw for z-list:")
print(P.ZLIST)
lenZ = len(P.ZLIST)
cmap1 = P.CMAP_NC
cmap2 = P.CMAP_NH
if P.MULTI:
    colors1 = [cmap1(i/(lenZ-1)) for i in range(lenZ)]
    colors2 = [cmap2(i/(lenZ-1)) for i in range(lenZ)]
fig, axes = plt.subplots(3,4, figsize=(16*1.2, 12), dpi=150)
'''
Massfunc    SHMR    CMD     SB          (halo and photometry)
SFMS        Size    Metal   MBH         (Galactic)
Cold        O/H     aFe/FeH N/A         (Gas)
'''
NCleg_elements = [Line2D([0], [0], linestyle='none', marker=r'$\mathbf{NC }$', mec='none', mfc=P.COLOR1A, markersize=12, label='')]
NHleg_elements = [Line2D([0], [0], linestyle='none', marker=r'$\mathbf{NH2}$', mec='none', mfc=P.COLOR2A, markersize=17, label='')]
# NHleg_elements = [Line2D([0],[0], marker='.',color='none', markeredgecolor='dodgerblue', markerfacecolor='none', markersize=0, label=r'$\mathbf{NH2}$')]
tax = [[0,0,'txt','top','left'] for i in range(12)]
maxout = 0
for ith, iz in enumerate(P.ZLIST):
    iaexp = 1/(1+iz)
    NC = NCs.get_from_aexp(iaexp)
    maxout = max(maxout, NC.iout)
    NH = NHs.get_from_aexp(iaexp)
    if P.MULTI:
        icolor1 = colors1[ith]
        P.COLOR1A = icolor1
        P.COLOR1B = icolor1
        P.COLOR1C = icolor1
        icolor2 = colors2[ith]
        P.COLOR2A = icolor2
        P.COLOR2B = icolor2
        P.COLOR2C = icolor2
    else:
        icolor1 = P.COLOR1A
        icolor2 = P.COLOR2A
    label = f"z={iz:.2f}" if (iz%1!=0) else f"z={int(iz)}"
    NCleg_elements.append(Line2D([0],[0], lw=2, ls='-', color=icolor1))
    NHleg_elements.append(Line2D([0],[0], lw=2, ls='-', color=icolor2, label=label))
    
    NCgals = uhmi.HaloMaker.load(NC, galaxy=True, extend=True)
    NChals = uhmi.HaloMaker.load(NC, galaxy=False, extend=False)
    mcontam = get_contam(NChals)
    hmask = (NChals['level']<=P.LEVELCUT)&(mcontam/NChals['m'] < 0.01)
    NChals1 = NChals[hmask]
    gmask = (NCgals['level']<=P.LEVELCUT)|((NCgals['nparts']>P.NPARTSCUT)&(NCgals['age']>P.AGECUT))
    NCgals1 = NCgals[gmask]
    NCgals2 = NCgals[~gmask]
    NCkwargs = dict(gals1=NCgals1, gals2=NCgals2, snap=NC, format_ax=False, colors=(P.COLOR1A, P.COLOR1B, P.COLOR1C), emarker="$NC$")

    NHgals = uhmi.HaloMaker.load(NH, galaxy=True, extend=True)
    NHhals = uhmi.HaloMaker.load(NH, galaxy=False, extend=False)
    mcontam = get_contam(NHhals)
    hmask = (NHhals['level']<=P.LEVELCUT)&(mcontam/NHhals['m'] < 0.01)
    NHhals1 = NHhals[hmask]
    gmask = (NHgals['level']<=P.LEVELCUT)|((NHgals['nparts']>P.NPARTSCUT)&(NHgals['age']>P.AGECUT))
    NHgals1 = NHgals[gmask]
    NHgals2 = NHgals[~gmask]
    NHkwargs = dict(gals1=NHgals1, gals2=NHgals2, snap=NH, format_ax=True, colors=(P.COLOR2A, P.COLOR2B, P.COLOR2C), emarker="$NH$")

    ax = axes[0,0]
    draw_massfunc(ax, NCgals1, NChals1, 23300, **NCkwargs)
    draw_massfunc(ax, NHgals1, NHhals1, 4/3*np.pi*(10/NH.h)**3, **NHkwargs)
    tax[0][0] = 0.98; tax[0][1] = 0.98; tax[0][2] = "Mass Functions"; tax[0][3] = 'top'; tax[0][4] = 'right'

    ax = axes[0,1]
    pair_hals, pair_gids = match_shmr(NCgals, NChals, P.DETAIL['SHMR']['Hmin'], P.DETAIL['SHMR']['Matchlvl'])
    draw_shmr(ax, pair_hals, pair_gids, gals=NCgals, **NCkwargs)
    pair_hals, pair_gids = match_shmr(NHgals, NHhals, P.DETAIL['SHMR']['Hmin'], P.DETAIL['SHMR']['Matchlvl'])
    draw_shmr(ax, pair_hals, pair_gids, gals=NHgals, **NHkwargs)
    tax[1][0] = 0.98; tax[1][1] = 0.02; tax[1][2] = "SHMR"; tax[1][3] = 'bottom'; tax[1][4] = 'right'

    ax = axes[0,2]
    draw_CMD(ax, **NCkwargs)
    draw_CMD(ax, **NHkwargs)
    tax[2][0] = 0.98; tax[2][1] = 0.02; tax[2][2] = "CMD"; tax[2][3] = 'bottom'; tax[2][4] = 'right'

    ax = axes[0,3]
    draw_SB(ax, **NCkwargs)
    draw_SB(ax, **NHkwargs)
    tax[3][0] = 0.98; tax[3][1] = 0.02; tax[3][2] = "Surface Brightness"; tax[3][3] = 'bottom'; tax[3][4] = 'right'

    ax = axes[1,0]
    draw_sfms(ax, **NCkwargs)
    draw_sfms(ax, **NHkwargs)
    tax[4][0] = 0.98; tax[4][1] = 0.02; tax[4][2] = "SFMS"; tax[4][3] = 'bottom'; tax[4][4] = 'right'

    ax = axes[1,1]
    draw_size(ax, **NCkwargs)
    draw_size(ax, **NHkwargs)
    tax[5][0] = 0.98; tax[5][1] = 0.02; tax[5][2] = "Size-Mass"; tax[5][3] = 'bottom'; tax[5][4] = 'right'

    ax = axes[1,2]
    draw_metal(ax, **NCkwargs)
    draw_metal(ax, **NHkwargs)
    tax[6][0] = 0.98; tax[6][1] = 0.02; tax[6][2] = "Metallicity"; tax[6][3] = 'bottom'; tax[6][4] = 'right'

    ax = axes[1,3]
    draw_mbh(ax, **NCkwargs)
    draw_mbh(ax, **NHkwargs)
    tax[7][0] = 0.02; tax[7][1] = 0.98; tax[7][2] = "MBH"; tax[7][3] = 'top'; tax[7][4] = 'left'

    ax = axes[2,0]
    draw_cold(ax, **NCkwargs)
    draw_cold(ax, **NHkwargs)
    tax[8][0] = 0.02; tax[8][1] = 0.02; tax[8][2] = "Cold Gas"; tax[8][3] = 'bottom'; tax[8][4] = 'left'

    ax = axes[2,1]
    draw_metal(ax, key='OH', **NCkwargs)
    draw_metal(ax, key='OH', **NHkwargs)
    tax[9][0] = 0.98; tax[9][1] = 0.02; tax[9][2] = "O/H"; tax[9][3] = 'bottom'; tax[9][4] = 'right'

    ax = axes[2,2]
    draw_alpha(ax, **NCkwargs)
    draw_alpha(ax, **NHkwargs)
    tax[10][0] = 0.02; tax[10][1] = 0.02; tax[10][2] = r"[$\alpha$/Fe]-[Fe/H]"; tax[10][3] = 'bottom'; tax[10][4] = 'left'

    ax = axes[2,3]
    # draw_DTM(ax, **NCkwargs)
    # draw_DTM(ax, **NHkwargs)
    # tax[11][0] = 0.98; tax[11][1] = 0.02; tax[11][2] = "DTM"; tax[11][3] = 'bottom'; tax[11][4] = 'right'

    NC.clear()
    NH.clear()


# leg_ncols = np.ceil(len(P.ZLIST)/4).astype(int)
leg_elements = NCleg_elements+NHleg_elements
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
        ncols=2, columnspacing=1
        )

if P.MULTI:
    plt.savefig(f'{P0.FIGOUTDIR}/fig_twin_evol_at_{maxout}.png', dpi=300, facecolor='w', bbox_inches='tight', pad_inches=0.10, transparent=False)
    print(f" > Saved `{P0.FIGOUTDIR}/fig_twin_evol_at_{maxout}.png`")
else:
    plt.savefig(f'{P0.FIGOUTDIR}/fig_twin_last_at_{maxout}.png', dpi=300, facecolor='w', bbox_inches='tight', pad_inches=0.10, transparent=False)
    print(f" > Saved `{P0.FIGOUTDIR}/fig_twin_evol_at_{maxout}.png`")