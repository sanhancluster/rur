from rur import uri, uhmi, painter
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

import params_nc as P
import params as P0




def draw_sfms(ax, **kwargs):
    global P
    gals1 = kwargs['gals1']
    gals2 = kwargs['gals2']

    key = 'SFMS'
    ykey = P.DETAIL[key]['key']
    sfr_year = 10 if 'SFR10' in ykey else 100

    xvals = gals1['m']
    yvals = gals1[ykey]
    if P.SHOW_SCATTER:
        ax.scatter(xvals, yvals, ec=P.COLOR1, fc='none', s=20, zorder=1, lw=0.5)
        if P.SHOW_OTHER: ax.scatter(gals2['m'], gals2[ykey], fc='none', ec=P.COLOR2, s=20, zorder=-1, lw=0.5)

    xmin = P.XMIN if P.DETAIL[key]['xmin'] is None else P.DETAIL[key]['xmin']
    xmax = P.XMAX if P.DETAIL[key]['xmax'] is None else P.DETAIL[key]['xmax']
    ymin = P.YMIN if P.DETAIL[key]['ymin'] is None else P.DETAIL[key]['ymin']
    ymax = P.YMAX if P.DETAIL[key]['ymax'] is None else P.DETAIL[key]['ymax']

    bins = np.logspace(np.log10(xmin), np.log10(xmax), P.NBIN+1)
    mbins = np.sqrt(bins[:-1]*bins[1:])
    ybins = np.zeros((3,P.NBIN))
    mask0 = (yvals>0)
    for i in range(P.NBIN):
        mask = (xvals>bins[i]) & (xvals<bins[i+1]) & mask0
        if np.sum(mask)>=3:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.percentile(yvals[mask], [P.PERCENTILE[0], 50, P.PERCENTILE[1]])
        else:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.nan, np.nan, np.nan
    ax.errorbar(
        mbins, ybins[1], yerr=[ybins[1]-ybins[0], ybins[2]-ybins[1]], 
        fmt='o', color=P.COLOR3, zorder=2, markersize=5,
        elinewidth=1, capsize=3, capthick=1
        )
    ax.plot(mbins, ybins[1], color=P.COLOR3, lw=1, ls='--', zorder=2)

    if xmin is not None or xmax is not None: ax.set_xlim(xmin, xmax)
    ax.set_xscale('log')
    ax.set_xlabel(r"$M_*\,[M_\odot]$")
    if ymin is not None or ymax is not None: ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')
    ax.set_ylabel(fr"$\mathrm{{SFR}}_{{{sfr_year}}}\,[M_\odot\,\mathrm{{yr}}^{{-1}}]$")

def draw_size(ax, **kwargs):
    global P
    gals1 = kwargs['gals1']
    gals2 = kwargs['gals2']
    snap = kwargs['snap']

    key = 'SIZE'
    ykey = P.DETAIL[key]['key']
    r_radii = 50 if '50' in ykey else 90
    runit = P.DETAIL[key]['unit']


    xvals = gals1['m']
    yvals = gals1[ykey]/snap.unit[runit]
    if P.SHOW_SCATTER:
        ax.scatter(xvals, yvals, ec=P.COLOR1, fc='none', s=20, zorder=1, lw=0.5)
        if P.SHOW_OTHER: ax.scatter(gals2['m'], gals2[ykey]/snap.unit[runit], fc='none', ec=P.COLOR2, s=20, zorder=-1, lw=0.5)

    xmin = P.XMIN if P.DETAIL[key]['xmin'] is None else P.DETAIL[key]['xmin']
    xmax = P.XMAX if P.DETAIL[key]['xmax'] is None else P.DETAIL[key]['xmax']
    ymin = P.YMIN if P.DETAIL[key]['ymin'] is None else P.DETAIL[key]['ymin']
    ymax = P.YMAX if P.DETAIL[key]['ymax'] is None else P.DETAIL[key]['ymax']

    bins = np.logspace(np.log10(xmin), np.log10(xmax), P.NBIN+1)
    mbins = np.sqrt(bins[:-1]*bins[1:])
    ybins = np.zeros((3,P.NBIN))
    for i in range(P.NBIN):
        mask = (xvals>bins[i]) & (xvals<bins[i+1])
        if np.sum(mask)>=3:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.percentile(yvals[mask], [P.PERCENTILE[0], 50, P.PERCENTILE[1]])
        else:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.nan, np.nan, np.nan
    ax.errorbar(
        mbins, ybins[1], yerr=[ybins[1]-ybins[0], ybins[2]-ybins[1]], 
        fmt='o', color=P.COLOR3, zorder=2, markersize=5,
        elinewidth=1, capsize=3, capthick=1
        )
    ax.plot(mbins, ybins[1], color=P.COLOR3, lw=1, ls='--', zorder=2)

    if xmin is not None or xmax is not None: ax.set_xlim(xmin, xmax)
    ax.set_xscale('log')
    ax.set_xlabel(r"$M_*\,[M_\odot]$")
    if ymin is not None or ymax is not None: ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')
    ax.set_ylabel(fr"$\mathrm{{R}}_{{{r_radii}}}\,[\mathrm{{kpc}}]$")

def draw_cold(ax, **kwargs):
    global P
    gals1 = kwargs['gals1']
    gals2 = kwargs['gals2']

    key = 'Cold'
    ykey = P.DETAIL[key]['key']
    # 'Mcold_gas', 'Mcold_gas_r50', 'Mcold_gas_r90', 'Mdense_gas', 'Mdense_gas_r50', 'Mdense_gas_r90'
    coldname = 'cold' if 'cold' in ykey else 'dense'
    if 'r50' in ykey:
        coldrange = 'R50'
    elif 'r90' in ykey:
        coldrange = 'R90'
    else:
        coldrange = 'Rmax'


    xvals = gals1['m']
    yvals = gals1[ykey]/xvals
    if P.SHOW_SCATTER:
        ax.scatter(xvals, yvals, ec=P.COLOR1, fc='none', s=20, zorder=1, lw=0.5)
        if P.SHOW_OTHER: ax.scatter(gals2['m'], gals2[ykey]/gals2['m'], fc='none', ec=P.COLOR2, s=20, zorder=-1, lw=0.5)

    xmin = P.XMIN if P.DETAIL[key]['xmin'] is None else P.DETAIL[key]['xmin']
    xmax = P.XMAX if P.DETAIL[key]['xmax'] is None else P.DETAIL[key]['xmax']
    ymin = P.YMIN if P.DETAIL[key]['ymin'] is None else P.DETAIL[key]['ymin']
    ymax = P.YMAX if P.DETAIL[key]['ymax'] is None else P.DETAIL[key]['ymax']

    bins = np.logspace(np.log10(xmin), np.log10(xmax), P.NBIN+1)
    mbins = np.sqrt(bins[:-1]*bins[1:])
    ybins = np.zeros((3,P.NBIN))
    for i in range(P.NBIN):
        mask = (xvals>bins[i]) & (xvals<bins[i+1]) & (gals1[ykey]>0)
        if np.sum(mask)>=3:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.percentile(yvals[mask], [P.PERCENTILE[0], 50, P.PERCENTILE[1]])
        else:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.nan, np.nan, np.nan
    ax.errorbar(
        mbins, ybins[1], yerr=[ybins[1]-ybins[0], ybins[2]-ybins[1]], 
        fmt='o', color=P.COLOR3, zorder=2, markersize=5,
        elinewidth=1, capsize=3, capthick=1
        )
    ax.plot(mbins, ybins[1], color=P.COLOR3, lw=1, ls='--', zorder=2)

    if xmin is not None or xmax is not None: ax.set_xlim(xmin, xmax)
    ax.set_xscale('log')
    ax.set_xlabel(r"$M_*\,[M_\odot]$")
    if ymin is not None or ymax is not None: ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')
    ax.set_ylabel(fr"$\mathrm{{M}}_{{{coldname},\,{coldrange}}}/M_*$")

def draw_metal(ax, key='Metal', **kwargs):
    global P
    gals1 = kwargs['gals1']
    gals2 = kwargs['gals2']

    # 'metal', 'metal_gas', 'O/H'
    ykey = P.DETAIL[key]['key']
    xvals = gals1['m']
    def get_metal(gals, ykey):
        global P
        if 'metal' in ykey:
            if 'gas' in ykey:
                yname = r"$Z_{gas}$"
            else:
                yname = r"$Z_{*}$"
            yvals = gals[ykey]
        elif 'O/H' in ykey:
            yname = r"12 + log(O/H)$_{\rm gas}$"
            yvals = 12 + np.log10(gals['O_gas']/16/gals['H_gas'])
        return yvals, yname
    yvals, yname = get_metal(gals1, ykey)

    if P.SHOW_SCATTER:
        ax.scatter(xvals, yvals, ec=P.COLOR1, fc='none', s=20, zorder=1, lw=0.5)
        if P.SHOW_OTHER: ax.scatter(gals2['m'], get_metal(gals2,ykey)[0], fc='none', ec=P.COLOR2, s=20, zorder=-1, lw=0.5)

    xmin = P.XMIN if P.DETAIL[key]['xmin'] is None else P.DETAIL[key]['xmin']
    xmax = P.XMAX if P.DETAIL[key]['xmax'] is None else P.DETAIL[key]['xmax']
    ymin = P.YMIN if P.DETAIL[key]['ymin'] is None else P.DETAIL[key]['ymin']
    ymax = P.YMAX if P.DETAIL[key]['ymax'] is None else P.DETAIL[key]['ymax']

    bins = np.logspace(np.log10(xmin), np.log10(xmax), P.NBIN+1)
    mbins = np.sqrt(bins[:-1]*bins[1:])
    ybins = np.zeros((3,P.NBIN))
    for i in range(P.NBIN):
        mask = (xvals>bins[i]) & (xvals<bins[i+1])
        if np.sum(mask)>=3:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.percentile(yvals[mask], [P.PERCENTILE[0], 50, P.PERCENTILE[1]])
        else:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.nan, np.nan, np.nan
    ax.errorbar(
        mbins, ybins[1], yerr=[ybins[1]-ybins[0], ybins[2]-ybins[1]], 
        fmt='o', color=P.COLOR3, zorder=2, markersize=5,
        elinewidth=1, capsize=3, capthick=1
        )
    ax.plot(mbins, ybins[1], color=P.COLOR3, lw=1, ls='--', zorder=2)

    if xmin is not None or xmax is not None: ax.set_xlim(xmin, xmax)
    ax.set_xscale('log')
    ax.set_xlabel(r"$M_*\,[M_\odot]$")
    if ymin is not None or ymax is not None: ax.set_ylim(ymin, ymax)
    if not 'O/H' in ykey:
        ax.set_yscale('log')
    ax.set_ylabel(yname)

def draw_mbh(ax, **kwargs):
    global P
    gals1 = kwargs['gals1']
    gals2 = kwargs['gals2']
    snap = kwargs['snap']

    key = 'MBH'
    ykey = P.DETAIL[key]['key']


    xvals = gals1['m']
    yvals = gals1[ykey]
    mask0 = (yvals >= P.DETAIL[key]['Mseed']) & (gals1['dBH']/snap.unit['pc'] < P.DETAIL[key]['dBH'])
    if P.SHOW_SCATTER:
        ax.scatter(xvals, yvals, ec=P.COLOR1, fc='none', s=20, zorder=1, lw=0.5)
        ax.scatter(xvals[mask0], yvals[mask0], fc=P.COLOR1, ec='none', s=20, zorder=1.5, lw=0.5)
        if P.SHOW_OTHER: ax.scatter(gals2['m'], gals2[ykey], fc='none', ec=P.COLOR2, s=20, zorder=-1, lw=0.5)

    xmin = P.XMIN if P.DETAIL[key]['xmin'] is None else P.DETAIL[key]['xmin']
    xmax = P.XMAX if P.DETAIL[key]['xmax'] is None else P.DETAIL[key]['xmax']
    ymin = P.YMIN if P.DETAIL[key]['ymin'] is None else P.DETAIL[key]['ymin']
    ymax = P.YMAX if P.DETAIL[key]['ymax'] is None else P.DETAIL[key]['ymax']

    bins = np.logspace(np.log10(xmin), np.log10(xmax), P.NBIN+1)
    mbins = np.sqrt(bins[:-1]*bins[1:])
    ybins = np.zeros((3,P.NBIN))
    for i in range(P.NBIN):
        mask = (xvals>bins[i]) & (xvals<bins[i+1]) & mask0
        if np.sum(mask)>=3:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.percentile(yvals[mask], [P.PERCENTILE[0], 50, P.PERCENTILE[1]])
        else:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.nan, np.nan, np.nan
    ax.errorbar(
        mbins, ybins[1], yerr=[ybins[1]-ybins[0], ybins[2]-ybins[1]], 
        fmt='o', color=P.COLOR3, zorder=2, markersize=5,
        elinewidth=1, capsize=3, capthick=1
        )
    ax.plot(mbins, ybins[1], color=P.COLOR3, lw=1, ls='--', zorder=2)

    if xmin is not None or xmax is not None: ax.set_xlim(xmin, xmax)
    ax.set_xscale('log')
    ax.set_xlabel(r"$M_*\,[M_\odot]$")
    if ymin is not None or ymax is not None: ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')
    ax.set_ylabel(fr"$\mathrm{{M}}_{{\mathrm{{BH}}}}\,[M_\odot]$")

def draw_shmr(ax, hals1, gids, **kwargs):
    global P
    gals = kwargs['gals']

    key = 'SHMR'

    Msep = P.DETAIL[key]['Msep']
    mask = (gids>0)&(hals1['mvir']<Msep)&(hals1['level']==1)
    xvals = hals1['mvir'][mask]
    yvals = gals[gids-1]['m'][mask] / xvals
    if P.SHOW_SCATTER:
        ax.scatter(xvals, yvals, ec=P.COLOR1, fc='none', s=20, zorder=1, lw=0.5)
        if P.SHOW_OTHER:
            mask2 = (gids>0)&(hals1['mvir']<Msep)&(hals1['level']>1)
            ax.scatter(hals1['mvir'][mask2], gals[gids[mask2]-1]['m']/hals1['mvir'][mask2], fc='none', ec=P.COLOR2, s=20, zorder=-1, lw=0.5)

    CLUSTER = hals1[hals1['mvir']>=Msep]
    cxvals = CLUSTER['mvir']
    cyvals = gals[gids[hals1['mvir']>=Msep]-1]['m'] / cxvals
    ax.scatter(cxvals, cyvals, fc=P.COLOR1, ec='none', marker='*', s=50, zorder=4)
    xmin = P.HXMIN if P.DETAIL[key]['xmin'] is None else P.DETAIL[key]['xmin']
    xmax = P.HXMAX if P.DETAIL[key]['xmax'] is None else P.DETAIL[key]['xmax']
    ymin = P.YMIN if P.DETAIL[key]['ymin'] is None else P.DETAIL[key]['ymin']
    ymax = P.YMAX if P.DETAIL[key]['ymax'] is None else P.DETAIL[key]['ymax']

    bins = np.logspace(np.log10(xmin), np.log10(min(xmax, Msep)), P.NBIN+1)
    mbins = np.sqrt(bins[:-1]*bins[1:])
    ybins = np.zeros((3,P.NBIN))
    for i in range(P.NBIN):
        mask = (xvals>bins[i]) & (xvals<bins[i+1])
        if np.sum(mask)>=3:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.percentile(yvals[mask], [P.PERCENTILE[0], 50, P.PERCENTILE[1]])
        else:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.nan, np.nan, np.nan
    ax.errorbar(
        mbins, ybins[1], yerr=[ybins[1]-ybins[0], ybins[2]-ybins[1]], 
        fmt='o', color=P.COLOR3, zorder=2, markersize=5,
        elinewidth=1, capsize=3, capthick=1
        )
    ax.plot(mbins, ybins[1], color=P.COLOR3, lw=1, ls='--', zorder=2)

    if xmin is not None or xmax is not None: ax.set_xlim(xmin, xmax)
    ax.set_xscale('log')
    ax.set_xlabel(r"$M_\mathrm{vir}\,[M_\odot]$")
    if ymin is not None or ymax is not None: ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')
    ax.set_ylabel(r"$M_*/M_\mathrm{vir}$")

def draw_DTM(ax, **kwargs):
    global P
    gals1 = kwargs['gals1']
    gals2 = kwargs['gals2']

    key = 'DTM'
    ykeys = P.DETAIL[key]['key']

    xvals = gals1['m']
    def get_DTM(gals):
        fdust = np.zeros(len(gals))
        for ykey, coeff in ykeys.items():
            fdust += gals[ykey] * coeff
        fmetal = gals['metal_gas']
        yvals = fdust / fmetal
        return yvals
    yvals = get_DTM(gals1)

    if P.SHOW_SCATTER:
        ax.scatter(xvals, yvals, ec=P.COLOR1, fc='none', s=20, zorder=1, lw=0.5)
        if P.SHOW_OTHER: ax.scatter(gals2['m'], get_DTM(gals2), fc='none', ec=P.COLOR2, s=20, zorder=-1, lw=0.5)

    xmin = P.XMIN if P.DETAIL[key]['xmin'] is None else P.DETAIL[key]['xmin']
    xmax = P.XMAX if P.DETAIL[key]['xmax'] is None else P.DETAIL[key]['xmax']
    ymin = P.YMIN if P.DETAIL[key]['ymin'] is None else P.DETAIL[key]['ymin']
    ymax = P.YMAX if P.DETAIL[key]['ymax'] is None else P.DETAIL[key]['ymax']

    bins = np.logspace(np.log10(xmin), np.log10(xmax), P.NBIN+1)
    mbins = np.sqrt(bins[:-1]*bins[1:])
    ybins = np.zeros((3,P.NBIN))
    mask0 = (yvals>0)
    for i in range(P.NBIN):
        mask = (xvals>bins[i]) & (xvals<bins[i+1]) & mask0
        if np.sum(mask)>=3:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.percentile(yvals[mask], [P.PERCENTILE[0], 50, P.PERCENTILE[1]])
        else:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.nan, np.nan, np.nan
    ax.errorbar(
        mbins, ybins[1], yerr=[ybins[1]-ybins[0], ybins[2]-ybins[1]], 
        fmt='o', color=P.COLOR3, zorder=2, markersize=5,
        elinewidth=1, capsize=3, capthick=1
        )
    ax.plot(mbins, ybins[1], color=P.COLOR3, lw=1, ls='--', zorder=2)

    if xmin is not None or xmax is not None: ax.set_xlim(xmin, xmax)
    ax.set_xscale('log')
    ax.set_xlabel(r"$M_*\,[M_\odot]$")
    if ymin is not None or ymax is not None: ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')
    ax.set_ylabel("DTM")

def draw_CMD(ax, **kwargs):
    from scipy.ndimage import gaussian_filter
    global P
    gals1 = kwargs['gals1']
    gals2 = kwargs['gals2']

    key = 'CMD'
    band = P.DETAIL[key]['band']
    xvals = gals1[f"{band}mag"]
    colorkey = P.DETAIL[key]['colorkey']
    band1,band2 = colorkey.split('-')
    yvals = gals1[f"{band1}mag"] - gals1[f"{band2}mag"]

    if P.SHOW_SCATTER:
        ax.scatter(xvals, yvals, ec=P.COLOR1, fc='none', s=20, zorder=1, lw=0.5)
        if P.SHOW_OTHER: ax.scatter(gals2[f"{band}mag"], gals2[f"{band1}mag"] - gals2[f"{band2}mag"], fc='none', ec=P.COLOR2, s=20, zorder=-1, lw=0.5)

    xmin = P.XMIN if P.DETAIL[key]['xmin'] is None else P.DETAIL[key]['xmin']
    xmax = P.XMAX if P.DETAIL[key]['xmax'] is None else P.DETAIL[key]['xmax']
    ymin = P.YMIN if P.DETAIL[key]['ymin'] is None else P.DETAIL[key]['ymin']
    ymax = P.YMAX if P.DETAIL[key]['ymax'] is None else P.DETAIL[key]['ymax']

    xbins = np.linspace(xmin, xmax, 3*P.NBIN+1)
    ybins = np.linspace(ymin, ymax, 3*P.NBIN+1)
    hist, xe, ye = np.histogram2d(xvals, yvals, bins=[xbins, ybins], density=True)
    hist = gaussian_filter(hist, sigma=0.4)
    hflat = hist.flatten()
    hflat = np.sort(hflat[hflat>0])[::-1]
    chflat = np.cumsum(hflat) / np.sum(hflat)
    sigmas = [0.3829, 0.6827, 0.9545] # 0.5, 1, 2 sigma
    levels = np.sort([hflat[np.searchsorted(chflat, s)] for s in sigmas])
    xe = 0.5*(xe[1:]+xe[:-1])
    ye = 0.5*(ye[1:]+ye[:-1])
    con = ax.contour(xe, ye, hist.T, levels=levels, colors=P.COLOR3, alpha=0.7, zorder=3, linewidths=[0.5,1,1.5])


    if xmin is not None or xmax is not None: ax.set_xlim(xmin, xmax)
    ax.set_xlabel(fr"$M_{band}$")
    ax.invert_xaxis()
    if ymin is not None or ymax is not None: ax.set_ylim(ymin, ymax)
    ax.set_ylabel(fr"${band1}-{band2}$")

def draw_SB(ax, **kwargs):
    from scipy.ndimage import gaussian_filter
    global P
    gals1 = kwargs['gals1']
    gals2 = kwargs['gals2']

    key = 'SB'
    band = P.DETAIL[key]['band']
    xvals = gals1[f"{band}mag"]
    yvals = gals1[f"SB{band}_r50"]

    if P.SHOW_SCATTER:
        ax.scatter(xvals, yvals, ec=P.COLOR1, fc='none', s=20, zorder=1, lw=0.5)
        if P.SHOW_OTHER: ax.scatter(gals2[f"{band}mag"], gals2[f"SB{band}_r50"], fc='none', ec=P.COLOR2, s=20, zorder=-1, lw=0.5)

    xmin = P.XMIN if P.DETAIL[key]['xmin'] is None else P.DETAIL[key]['xmin']
    xmax = P.XMAX if P.DETAIL[key]['xmax'] is None else P.DETAIL[key]['xmax']
    ymin = P.YMIN if P.DETAIL[key]['ymin'] is None else P.DETAIL[key]['ymin']
    ymax = P.YMAX if P.DETAIL[key]['ymax'] is None else P.DETAIL[key]['ymax']

    bins = np.linspace(xmin, xmax, P.NBIN+1)
    mbins = 0.5*(bins[:-1]+bins[1:])
    ybins = np.zeros((3,P.NBIN))
    for i in range(P.NBIN):
        mask = (xvals>bins[i]) & (xvals<bins[i+1])
        if np.sum(mask)>=3:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.percentile(yvals[mask], [P.PERCENTILE[0], 50, P.PERCENTILE[1]])
        else:
            ybins[0,i], ybins[1,i], ybins[2,i] = np.nan, np.nan, np.nan
    ax.errorbar(
        mbins, ybins[1], yerr=[ybins[1]-ybins[0], ybins[2]-ybins[1]], 
        fmt='o', color=P.COLOR3, zorder=2, markersize=5,
        elinewidth=1, capsize=3, capthick=1
        )
    ax.plot(mbins, ybins[1], color=P.COLOR3, lw=1, ls='--', zorder=2)


    if xmin is not None or xmax is not None: ax.set_xlim(xmin, xmax)
    ax.set_xlabel(fr"$M_{band}$")
    ax.invert_xaxis()
    if ymin is not None or ymax is not None: ax.set_ylim(ymin, ymax)
    ax.set_ylabel(fr"$\mu_{{e,{band}}}$")
    ax.invert_yaxis()

def draw_massfunc(ax, gals, hals):
    global P
    hdex = 0.45; hmin = 8; hmax=15
    hbins = np.arange(hmin, hmax+hdex, hdex)
    hx = (hbins[:-1] + hbins[1:]) / 2
    gdex = 0.45; smin=6.25; smax = 13
    gbins = np.arange(smin, smax+gdex, gdex)
    gx = (gbins[:-1] + gbins[1:]) / 2

    hmf = np.histogram(np.log10(hals['m']), bins=hbins)[0] # number
    volume = 23300 # NC
    hmf = hmf / volume / (hbins[1] - hbins[0]) # number density
    zeromask = hmf>0
    ax.plot(hx[zeromask], hmf[zeromask], lw=1, color=P.COLOR1, ls='--', zorder=0)

    gmf = np.histogram(np.log10(gals['m']), bins=gbins)[0] # number
    gmf = gmf / volume / (gbins[1] - gbins[0]) # number density
    zeromask = gmf>0
    ax.plot(gx[zeromask], gmf[zeromask], lw=1, color=P.COLOR1, ls='-', zorder=0)

    ax.set_yscale('log')
    ax.set_ylabel(r'd$n$/d$\log M\ [\rm{ cMpc^{-3}\,dex^{-1}}]$')
    ax.set_xlabel(r'$\log( M\,/\,{\rm M_\odot})$')

def draw_alpha(ax, **kwargs):
    from scipy.ndimage import gaussian_filter
    from rur.sci.chemistry import solar_frac
    FeH_solar = solar_frac['Fe'] / solar_frac['H']
    OFe_solar = solar_frac['O'] / solar_frac['Fe']
    MgFe_solar = solar_frac['Mg'] / solar_frac['Fe']
    SiFe_solar = solar_frac['Si'] / solar_frac['Fe']
    aFe_solar = (OFe_solar + MgFe_solar + SiFe_solar) / 3

    global P
    gals1 = kwargs['gals1']
    gals2 = kwargs['gals2']

    key = 'alpha'
    def get_FeH(gals):
        Fe = gals['Fe_gas']
        H = gals['H_gas']
        return np.log10( Fe/H/FeH_solar )
    def get_aFe(gals):
        alpha = (gals['O_gas']+gals['Mg_gas']+gals['Si_gas'])/3
        Fe = gals['Fe_gas']
        return np.log10( alpha/Fe/aFe_solar )
    xvals = get_FeH(gals1)
    yvals = get_aFe(gals1)

    if P.SHOW_SCATTER:
        ax.scatter(xvals, yvals, ec=P.COLOR1, fc='none', s=20, zorder=1, lw=0.5)
        if P.SHOW_OTHER: ax.scatter(get_FeH(gals2), get_aFe(gals2), fc='none', ec=P.COLOR2, s=20, zorder=-1, lw=0.5)

    xmin = P.XMIN if P.DETAIL[key]['xmin'] is None else P.DETAIL[key]['xmin']
    xmax = P.XMAX if P.DETAIL[key]['xmax'] is None else P.DETAIL[key]['xmax']
    ymin = P.YMIN if P.DETAIL[key]['ymin'] is None else P.DETAIL[key]['ymin']
    ymax = P.YMAX if P.DETAIL[key]['ymax'] is None else P.DETAIL[key]['ymax']

    xbins = np.linspace(xmin, xmax, 4*P.NBIN+1)
    ybins = np.linspace(ymin, ymax, 4*P.NBIN+1)
    hist, xe, ye = np.histogram2d(xvals, yvals, bins=[xbins, ybins], density=True)
    hist = gaussian_filter(hist, sigma=0.6)
    hflat = hist.flatten()
    hflat = np.sort(hflat[hflat>0])[::-1]
    chflat = np.cumsum(hflat) / np.sum(hflat)
    sigmas = [0.3829, 0.6827, 0.9545] # 0.5, 1, 2 sigma
    levels = np.sort([hflat[np.searchsorted(chflat, s)] for s in sigmas])
    xe = 0.5*(xe[1:]+xe[:-1])
    ye = 0.5*(ye[1:]+ye[:-1])
    con = ax.contour(xe, ye, hist.T, levels=levels, colors=P.COLOR3, alpha=0.7, zorder=3, linewidths=[0.5,1,1.5])


    if xmin is not None or xmax is not None: ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r"$\mathrm{[Fe/H]}$")
    if ymin is not None or ymax is not None: ax.set_ylim(ymin, ymax)
    ax.set_ylabel(r"$\mathrm{[\alpha/Fe]}$")

