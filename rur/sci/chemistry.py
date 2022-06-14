import numpy as np
import matplotlib.pyplot as plt
from rur import drawer
from matplotlib.colors import LogNorm

solar_frac = {
    'H' : 0.08884559 / 0.001767991866,
    'C' : 0.16245322,
    'N' : 0.04757141,
    'O' : 0.39389979,
    'Mg': 0.04865350,
    'Si': 0.04570737,
    'S' : 0.02126050,
    'Fe': 0.08884559,
}
solar_abundance = solar_frac

def O_over_Fe(data, total=False):
    solar_OFe = 0.39389979 / 0.08884559
    if(total):
        return np.log10(np.average(data['O'] / data['Fe'], weights=data['m']) / solar_OFe)
    else:
        return np.log10(data['O'] / data['Fe'] / solar_OFe)


## figures

def draw_alpha(data, lims=[[-2.5, 0.5], [-1.0, 2.0]], ncols=2, nrows=2, chems=['O', 'Mg', 'Si', 'S'], chem_metal='Fe', ax_color='k', norm=LogNorm(), **kwargs):
    chems = iter(chems)

    fig, axes = plt.subplots(figsize=(ncols*4.5, nrows*4.5), dpi=150, ncols=ncols, nrows=nrows, sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for icol in range(ncols):
        for irow in range(nrows):
            chem = next(chems)
            plt.sca(axes[irow, icol])
            if(icol + irow == 0):
                drawer.axlabel('z = %.3f' % data.snap.z, color='k', pos='left top')
            drawer.axlabel(chem, color=ax_color, pos='right top')
            if(irow == nrows - 1):
                plt.xlabel('[%s/H]' % chem_metal)
            if(icol == 0):
                plt.ylabel('[X/%s]' % chem_metal)
            XM = (solar_frac[chem] / solar_frac[chem_metal])
            MH = (solar_frac[chem_metal] / solar_frac['H'])
            drawer.hist_imshow(np.log10(data[chem_metal] / data['H'] / MH), np.log10(data[chem] / data[chem_metal] / XM),
                               lims=lims, weights=data['m'], norm=norm, **kwargs)
            plt.axhline(0, color=ax_color, lw=0.5)
            plt.axvline(0, color=ax_color, lw=0.5)