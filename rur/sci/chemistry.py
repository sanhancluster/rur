import numpy as np
import matplotlib.pyplot as plt
from rur import uri, painter, uhmi, drawer
from matplotlib.colors import LogNorm

solar_OFe = 0.39389979 / 0.08884559
solar_FeH = 0.001767991866
solar_MgFe = 0.0486535 / 0.08884559
solar_SiFe = 0.04570737 / 0.08884559
solar_SFe = 0.0212605 / 0.08884559

def O_over_Fe(data, total=False):
    solar_OFe = 0.39389979 / 0.08884559
    if(total):
        return np.log10(np.average(data['O'] / data['Fe'], weights=data['m']) / solar_OFe)
    else:
        return np.log10(data['O'] / data['Fe'] / solar_OFe)


## figures

def draw_alpha(data, lims=[[-2.5, 0.5], [-1.0, 2.0]], **kwargs):

    fig, axes = plt.subplots(figsize=(10, 10), dpi=150, ncols=2, nrows=2)
    plt.sca(axes[0, 0])
    drawer.axlabel('z = %.3f' % data.snap.z, color='k', pos='left top')
    drawer.axlabel('O', color='k', pos='right top')
    drawer.hist_imshow(np.log10(data['Fe'] / data['H'] / solar_FeH), np.log10(data['O'] / data['Fe'] / solar_OFe),
                       lims=lims, weights=data['m'], norm=LogNorm(), **kwargs)
    plt.xlabel('[Fe/H]')
    plt.ylabel('[O/Fe]')

    plt.sca(axes[0, 1])
    drawer.axlabel('Mg', color='k', pos='right top')
    drawer.hist_imshow(np.log10(data['Fe'] / data['H'] / solar_FeH), np.log10(data['Mg'] / data['Fe'] / solar_MgFe),
                       lims=lims, weights=data['m'], norm=LogNorm(), **kwargs)
    plt.xlabel('[Fe/H]')
    plt.ylabel('[Mg/Fe]')

    plt.sca(axes[1, 0])
    drawer.axlabel('Si', color='k', pos='right top')
    drawer.hist_imshow(np.log10(data['Fe'] / data['H'] / solar_FeH), np.log10(data['Si'] / data['Fe'] / solar_SiFe),
                       lims=lims, weights=data['m'], norm=LogNorm(), **kwargs)
    plt.xlabel('[Fe/H]')
    plt.ylabel('[Si/Fe]')

    plt.sca(axes[1, 1])
    drawer.axlabel('S', color='k', pos='right top')
    drawer.hist_imshow(np.log10(data['Fe'] / data['H'] / solar_FeH), np.log10(data['S'] / data['Fe'] / solar_SFe),
                       lims=lims, weights=data['m'], norm=LogNorm(), **kwargs)
    plt.xlabel('[Fe/H]')
    plt.ylabel('[S/Fe]')