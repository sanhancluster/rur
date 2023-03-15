from rur import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from rur.config import m_H, c_const, G_const, sigma_T
from rur.sci.relations import mgal_mbh, sigma_mbh

def set_unit(aexps, snap):
    aexp_scale = snap.aexp / aexps
    unit = {}
    unit['m'] = snap.unit_m  # unit to g
    unit['l'] = snap.unit_l / aexp_scale  # unit to cm
    unit['d'] = unit['m'] / unit['l'] ** 3  # unit to g/cm^3
    unit['t'] = snap.unit_t / aexp_scale ** 2  # unit to s
    unit['v'] = unit['l'] / unit['t']  # unit to cm/s

    unit['to_kms'] = unit['v'] / uri.km
    unit['to_Msol'] = unit['m'] / uri.Msol
    unit['to_Hcc'] = unit['d'] / uri.m_H
    unit['to_yr'] = unit['t'] / uri.yr

    return unit


def draw_sink_timeline(snap, tl, modes=None, xmode='aexp', xlim=None, plot_params=dict(),
                       smooth=1, eagn_T=0.05, vlines=[], title=None, savefile=None):
    """
    Available modes = ['mass', 'velocity', 'density', 'accretion_rate', 'eddington_rate', 'spin', 'epsilon', 'energy',
                 'tot_energy']
    """
    def draw(x, y, lsmooth=False, **kwargs):
        if (smooth > 0):
            if (lsmooth):
                y = np.log10(gaussian_filter1d(10 ** y, smooth))
            else:
                y = gaussian_filter1d(y, smooth)
        plt.plot(x, y, **kwargs)

    if modes is None:
        modes = ['mass', 'velocity', 'density', 'accretion_rate', 'eddington_rate', 'spin', 'epsilon', 'energy',
                 'tot_energy']
    nrows = len(modes)
    fig, axes = plt.subplots(ncols=1, nrows=nrows, figsize=(8, nrows * 2), dpi=150, sharex=True)
    if(title is not None):
        axes[0].set_title(title)
    plt.subplots_adjust(hspace=0.1)
    tl.sort(order='aexp')
    if (xmode == 'aexp'):
        xarr = tl['aexp']
    if (xmode == 'icoarse'):
        xarr = tl['icoarse']

    aexp_scale = snap.aexp / tl['aexp']
    unit_m = snap.unit_m  # unit to g
    unit_l = snap.unit_l / aexp_scale  # unit to cm
    unit_d = unit_m / unit_l ** 3  # unit to g/cm^3
    unit_t = snap.unit_t / aexp_scale ** 2  # unit to s
    unit_v = unit_l / unit_t  # unit to cm/s

    unit_to_kms = unit_v / uri.km
    unit_to_Msol = unit_m / uri.Msol
    unit_to_Hcc = unit_d / uri.m_H
    unit_to_yr = unit_t / uri.yr
    dt_yr = np.gradient(tl['aexp']) / snap.aexp_to_dadt(tl['aexp']) * 1E9

    for irow, mode in enumerate(modes):
        plt.sca(axes[irow])
        if (mode == 'mass'):
            yarr = np.log10(tl['m'] * unit_to_Msol)
            draw(xarr, yarr, **plot_params)
            plt.ylabel('log M$_{BH}$\n(M$_\odot$)')
            # plt.ylim(5, 9)
        elif (mode == 'velocity'):
            draw(xarr, np.log10(tl['v_avgptr'] * unit_to_kms), lw=0.5, label='v$_{gas}$')
            draw(xarr, np.log10(tl['c_avgptr'] * unit_to_kms), lw=0.5, label='c$_{gas}$', color='r')
            # plt.plot(aexp, np.log10(tl['c_avgptr']/unit), lw=0.5, label='c$_{gas}$', color='red')
            if ('star_vx' in tl.dtype.names):
                vstar = utool.rss(utool.get_vector(tl, 'star_v') - utool.get_vector(tl, 'v')) * unit_to_kms
                vdm = utool.rss(utool.get_vector(tl, 'dm_v') - utool.get_vector(tl, 'v')) * unit_to_kms
                draw(xarr, np.log10(vstar), lw=0.5, label='v$_{star}$')
                draw(xarr, np.log10(vdm), lw=0.5, label='v$_{dm}$')
            plt.ylabel('log v\n(km/s)')
            plt.ylim(0.5, 3)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        elif (mode == 'density'):
            draw(xarr, np.log10(tl['d_avgptr'] * unit_to_Hcc), lw=0.5, label='$\\rho_{gas}$')
            if ('star_vx' in tl.dtype.names):
                draw(xarr, np.log10(tl['rho_star'] * unit_to_Hcc), lw=0.5, label='$\\rho_{star}$')
                draw(xarr, np.log10(tl['rho_dm'] * unit_to_Hcc), lw=0.5, label='$\\rho_{DM}$')
            # plt.plot(aexp, np.repeat(np.log10(5), aexp.size), lw=0.5, color='gray')
            plt.ylabel('log $\\rho$\n(H/cc)')
            plt.ylim(0.5, 4.5)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        elif (mode == 'accretion_rate'):
            # acc = lambda bh: 4*np.pi* bh['d_avgptr'] * unit_d * (6.674E-8)**2*(bh['m']*unit_m)**2 / (bh['v_avgptr']**2 + bh['c_avgptr']**2)**1.5/unit_v**3 / (uri.Msol/uri.yr)
            Mdot = tl['dM'] * unit_to_Msol / dt_yr / (1-tl['eps_sink'])
            Macc = tl['Mdot'] * unit_to_Msol / unit_to_yr
            Medd = tl['Medd'] * unit_to_Msol / unit_to_yr
            draw(xarr, np.log10(Mdot), lsmooth=True, lw=0.7, label='$\dotM_{BH}$')
            draw(xarr, np.log10(Macc), lsmooth=True, lw=0.7, label='$\dotM_{Bon}$')
            draw(xarr, np.log10(Medd), lsmooth=True, lw=0.7, label='$\dotM_{Edd}$')
            plt.ylabel('log M$_{acc}$\n(M$_{\odot}$/yr)')
            plt.ylim(-5, 2)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        elif (mode == 'eddington_rate'):
            Mdot = gaussian_filter1d(tl['dM'] * unit_to_Msol / dt_yr / (1-tl['eps_sink']), smooth)
            Macc = gaussian_filter1d(tl['Mdot'] * unit_to_Msol / unit_to_yr, smooth)
            Medd = gaussian_filter1d(tl['Medd'] * unit_to_Msol / unit_to_yr, smooth)
            plt.plot(xarr, np.log10(np.minimum(Mdot / Medd, 1)), label='f$_{acc}$')
            #plt.plot(xarr, np.log10(np.minimum(Macc / Medd, 1)), label='f$_{Bon}$')
            plt.plot(xarr, np.log10(np.minimum(tl['dM'] / dt_yr / tl['Medd'] * unit_to_yr / (1-tl['eps_sink']), 1)), color='k', alpha=0.3, lw=0.5, zorder=-1)
            plt.axhline(-2, color='gray', lw=0.5)
            plt.ylabel('log f$_{Edd}$')
            plt.ylim(-4, 0.05)
            #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        elif (mode == 'spin'):
            plt.plot(xarr, np.log10(1-np.abs(tl['spinmag'])), label='Mdot')
            # plt.plot(xarr, np.log10(np.minimum(tl['Mdot'] / tl['Medd'], 1)), color='k', alpha=0.3, lw=0.5, zorder=-1)
            # plt.axhline(-2, color='gray', lw=0.5)
            plt.ylabel('log (1-a/M)')
            plt.ylim(-3, 0)

        elif (mode == 'epsilon'):
            # draw(xarr, np.log10(np.minimum(tl['Mdot']/tl['Medd'], 1)), label='Mdot')
            eff = eff_mad(tl['spinmag'])
            plt.ylabel('$\eta_{AGN}$')

            draw(xarr, tl['eps_sink'] * eagn_T, lw=0.7, label='Thermal')
            draw(xarr, eff, lw=0.7, label='Kinetic')
            # plt.axhline(-2, color='gray', lw=0.5)
            # plt.ylim(-4, 0.05)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        elif (mode == 'energy'):
            # draw(xarr, np.log10(np.minimum(tl['Mdot']/tl['Medd'], 1)), label='Mdot')
            eff = eff_mad(tl['spinmag'])
            ek = (np.minimum(tl['Mdot'], tl['Medd']) * eff) * (tl['Mdot'] / tl['Medd'] < 0.01)
            et = (np.minimum(tl['Mdot'], tl['Medd']) * tl['eps_sink']) * (tl['Mdot'] / tl['Medd'] > 0.01) * eagn_T
            # stat = utool.binned_stat(ages_sink, np.stack([et, ek], axis=-1) * (unit_m / unit_t * 29979245800**2), bins=age_bin)
            ek *= (unit_m / unit_t * c_const ** 2)
            et *= (unit_m / unit_t * c_const ** 2)
            plt.ylabel('AGN Energy\n(10$^{43}$ erg s$^{-1}$)')

            draw(xarr, et / 1E43, lw=0.7, label='Thermal')
            draw(xarr, ek / 1E43, lw=0.7, label='Kinetic')
            # plt.axhline(-2, color='gray', lw=0.5)
            # plt.ylim(-4, 0.05)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        elif (mode == 'tot_energy'):
            # draw(xarr, np.log10(np.minimum(tl['Mdot']/tl['Medd'], 1)), label='Mdot')
            eff = eff_mad(tl['spinmag'])
            ek = (np.minimum(tl['Mdot'], tl['Medd']) * eff) * (tl['Mdot'] / tl['Medd'] < 0.01)
            et = (np.minimum(tl['Mdot'], tl['Medd']) * tl['eps_sink']) * (tl['Mdot'] / tl['Medd'] > 0.01) * eagn_T
            # stat = utool.binned_stat(ages_sink, np.stack([et, ek], axis=-1) * (unit_m / unit_t * 29979245800**2), bins=age_bin)
            ek *= (unit_m / unit_t * 29979245800. ** 2)
            et *= (unit_m / unit_t * 29979245800. ** 2)
            dt = np.diff(snap.aexp_to_age(tl['aexp'])) * uri.Gyr

            draw(xarr[1:], np.cumsum(dt * et[1:]) / 1E61, lw=0.7, label='Thermal')
            draw(xarr[1:], np.cumsum(dt * ek[1:]) / 1E61, lw=0.7, label='Kinetic')
            plt.ylabel('Released\nAGN Energy\n(10$^{61}$ erg)')
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            # plt.axhline(-2, color='gray', lw=0.5)
            # plt.ylim(-4, 0.05)
        if xlim is None:
            xlim = [None, None]
        if xlim[0] is None:
            xlim[0] = np.min(xarr)
        if xlim[1] is None:
            xlim[1] = np.max(xarr)
        plt.xlim(xlim)

        for vline in vlines:
            plt.axvline(vline, color='k', lw=0.5)
    if (savefile is not None):
        painter.save_figure(savefile)


def edd_acc_rate(mbh, epsilon):
    # eddington accretion rate
    return 4 * np.pi * G_const * mbh * m_H / (epsilon * c_const * sigma_T)


def eps_spin(spinmag):
    # epsilon from spin magnitude (implementation on RAMSES)
    ZZ1 = 1. + (1. - spinmag ** 2) ** (1. / 3.) * ((1. + spinmag)) ** (1. / 3.) + (1. - spinmag ** (1. / 3.))
    ZZ2 = np.sqrt(3. * spinmag ** 2. + ZZ1 ** 2.)
    r_lso = np.select([spinmag > 0., True], [3. + ZZ2 - np.sqrt((3. - ZZ1) * (3. + ZZ1 + 2. * ZZ2)),
                                             3. + ZZ2 + np.sqrt((3. - ZZ1) * (3. + ZZ1 + 2. * ZZ2))])
    return 1. - np.sqrt(1. - 2. / (3. * r_lso))


def eff_mad(spinmag):
    # energy efficiency in MAD model
    # Fourth-order polynomial fit to the spinup parameters of McKinney et al, 2012
    eff_mad = (4.10507 + 0.328712 * spinmag + 76.0849 * spinmag ** 2
               + 47.9235 * spinmag ** 3 + 3.86634 * spinmag ** 4) / 100.
    return eff_mad
