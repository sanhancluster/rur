import numpy as np
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
import warnings

def check_range(x, range):
    if np.any(x < range[0]) or np.any(x > range[1]):
        warnings.warn("Value exceeded allowed range: %s" % str(range), UserWarning)

def mgal_mbh(tag, logscale=False):
    # galaxy-BH mass relation
    try:
        params_dict = {
            'RV15a': [1.05, 1E11, 7.45],  # Reines & Volonteri 2015 (AGNs)
            'RV15b': [1.40, 1E11, 8.95],  # Reines & Volonteri 2015 (E/CBs)
            'BM19': [1.64, 1E11, 7.88],  # Baron & Ménard 2019 (All?)
            'S20a': [1.64, 1, -10.29],  # Suh+ 2020 (All)
            'S20b': [0.78, 1, -0.33],  # Suh+ 2020 (High-z AGNs)
            'S20c': [1.473, 1E11,  7.620],  # Suh+ 2020 (All AGNs), derived from Figure 5
            'Z23': [0.98, 1, -1.92],  # Zhang+ 2023 (z~2 Type I AGNs)
            'KH13b': [1.19, 1E11, np.log10(0.49*1E9)],  # Kormendy & Ho 2013 (E/CBs)
        }
    except KeyError:
        raise ValueError("Available tags: RV15a, RV15b, BM19, S20a, S20b")
    params = params_dict[tag]

    if not logscale:
        return lambda mgal: 10**(params[0]*np.log10(mgal/params[1])+params[2])
    else:
        return lambda mgal: params[0]*np.log10(10**mgal/params[1])+params[2]

def sigma_mbh(tag, logscale=False):
    # sigma*-BH mass relation
    params_dict = {
        'G09': [8.12, 4.24],  # Gultekin 09
        'K13': [8.49, 4.377],  # Kormendy 13
        'M13': [8.32, 5.64],  # McConnel& Ma 13
    }

    def msig(sig, params):
        if logscale:
            sig = 10**sig
        mbh = params[0] + params[1] * np.log10(sig / 200)
        if not logscale:
            mbh = 10**mbh
        return mbh

    def robertson(sig, z=0):
        # Robertson+ 06
        zarr = [0, 2, 3, 6]
        params = [[8.01, 7.83, 7.72, 7.44], [3.87, 4.10, 4.02, 3.62]]
        interp = interp1d(zarr, np.array(params))
        params = interp(z)
        return msig(sig, params)

    def shen(sig, z=0):
        # Shen+ 15
        zarr = [0.26, 0.53, 0.70, 0.84]
        params = [[8.324, 8.372, 8.388, 8.364], [1.695, 1.592, 1.199, 0.960]]
        interp = interp1d(zarr, np.array(params))
        params = interp(z)
        return msig(sig, params)


    if tag == 'R06':
        return robertson
    if tag == 'S15':
        return shen
    else:
        params = params_dict[tag]
        return lambda sig: msig(sig, params)

def mgal_mdmh(tag):
    # stellar to halo mass ratio
    def guo(m, c=0.129, m_0=10 ** 11.4, alpha=0.926, beta=0.261, gamma=2.440):
        # Guo+ 10
        return c * ((m / m_0) ** -alpha + (m / m_0) ** beta) ** (-gamma)

    def moster(m, z=0.):
        #   Moster+ 10
        table = np.array(
            [
                (0.0, 11.88, 0.02, 0.0282, 0.0005, 1.06, 0.05, 0.05, 0.56, 0.00),
                (0.5, 11.95, 0.24, 0.0254, 0.0047, 1.37, 0.22, 0.27, 0.55, 0.17),
                (0.7, 11.93, 0.23, 0.0215, 0.0048, 1.18, 0.23, 0.28, 0.48, 0.16),
                (0.9, 11.98, 0.24, 0.0142, 0.0034, 0.91, 0.16, 0.19, 0.43, 0.12),
                (1.1, 12.05, 0.18, 0.0175, 0.0060, 1.66, 0.26, 0.31, 0.52, 0.40),
                (1.5, 12.15, 0.30, 0.0110, 0.0044, 1.29, 0.25, 0.32, 0.41, 0.41),
                (1.8, 12.28, 0.27, 0.0116, 0.0051, 1.53, 0.33, 0.41, 0.41, 0.41),
                (2.5, 12.22, 0.38, 0.0130, 0.0037, 0.90, 0.20, 0.24, 0.30, 0.30),
                (3.5, 12.21, 0.19, 0.0101, 0.0020, 0.82, 0.72, 1.16, 0.46, 0.21),
            ], dtype=[('z', 'f8'), ('m1', 'f8'), ('m1err', 'f8'), ('mm0', 'f8'), ('mm0err', 'f8'),
                      ('beta', 'f8'), ('betaep', 'f8'), ('betaem', 'f8'), ('gamma', 'f8'), ('gammaerr', 'f8')])

        m1 = 10. ** np.interp(z, table['z'], table['m1'])
        mm0 = np.interp(z, table['z'], table['mm0'])
        beta = np.interp(z, table['z'], table['beta'])
        gamma = np.interp(z, table['z'], table['gamma'])

        return 2 * mm0 * ((m / m1) ** -beta + (m / m1) ** gamma) ** -1

    def behroozi(mh, z):
        #  Behroozi+ 13
        exp = np.exp
        logmh = np.log10(mh)
        a = 1. / (1. + z)

        nu = exp(-4 * a ** 2)

        logeps = -1.777 + (-0.006 * (a - 1)) * nu - 0.119 * (a - 1)
        logm1 = 11.514 + (-1.793 * (a - 1) - 0.251 * z) * nu

        alpha = -1.412 + (0.731 * (a - 1)) * nu
        delta = 3.508 + (2.608 * (a - 1) - 0.043 * z) * nu
        gamma = 0.316 + (1.319 * (a - 1) + 0.279 * z) * nu

        f = lambda x: -np.log10(10 ** (alpha * x) + 1) + delta * np.log10(1 + exp(x)) ** gamma / (1 + exp(10 ** -x))

        logmg = logeps + logm1 + f(logmh - logm1) - f(0)

        return 10. ** (logmg - logmh)

    if tag == 'B13':
        return behroozi
    elif tag == 'M10':
        return moster
    elif tag == 'G10':
        return guo
    else:
        raise ValueError("Available tags: B13 (Behroozi+ 13), M10 (Moster+ 10), G10 (Guo+ 10)")


def mgal_size(tag='V14', logscale=False):
    # mass-size relation
    def V14(m, z=0, shape='circular', radius_type=1, gal_type='late', return_error=False): # van der Wel+ 2014 (SDSS)
        if logscale:
            logm = m
        else:
            logm = np.log10(m)
        if(shape == 'circular'):
            table_16 = [
                [[-0.07,  0.08], [-0.05,  0.04], [ 0.02,  0.03], [ 0.29,  0.04], [np.nan, np.nan], [ 0.07,  0.01], [ 0.21,  0.02], [ 0.27,  0.02], [ 0.51,  0.03], [np.nan, np.nan]],
                [[-0.11,  0.04], [-0.25,  0.02], [-0.11,  0.02], [ 0.11,  0.02], [ 0.57,  0.02], [ 0.04,  0.01], [ 0.17,  0.01], [ 0.27,  0.01], [ 0.39,  0.01], [ 0.64,  0.04]],
                [[np.nan, np.nan], [-0.21,  0.02], [-0.23,  0.01], [-0.07,  0.02], [ 0.32,  0.04], [-0.05,  0.01], [ 0.1 ,  0.01], [ 0.23,  0.01], [ 0.36,  0.01], [ 0.51,  0.01]],
                [[np.nan, np.nan], [-0.12,  0.06], [-0.36,  0.02], [-0.14,  0.02], [ 0.11,  0.06], [-0.09,  0.01], [ 0.02,  0.01], [ 0.16,  0.01], [ 0.24,  0.02], [ 0.41,  0.03]],
                [[np.nan, np.nan], [np.nan, np.nan], [-0.53,  0.09], [-0.34,  0.03], [ 0.02,  0.05], [np.nan, np.nan], [-0.06,  0.01], [ 0.06,  0.02], [ 0.16,  0.02], [ 0.29,  0.04]],
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [-0.37,  0.05], [-0.08,  0.11], [np.nan, np.nan], [np.nan, np.nan], [ 0.01,  0.02], [ 0.08,  0.05], [ 0.24,  0.03]]]
            table_16 = np.array(table_16)

            table_50 = [
                [[ 0.17,  0.01], [ 0.17,  0.03], [ 0.24,  0.04], [ 0.57,  0.03], [ 0.72,  0.07], [ 0.3 ,  0.01], [ 0.42,  0.01], [ 0.49,  0.03], [ 0.7 ,  0.02], [ 0.91,  0.04]],
                [[ 0.12,  0.02], [ 0.1 ,  0.02], [ 0.13,  0.02], [ 0.36,  0.01], [ 0.74,  0.03], [ 0.26,  0.  ], [ 0.39,  0.01], [ 0.49,  0.01], [ 0.61,  0.02], [ 0.82,  0.03]],
                [[np.nan, np.nan], [ 0.06,  0.03], [-0.01,  0.02], [ 0.16,  0.02], [ 0.49,  0.03], [ 0.19,  0.  ], [ 0.32,  0.  ], [ 0.42,  0.  ], [ 0.53,  0.01], [ 0.67,  0.02]],
                [[np.nan, np.nan], [ 0.12,  0.04], [-0.05,  0.04], [ 0.08,  0.02], [ 0.37,  0.03], [ 0.15,  0.  ], [ 0.24,  0.01], [ 0.38,  0.01], [ 0.47,  0.01], [ 0.58,  0.02]],
                [[np.nan, np.nan], [np.nan, np.nan], [-0.12,  0.05], [-0.02,  0.03], [ 0.25,  0.05], [np.nan, np.nan], [ 0.17,  0.01], [ 0.29,  0.01], [ 0.39,  0.01], [ 0.52,  0.02]],
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [-0.01,  0.04], [ 0.29,  0.09], [np.nan, np.nan], [np.nan, np.nan], [ 0.27,  0.01], [ 0.34,  0.01], [ 0.43,  0.02]]]
            table_50 = np.array(table_50)

            table_84 = [
                [[ 0.31,  0.03], [ 0.39,  0.05], [ 0.48,  0.04], [ 0.84,  0.05], [ 1.01,  0.11], [ 0.54,  0.01], [ 0.62,  0.02], [ 0.73,  0.02], [ 0.86,  0.05], [ 1.06,  0.06]],
                [[ 0.34,  0.02], [ 0.34,  0.03], [ 0.33,  0.01], [ 0.56,  0.01], [ 0.93,  0.04], [ 0.47,  0.01], [ 0.58,  0.01], [ 0.66,  0.01], [ 0.78,  0.01], [ 0.98,  0.02]],
                [[np.nan, np.nan], [ 0.33,  0.03], [ 0.27,  0.03], [ 0.44,  0.03], [ 0.75,  0.03], [ 0.41,  0.01], [ 0.51,  0.01], [ 0.6 ,  0.01], [ 0.69,  0.01], [ 0.83,  0.01]],
                [[np.nan, np.nan], [ 0.36,  0.04], [ 0.27,  0.04], [ 0.42,  0.03], [ 0.67,  0.03], [ 0.36,  0.01], [ 0.46,  0.01], [ 0.55,  0.01], [ 0.64,  0.01], [ 0.75,  0.02]],
                [[np.nan, np.nan], [np.nan, np.nan], [ 0.25,  0.03], [ 0.41,  0.06], [ 0.72,  0.12], [np.nan, np.nan], [ 0.38,  0.01], [ 0.49,  0.01], [ 0.56,  0.01], [ 0.7 ,  0.02]],
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [ 0.4 ,  0.06], [ 0.62,  0.1 ], [np.nan, np.nan], [np.nan, np.nan], [ 0.48,  0.02], [ 0.54,  0.02], [ 0.63,  0.06]]]
            table_84 = np.array(table_84)

            tables = [table_16, table_50, table_84]
            table = tables[radius_type]

            if(gal_type == 'late'):
                table = table[:, 5:]
            else:
                table = table[:, :5]

            zgrid = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
            mgrid = [9.25, 9.75, 10.25, 10.75, 11.25]
            f = interp2d(zgrid, mgrid, table)
            data = f(logm, z)
            out = data[:, 0]
            err = data[:, 1]

            if not logscale:
                out = 10**out

            return out

    def M19(m, z): # Mowla+ 19
        check_range(z, [0.37, 2.69])
        if logscale:
            m = 10**m
        zarr = [0.37, 0.79, 1.24, 1.72, 2.24, 2.69]
        table = np.array([
            (3.8, 10.3, 0.09, 0.37),
            (4.0, 10.7, 0.10, 0.45),
            (4.2, 11.1, 0.13, 0.53),
            (3.7, 11.1, 0.11, 0.50),
            (3.1, 11.0, 0.11, 0.42),
            (2.8, 10.9, 0.06, 0.38),])

        # Create an interpolator for each column
        interp_funcs = [interp1d(zarr, table[:, i], kind='linear', fill_value="extrapolate") for i in range(table.shape[1])]

        # Evaluate at z
        rp, logMp, alpha, beta = tuple(f(z) for f in interp_funcs)
        delta = 6

        re = rp * (m/10**logMp)**alpha * (0.5*(1 + (m/10**logMp)**delta))**((beta - alpha)/delta)

        if logscale:
            re = np.log10(re)
        return re

    def N21(m, z, type='all'): # Nedkova+ 21
        check_range(z, [0, 2.0])
        if logscale:
            m = 10**m
        if type == 'all':
            params = np.select([
                (0.2 <= z) & (z <= 0.5),
                (0.5 < z) & (z <= 1.0),
                (1.0 < z) & (z <= 1.5),
                (1.5 < z) & (z <= 2.0)],
                [(0.04, 1.82, -0.24, 10.94),
                 (-0.03, 1.60, 0.48, 10.95),
                 (-0.33, 1.25, 3.24, 10.63),
                 (-0.17, 1.84, 1.76, 11.09)],)
            alpha, beta, loggamma, delta = tuple(np.array(params).T)
            re = lambda m: 10**loggamma * m**alpha * (1+m/10**delta) ** (beta - alpha)
        elif type == 'high':
            params = np.select([
                (0.2 <= z) & (z <= 0.5),
                (0.5 < z) & (z <= 1.0),
                (1.0 < z) & (z <= 1.5),
                (1.5 < z) & (z <= 2.0)],
                [(0.61, 1.82),
                 (0.45, 0.64),
                 (0.29, 0.63),
                 (0.18, 0.61)],)
            A, B = tuple(np.array(params).T)
            re = lambda m: A * (m/5E10)**B
        out = re(m)
        if logscale:
            out = np.log10(out)
        return out

    if tag == 'V14':
        return V14
    if tag == 'M19':
        return M19
    elif tag == 'N21':
        return N21

def mgal_sfr(tag='W12'):
    """
    Star formation main sequence. Each function returns log(SFR) given log(M*).
    """
    def W12(logm, z=0): # Whitaker+ 12
        alpha = 0.70 - 0.13 * z
        beta = 0.38 + 1.14 * z - 0.19 * z**2
        log_sfr = alpha * (logm - 10.5) + beta
        return log_sfr

    def W14(logm, z=0): # Whitaker+ 14
        params_arr = np.array([
            [-27.40, -26.03, -24.04, -19.99],
            [5.02, 4.62, 4.17, 3.44],
            [-0.22, -0.19, -0.16, -0.13],
        ])
        z_arr = [0.75, 1.25, 1.75, 2.25]

        table = np.rec.fromarrays([
        ], dtype=[('z', 'f8'), ('a', 'f8'),
                  ('b', 'f8'), ('c', 'f8')])

        a = np.interp(z, table['z'], params_arr[0])
        b = np.interp(z, table['z'], params_arr[1])
        c = np.interp(z, table['z'], params_arr[2])
        return a + b * logm + c * logm**2
    
    def S14(logm, age):
        # Speagle+ 14
        # log SFR(M∗,t) = (0.84 ± 0.02 − 0.026 ± 0.003 × t) log M∗−(6.51 ± 0.24 − 0.11 ± 0.03 × t)
        a = 0.84 - 0.026 * age
        b = -6.51 + 0.11 * age
        return a * logm + b

    if tag == 'W12':
        return W12
    elif tag == 'W14':
        return W14
    elif tag == 'S14':
        return S14


def gmf():
    # galaxy mass function
    def schechter(m, ms, phi, alpha):
        return np.log(10) * phi * 10**((m-ms) * (1+alpha)) * np.exp(-10**(m - ms))


def gpm(m, params):
    z0, m0, gamma = params
    return z0+np.log10(1-10**(-(m/m0)**gamma))

def mzr_12OH(tag):
    # mass-metallicity relation
    def T04(logm): # Tremonti+ 04, z=0
        return -1.492 + 1.847*logm - 0.08026*logm**2

    def E06(logm): # Erb+ 06
        mzr_E06 = np.rec.fromarrays([np.array([0.71, 1.5, 2.6, 4.1, 10.5]) * 1E10,
                                     np.array([0.17, 0.3, 0.4, 0.6, 5.4]) * 1E10,
                                     [8.33, 8.42, 8.46, 8.52, 8.58],
                                     [0.07, 0.06, 0.06, 0.06, 0.06],
                                     [0.07, 0.05, 0.05, 0.05, 0.04]],
                                    dtype=[('m', 'f8'), ('merr', 'f8'),
                                           ('OH12', 'f8'), ('OH12uerr', 'f8'), ('OH12lerr', 'f8')])
        return np.interp(mzr_E06['m'], 10**logm, mzr_E06['OH12'])

    def S14(logm):
        mzr_S14 = np.rec.fromarrays([[8.87, 9.34, 9.69, 9.87, 10.11, 10.37, 10.66, 11.19],
                                     [8.20, 8.23, 8.31, 8.35, 8.38, 8.47, 8.51, 8.65]],
                                    dtype=[('m', 'f8'), ('OH12', 'f8')])
        return np.interp(mzr_S14['m'], 10 ** logm, mzr_S14['OH12'])

    def S06(logm, t_H): # Salvaglio+ 06
        dlogm = -2.0436 * np.log10(t_H) + 2.2223
        return -2.4412 + 2.1026 * (logm - dlogm) - 0.09649 * (logm - dlogm)**2

    def W14(logm, z): # Wuyts+ 14
        check_range(z, [0., 2.3])
        def gpm(m, params):
            z0, m0, gamma = params
            return z0 + np.log10(1 - 10 ** (-(m / m0) ** gamma))

        table = [[8.69, 8.8, 8.7], [9.02, 10.2, 10.5], [0.4, 0.4, 0.5]]
        zarr = [0, 0.9, 2.3]
        z0 = np.interp(z, zarr, table[0])
        m0 = 10**np.interp(z, zarr, table[1])
        gamma = np.interp(z, zarr, table[2])

        return gpm(10**logm, (z0, m0, gamma))

    def S21(logm, z): # Sanders+ 21
        check_range(z, [2.3, 3.3])
        check_range(logm, [9., 11.])
        table = np.array(
            [(2.3, 0.30, 8.51),
             (3.3, 0.29, 8.41)],
            dtype=[('z', 'f8'), ('gamma', 'f8'), ('Z10', 'f8')]
        )
        gamma = np.interp(z, table['z'], table['gamma'])
        Z10 = np.interp(z, table['z'], table['Z10'])
        return gamma * (logm-10) + Z10


    if tag == 'T04':
        return T04
    elif tag == 'S14':
        return S14
    elif tag == 'E06':
        return E06
    elif tag == 'S06':
        return S06
    elif tag == 'W14':
        return W14
    elif tag == 'S21':
        return S21
