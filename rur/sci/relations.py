import numpy as np
from scipy.interpolate import interp2d

@staticmethod
class mgal_mbh:
    params_dict = {
        'RV15a': [1.05, 1E11, 7.45], # Reines & Volonteri 2015 (AGNs)
        'RV15b': [1.40, 1E11, 8.95], # Reines & Volonteri 2015 (E/CBs)
        'BM19':  [1.64, 1E11, 7.88], # Baron & Ménard 2019 (All?)
    }
    def evaluate(self, tag):
        params = self.params_dict[tag]
        return lambda mgal: 10**(params[0]*np.log10(params[1])+params[2])

    __call__ = evaluate

class mgal_mdmh:
    def guo(m, c=0.129, m_0=10 ** 11.4, alpha=0.926, beta=0.261, gamma=2.440):
        # from Guo et al. (2010)
        return c * ((m / m_0) ** -alpha + (m / m_0) ** beta) ** (-gamma)

    def moster(m, z=0.):
        #  Stellar-to-halo mass relation from Moster et al. (2010)
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
        #  Stellar-to-halo mass relation from Behroozi et al. (2013)
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

class mgal_size:

    def V14(self, mlog, z=0, shape='circular', radius_type=1, gal_type='late', return_error=False): # van der Wel+ 2014 (SDSS)
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
            data = f(mlog, z)
            out = data[:, 0]
            err = data[:, 1]

            if(return_error):
                return out, err
            else:
                return out

class mgal_sfr:
    @staticmethod
    def W14(logm, z=0): # Whitaker+ 2014
        alpha = 0.70 - 0.13 * z
        beta = 0.38 + 1.14 * z - 0.19 * z**2
        log_sfr = alpha * (logm - 10.5) + beta
        return log_sfr
