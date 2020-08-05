from scipy.spatial import cKDTree as Tree

from scipy.spatial.distance import cdist
import numpy as np
from scipy.linalg import expm
from scipy.stats import norm
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from numpy.linalg import det

import warnings

import pickle as pkl
import os
import time
from numpy.linalg import inv


def type_of_script():
    """
    Detects and returns the type of python kernel
    :return: string 'jupyter' or 'ipython' or 'terminal'
    """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'


class Table:
    """
    Table class to store RAMSES particle/AMR data.
    Basically acts as numpy.recarray, but some functions do not work.
    """
    def __init__(self, table, snap):
        self.table = table
        self.snap = snap

    def __getitem__(self, item):
        pass

    def __str__(self):
        return self.table.__str__()
    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.__dict__)
    def __getattr__(self, item):
        return self.table.__getattribute__(item)
    def __setitem__(self, key, value):
        return self.table.__setitem__(key, value)

# pickle load/save
def uopen(path, mode):
    # create directory if there's no one
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode)

def dump(data, path, msg=True, protocol=4):
    t = Timer()
    path = os.path.expanduser(path)

    with uopen(path, 'wb') as opened:
        pkl.dump(data, opened, protocol=protocol)
    if(msg):
        filesize = os.path.getsize(path)
        print("File %s dump complete (%s): %.3f seconds elapsed" % (path, format_bytes(filesize), t.time()))

def load(path, msg=True):
    t = Timer()
    path = os.path.expanduser(path)
    with open(path, 'rb') as opened:
        data = pkl.load(opened, encoding='latin1')
    if(msg):
        filesize = os.path.getsize(path)
        print("File %s load complete (%s): %.3f seconds elapsed" % (path, format_bytes(filesize), t.time()))
    return data


# Some RAMSES-related stuff
dim_keys = ['x', 'y', 'z']

def los(proj, ndim=3):
    # return line-of-sight dimension from given projection
    if(len(proj) != ndim-1):
        raise ValueError('Invalid projection')
    dims = np.arange(ndim)
    return dims[np.isin(dims, proj, invert=True)][0]

def get_vector(table, prefix='', ndim=3):
    return np.stack([table[prefix + key] for key in dim_keys[:ndim]], axis=-1)

def set_vector(table, vector, prefix='', ndim=3, where=None, copy=False):
    if(isinstance(table, Table)):
        table = table.table
    if(copy):
        table = table.copy()
    for idim in np.arange(ndim):
        if(where is None):
            table[prefix+dim_keys[idim]] = vector[..., idim]
        else:
            table[prefix+dim_keys[idim]][where] = vector[..., idim]
    if(copy):
        return table

def get_box(center, extent):
    center = np.array(center)
    if(not np.isscalar(extent)):
        extent = np.array(extent)
    return np.stack([center-extent/2, center+extent/2], axis=-1)

def get_bounding_box(coo):
    # return bounding box from coordinates
    coo = np.array(coo)
    ndim = coo.shape[-1]
    bbox = ([np.min(coo, axis=0), np.max(coo, axis=0)])


def get_distance(a, b, prefix='', ndim=3):
    return rss(get_vector(a, prefix, ndim) - get_vector(b, prefix, ndim))

def get_polar_coord(coo, pos=[0.5, 0.5, 0.5]):
    coo = coo-np.array(pos)
    r = rss(coo, axis=-1)
    theta = np.arccos(coo[:, 2]/r)
    phi = np.arctan2(coo[:, 1], coo[:, 0])

    return np.stack([r, theta, phi], axis=-1)


def shift(table, vec, ndim=3, periodic=True):
    if(isinstance(table, Table)):
        table = table.table
    for idim in np.arange(ndim):
        table[dim_keys[idim]] += vec[idim]
        if(periodic):
            table[dim_keys[idim]] %= 1

def rss(coo, axis=-1):
    # root sum square
    return np.sqrt(ss(coo, axis))

def ss(coo, axis=-1):
    # square sum
    return np.sum(coo ** 2, axis=axis)

def rms(coo, axis=-1):
    # root mean square
    return np.sqrt(np.mean(coo**2, axis=axis))


def pairing(a, b, ignore=None):
    # cantor pairing function
    output = (a + b) * (a + b + 1) // 2 + b
    if(ignore is not None):
        mask = (a == ignore) | (b == ignore)
        output[mask] = ignore
    return output

# some custom array-related functions
def bin_centers(start, end, num):
    # get center of num bins that divide the range equally
    arr = np.linspace(start, end, num+1)
    return (arr[1:] + arr[:-1]) / 2

def append_rows(array, new_rows, idx=None):
    # Calculate the number of old and new rows
    len_array = array.shape[0]
    len_new_rows = new_rows.shape[0]
    # Resize the old recarray
    array.resize(len_array + len_new_rows, refcheck=False)
    # Write to the end of recarray

    if(idx is None):
        array[-len_new_rows:] = new_rows
    else:
        array[idx-len_array:] = array[idx:len_array]
        array[idx:idx+len_new_rows] = new_rows
    return array

def rank(arr):
    temp = np.argsort(arr)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(arr))
    return ranks

def expand_shape(arr, axes, ndim):
    # axis: array that has same size with current number of dimensions, positions of axis in the resulting array
    # ndim: number of dimensions of resulting array
    axes = np.array(axes)
    arr = np.array(arr)
    if (arr.ndim != axes.size):
        raise ValueError("Invalid axes, make sure arr.ndim == axes.size")

    sources = np.arange(arr.ndim)
    dests = rank(axes)
    new_arr = np.moveaxis(arr, sources, dests)

    newshape = np.full(ndim, 1, dtype=np.int)
    newshape[np.array(axes)] = np.array(arr.shape)[np.arange(axes.size)]

    return np.reshape(new_arr, newshape)


def make_broadcastable(arr_tuple, axes, ndim=None):
    # makes two arrays to be able to broadcast with each other
    if(ndim is None):
        axes_all = np.concatenate(axes)
        ndim = max(axes_all) + 1

    out = ()
    for arr, axis in zip(arr_tuple, axes):
        out += (expand_shape(arr, axis, ndim), )
    return out

def find_cross(x1, y1, x2, y2):
    """
    :param x1, y1: ndarray of coordinates 1, larger at initial
    :param x2, y2: ndarray of coordnitaes 2, smaller at initial
    :return: first crossing point of two graph
    """
    res = y2 - np.interp(x2, x1, y1)
    mask = res > 0
    if(np.all(mask)):
        mean_ini = np.mean([[x1[0], y1[0]], [x2[0], y2[0]]], axis=0)
        return tuple(mean_ini)
    if(np.all(~mask)):
        last_ini = np.mean([[x1[-1], y1[-1]], [x2[-1], y2[-1]]], axis=0)
        return tuple(last_ini)

    idx = np.argwhere(mask)[0]
    coo = intersection(
        np.array([x1[idx - 1], y1[idx - 1]]),
        np.array([x1[idx], y1[idx]]),
        np.array([x2[idx - 1], y2[idx - 1]]),
        np.array([x2[idx], y2[idx]]),
    )

    return tuple(coo)

def intersection(x1, x2, x3, x4):
    a = x2 - x1
    b = x4 - x3
    c = x3 - x1

    x = x1 + a * np.dot(np.cross(c, b), np.cross(a, b)) / ss(np.cross(a, b))
    return x

def bin_cut(arr, value, bins, return_centers=False, func=None):
    out = []
    binarr = bins[:-1], bins[1:]
    for bot, top in zip(*binarr):
        cut = arr[(bot<=value) & (value<top)]
        if(func is None):
            out.append(cut)
        else:
            out.append(func(cut))
    if(not return_centers):
        return out
    else:
        center = np.average(binarr, axis=0)
        return out, center

def cartesian(*arrays):
    # cartesian product of arrays
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def discrete_hist2d(shape, hid_arr, use_long=False):
    if (use_long):
        hid_arr = hid_arr.astype('i8')
    idxs = hid_arr[0] * shape[1] + hid_arr[1]
    hist = np.bincount(idxs, minlength=shape[0] * shape[1])
    hist = np.reshape(hist, shape)
    return hist

def format_bytes(size, format='{:#.4g}'):
    power = 1024
    n = 0
    power_labels = {0 : '', 1: 'Ki', 2: 'Mi', 3: 'Gi', 4: 'Ti', 5: 'Pi'}
    while size > power:
        size /= power
        n += 1
    return (format.format(size)) + ' ' + power_labels[n]+'B'

#not working for now
def ugenfromtxt(fname, comments='#', delimeter=' ', skip_header=0, dtype_int='i4', dtype_float='f8', *kwargs):
    with open(fname) as fi:
        fi.readline(skip_header)
        while(True):
            line = fi.readline()
            if(not line.startswith('#')):
                break
        elements = line.split(delimeter)
        dtype = []
        for element in elements:
            if(isint(element)):
                dtype.append(dtype_int)
            elif(isfloat(element)):
                dtype.append(dtype_float)
            else:
                dtype.append('U%d' % len(element))
    print(dtype)

    np.genfromtxt(fname, dtype=dtype, comments=comments, delimiter=delimeter, skip_header=skip_header, names=True, *kwargs)

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def rot_euler(v, xyz):
    ''' Rotate vector v (or array of vectors) by the euler angles xyz '''
    # https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    for theta, axis in zip(xyz, np.eye(3)):
        v = np.dot(np.array(v), expm(np.cross(np.eye(3), axis*-theta)))
    return v

def projection(data, normal, origin=np.array([0., 0., 0.]), prefix=''):
    """
    :param pos: coordinates of original points
    :param normal: normal vector
    :return: rotated coordinate
    """
    nx, ny, nz = tuple(normal)
    alpha = np.arccos(-ny/np.sqrt(1.-nz**2))
    beta = np.arccos(nz)
    gamma = 0.
    pos = get_vector(data, prefix) - origin
    pos_r = rotate_vector(pos, normal) + origin
    #pos_r = np.dot(rotation_matrix([alpha, beta, gamma]), pos) + origin
    data.table['x'] = pos_r[:, 0]
    data.table['y'] = pos_r[:, 1]
    data.table['z'] = pos_r[:, 2]

    return data

def rotation_matrix(angles):
    R_zz = np.array([[np.cos(angles[0]), np.sin(angles[0]), 0],
                    [-np.sin(angles[0]), np.cos(angles[0]), 0],
                    [0, 0, 1]
                    ])

    R_y = np.array([[np.cos(angles[1]), 0, -np.sin(angles[1])],
                    [0, 1, 0],
                    [np.sin(angles[1]), 0, np.cos(angles[1])]
                    ])

    R_z = np.array([[np.cos(angles[2]), np.sin(angles[2]), 0],
                    [-np.sin(angles[2]), np.cos(angles[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_zz, np.dot(R_y, R_z))

    return R

def rotate_vector(r, J):
    # Thank to MJ
    r = np.array(r)
    z = J/np.sqrt(sum(J*J))
    x = np.cross(np.array([0,0,1]),z)
    x = x/np.sqrt(sum(x*x))
    y = np.cross(z,x)
    y = y/np.sqrt(sum(y*y))
    rotate = np.vstack((x,y,z)).T
    rotate = np.matrix(rotate)
    rotate = inv(rotate)
    R = (rotate*r.T).T

    return np.array(R)

class dtfe(object):

    def __init__(self, dataset, weights=None, smooth=0):
        self.dataset = np.atleast_2d(dataset).T
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        self.n, self.d = self.dataset.shape
        #center = np.median(self.dataset, axis=0)
        center=0

        self.tri=Delaunay(self.dataset-center, qhull_options='')

        simplices = self.tri.simplices
        vertices = self.dataset[simplices]

        matrices = np.insert(vertices, 2, 1., axis=-1)
        matrices = np.swapaxes(matrices, -1, -2)
        tri_areas = np.abs(det(matrices)) / np.math.factorial(self.d+1)

        hull_areas = np.zeros(self.n, dtype='f8')
        np.add.at(hull_areas, simplices, np.expand_dims(tri_areas, -1))

        if (weights is not None):
            hull_areas /= weights

        if (smooth > 0):
            indptr, neighbor_indices = self.tri.vertex_neighbor_vertices
            neighbor_nums = np.full(self.n, 0, dtype='i8')
            center_indices = np.repeat(np.arange(neighbor_nums.size), np.diff(indptr))
            np.add.at(neighbor_nums, center_indices, 1)

            for _ in np.arange(smooth):
                hull_areas_new = np.zeros(hull_areas.shape, dtype='f8')
                np.add.at(hull_areas_new, center_indices, hull_areas[neighbor_indices])
                hull_areas = hull_areas_new / neighbor_nums

        densities = 1 / hull_areas

        self.lip = LinearNDInterpolator(self.tri, densities, fill_value=0)

    def evaluate(self, points):
        return self.lip(*points)

    __call__ = evaluate


class gaussian_kde(object):
    """Representation of a kernel-density estimate using Gaussian kernels.

    Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.
    `gaussian_kde` works for both uni-variate and multi-variate data.   It
    includes automatic bandwidth determination.  The estimation works best for
    a unimodal distribution; bimodal or multi-modal distributions tend to be
    oversmoothed.

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of dims, # of data).
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `gaussian_kde` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See Notes for more details.
    weights : array_like, shape (n, ), optional, default: None
        An array of weights, of the same shape as `x`.  Each value in `x`
        only contributes its associated weight towards the bin count
        (instead of 1).

    Attributes
    ----------
    dataset : ndarray
        The dataset with which `gaussian_kde` was initialized.
    d : int
        Number of dimensions.
    n : int
        Number of datapoints.
    neff : float
        Effective sample size using Kish's approximation.
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of `covariance`.

    Methods
    -------
    kde.evaluate(points) : ndarray
        Evaluate the estimated pdf on a provided set of points.
    kde(points) : ndarray
        Same as kde.evaluate(points)
    kde.pdf(points) : ndarray
        Alias for ``kde.evaluate(points)``.
    kde.set_bandwidth(bw_method='scott') : None
        Computes the bandwidth, i.e. the coefficient that multiplies the data
        covariance matrix to obtain the kernel covariance matrix.
        .. versionadded:: 0.11.0
    kde.covariance_factor : float
        Computes the coefficient (`kde.factor`) that multiplies the data
        covariance matrix to obtain the kernel covariance matrix.
        The default is `scotts_factor`.  A subclass can overwrite this method
        to provide a different method, or set it through a call to
        `kde.set_bandwidth`.

    Notes
    -----
    Bandwidth selection strongly influences the estimate obtained from the KDE
    (much more so than the actual shape of the kernel).  Bandwidth selection
    can be done by a "rule of thumb", by cross-validation, by "plug-in
    methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
    uses a rule of thumb, the default is Scott's Rule.

    Scott's Rule [1]_, implemented as `scotts_factor`, is::

        n**(-1./(d+4)),

    with ``n`` the number of data points and ``d`` the number of dimensions.
    Silverman's Rule [2]_, implemented as `silverman_factor`, is::

        (n * (d + 2) / 4.)**(-1. / (d + 4)).

    Good general descriptions of kernel density estimation can be found in [1]_
    and [2]_, the mathematics for this multi-dimensional implementation can be
    found in [1]_.

    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
           conditional density estimation", Computational Statistics & Data
           Analysis, Vol. 36, pp. 279-298, 2001.

    Examples
    --------
    Generate some random two-dimensional data:

    >>> from scipy import stats
    >>> def measure(n):
    >>>     "Measurement model, return two coupled measurements."
    >>>     m1 = np.random.normal(size=n)
    >>>     m2 = np.random.normal(scale=0.5, size=n)
    >>>     return m1+m2, m1-m2

    >>> m1, m2 = measure(2000)
    >>> xmin = m1.min()
    >>> xmax = m1.max()
    >>> ymin = m2.min()
    >>> ymax = m2.max()

    Perform a kernel density estimate on the data:

    >>> X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    >>> positions = np.vstack([X.ravel(), Y.ravel()])
    >>> values = np.vstack([m1, m2])
    >>> kernel = stats.gaussian_kde(values)
    >>> Z = np.reshape(kernel(positions).T, X.shape)

    Plot the results:

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
    ...           extent=[xmin, xmax, ymin, ymax])
    >>> ax.plot(m1, m2, 'k.', markersize=2)
    >>> ax.set_xlim([xmin, xmax])
    >>> ax.set_ylim([ymin, ymax])
    >>> plt.show()

    """

    def __init__(self, dataset, bw_method=None, weights=None):
        self.dataset = np.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        self.d, self.n = self.dataset.shape

        if weights is not None:
            self.weights = weights / np.sum(weights)
        else:
            self.weights = np.ones(self.n) / self.n

        # Compute the effective sample size
        # http://surveyanalysis.org/wiki/Design_Effects_and_Effective_Sample_Size#Kish.27s_approximate_formula_for_computing_effective_sample_size
        self.neff = 1.0 / np.sum(self.weights ** 2)

        self.set_bandwidth(bw_method=bw_method)

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        points = np.atleast_2d(points)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                                                                              self.d)
                raise ValueError(msg)

        # compute the normalised residuals
        chi2 = cdist(points.T, self.dataset.T, 'mahalanobis', VI=self.inv_cov) ** 2
        # compute the pdf
        result = np.sum(np.exp(-.5 * chi2) * self.weights, axis=1) / self._norm_factor

        return result

    __call__ = evaluate

    def scotts_factor(self):
        return np.power(self.neff, -1. / (self.d + 4))

    def silverman_factor(self):
        return np.power(self.neff * (self.d + 2.0) / 4.0, -1. / (self.d + 4))

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor

    def set_bandwidth(self, bw_method=None):
        """Compute the estimator bandwidth with given method.

        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluations of the estimated density.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.  This can be
            'scott', 'silverman', a scalar constant or a callable.  If a
            scalar, this will be used directly as `kde.factor`.  If a callable,
            it should take a `gaussian_kde` instance as only parameter and
            return a scalar.  If None (default), nothing happens; the current
            `kde.covariance_factor` method is kept.

        Notes
        -----
        .. versionadded:: 0.11

        Examples
        --------
        >>> x1 = np.array([-7, -5, 1, 4, 5.])
        >>> kde = stats.gaussian_kde(x1)
        >>> xs = np.linspace(-10, 10, num=50)
        >>> y1 = kde(xs)
        >>> kde.set_bandwidth(bw_method='silverman')
        >>> y2 = kde(xs)
        >>> kde.set_bandwidth(bw_method=kde.factor / 3.)
        >>> y3 = kde(xs)

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> ax.plot(x1, np.ones(x1.shape) / (4. * x1.size), 'bo',
        ...         label='Data points (rescaled)')
        >>> ax.plot(xs, y1, label='Scott (default)')
        >>> ax.plot(xs, y2, label='Silverman')
        >>> ax.plot(xs, y3, label='Const (1/3 * Silverman)')
        >>> ax.legend()
        >>> plt.show()

        """
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, string_types):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            # Compute the mean and residuals
            _mean = np.sum(self.weights * self.dataset, axis=1)
            _residual = (self.dataset - _mean[:, None])
            # Compute the biased covariance
            self._data_covariance = np.atleast_2d(np.dot(_residual * self.weights, _residual.T))
            # Correct for bias (http://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance)
            self._data_covariance /= (1 - np.sum(self.weights ** 2))
            self._data_inv_cov = np.linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor ** 2
        self.inv_cov = self._data_inv_cov / self.factor ** 2
        self._norm_factor = np.sqrt(np.linalg.det(2 * np.pi * self.covariance))  # * self.n



class gaussian_kde_tree(object):
    # input: array or list with (n, d) shape

    def __init__(self, dataset, weights=None, nsearch=100, smooth_factor=3, niter=3):
        self.dataset = np.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        self.n, self.d = self.dataset.shape

        if(weights is None):
            self.weights = np.full(dataset.size, 1)
        else:
            self.weights = weights
        self.nsearch = nsearch
        self.smooth_factor = smooth_factor
        self.niter = niter

        self.tree = Tree(self.dataset, leafsize=16, compact_nodes=False, balanced_tree=False)


    def evaluate(self, points):
        points = np.atleast_2d(points)

        # compute the normalised residuals
        distances, indices = self.tree.query(points, self.nsearch, n_jobs=-1)
        normpdf = lambda x, b: (2*np.pi*b**2)**-(0.5*self.d)*np.exp(-0.5*(x/b)**2)
        bandwidth = distances[:, -1]/np.sqrt(self.nsearch)*self.smooth_factor

        if self.weights is not None:
            weights = self.weights[indices]
        else:
            weights = 1

        return np.sum(normpdf(distances.T, bandwidth) * weights.T, axis=0)

    __call__ = evaluate


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    source : https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

class Timer:
    def __init__(self, unitl='s', verbose=0):
        self.t = time.time()
        self.unitl = unitl
        if(unitl == 'h'):
            self.unit = 3600
        elif (unitl == 'm'):
            self.unit = 60
        else:
            self.unit = 1
        self.verbose = verbose
        self.verbose_lim = 1

    def start(self, message=None, verbose_lim=None):
        if(verbose_lim is not None):
            self.verbose_lim = verbose_lim

        if(self.verbose >= self.verbose_lim and message is not None):
            print(message)
        self.t = time.time()


    def time(self):
        return (time.time()-self.t) / self.unit

    def record(self, verbose_lim=None):
        if(verbose_lim is not None):
            self.verbose_lim = verbose_lim

        if (self.verbose >= self.verbose_lim):
            print('Done (%.3f%s).' % (self.time(), self.unitl))

    def measure(self, func, message=None, **kwargs):
        self.start(message)
        result = func(**kwargs)
        self.record()
        return result


