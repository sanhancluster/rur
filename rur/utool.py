from scipy.spatial import cKDTree as Tree

from scipy.spatial.distance import cdist
import numpy as np
from scipy.stats import norm
from scipy.spatial import Delaunay, cKDTree as KDTree
from scipy.interpolate import LinearNDInterpolator
from numpy.linalg import det
import h5py
from rur.sci.geometry import rss, ss, rms, align_to_vector, euler_angle
from rur.config import Table, get_vector, Timer, alias_dict
from collections.abc import Iterable
from collections import defaultdict
import warnings
from numpy.lib import recfunctions as rf
from IPython import get_ipython

import pickle as pkl
import os

from multiprocessing import Process, cpu_count, Manager
from time import sleep


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


if (type_of_script() == 'jupyter'):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# pickle load/save
def uopen(path, mode):
    # create directory if there's no one
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode)


def dump(data, path, msg=True, format='pkl'):
    t = Timer()
    path = os.path.expanduser(path)

    if (format == 'pkl'):
        with uopen(path, 'wb') as opened:
            pkl.dump(data, opened, protocol=4)

    elif (format == 'hdf5'):
        with h5py.File(path, 'w') as f:
            f.create_dataset('table', data=data)

    else:
        raise ValueError("Unknown format: %s" % format)

    if (msg):
        filesize = os.path.getsize(path)
        print("File %s dump complete (%s): %.3f seconds elapsed" % (path, format_bytes(filesize), t.time()))


def load(path, msg=True, format=None):
    t = Timer()
    if (format is None):
        ext = os.path.splitext(os.path.basename(path))[1]
        format = ext[1:]
    path = os.path.expanduser(path)
    if (format == 'pkl'):
        with open(path, 'rb') as opened:
            data = pkl.load(opened, encoding='latin1')
    elif (format == 'hdf5'):
        f = h5py.File(path, 'r')
        data = f['table']
    else:
        raise ValueError("Unknown format: %s" % format)
    if (msg):
        filesize = os.path.getsize(path)
        print("File %s load complete (%s): %.3f seconds elapsed" % (path, format_bytes(filesize), t.time()))
    return data


# Some RAMSES-related stuff
dim_keys = ['x', 'y', 'z']


def los(proj, ndim=3):
    # return line-of-sight dimension from given projection
    if (len(proj) != ndim - 1):
        raise ValueError('Invalid projection')
    dims = np.arange(ndim)
    return dims[np.isin(dims, proj, invert=True)][0]


def set_vector(table, vector, prefix='', ndim=3, where=None, copy=False):
    if (isinstance(table, Table)):
        table = table.table
    if (copy):
        table = table.copy()
    for idim in range(ndim):
        if (where is None):
            table[prefix + dim_keys[idim]] = vector[..., idim]
        else:
            table[prefix + dim_keys[idim]][where] = vector[..., idim]
    if (copy):
        return table


def get_box(center, extent):
    center = np.array(center)
    if (not np.isscalar(extent)):
        extent = np.array(extent)
    return np.stack([center - extent / 2, center + extent / 2], axis=-1)


def get_bounding_box(coo):
    # return bounding box from coordinates
    coo = np.array(coo)
    ndim = coo.shape[-1]
    bbox = ([np.min(coo, axis=0), np.max(coo, axis=0)])


def get_distance(a, b, prefix='', ndim=3):
    return rss(get_vector(a, prefix, ndim) - get_vector(b, prefix, ndim))


def get_polar_coord(coo, pos=[0.5, 0.5, 0.5]):
    coo = coo - np.array(pos)
    r = rss(coo, axis=-1)
    theta = np.arccos(coo[:, 2] / r)
    phi = np.arctan2(coo[:, 1], coo[:, 0])

    return np.stack([r, theta, phi], axis=-1)


def shift(table, vec, ndim=3, periodic=True):
    if (isinstance(table, Table)):
        table = table.table
    for idim in range(ndim):
        table[dim_keys[idim]] += vec[idim]
        if (periodic):
            table[dim_keys[idim]] %= 1


def pairing(a, b, ignore=None):
    # cantor pairing function
    output = (a + b) * (a + b + 1) // 2 + b
    if (ignore is not None):
        mask = (a == ignore) | (b == ignore)
        output[mask] = ignore
    return output


# some custom array-related functions
def bin_centers(start, end, num):
    # get center of num bins that divide the range equally
    arr = np.linspace(start, end, num + 1)
    return (arr[1:] + arr[:-1]) / 2


def append_rows(array, new_rows, idx=None):
    # Calculate the number of old and new rows
    len_array = array.shape[0]
    len_new_rows = new_rows.shape[0]
    # Resize the old recarray
    array.resize(len_array + len_new_rows, refcheck=False)
    # Write to the end of recarray

    if (idx is None):
        array[-len_new_rows:] = new_rows
    else:
        array[idx - len_array:] = array[idx:len_array]
        array[idx:idx + len_new_rows] = new_rows
    return array


def rank(arr):
    temp = np.argsort(arr)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(arr.size)
    return ranks


def expand_shape(arr, axes, ndim):
    # axis: array that has same size with current number of dimensions, positions of axis in the resulting array
    # ndim: number of dimensions of resulting array
    axes = np.atleast_1d(axes)
    arr = np.atleast_1d(arr)
    if (arr.ndim != axes.size):
        raise ValueError("Invalid axes, make sure arr.ndim == axes.size")

    sources = np.arange(arr.ndim)
    dests = rank(axes)
    new_arr = np.moveaxis(arr, sources, dests)

    newshape = np.full(ndim, 1, dtype=int)
    newshape[np.array(axes)] = np.array(arr.shape)[np.arange(axes.size)]

    return np.reshape(new_arr, newshape)


def make_broadcastable(arr_tuple, axes, ndim=None):
    # makes two arrays to be able to broadcast with each other
    if (ndim is None):
        axes_all = np.concatenate(axes)
        ndim = max(axes_all) + 1

    out = ()
    for arr, axis in zip(arr_tuple, axes):
        out += (expand_shape(arr, axis, ndim),)
    return out


def find_intersect(x1, y1, x2, y2):
    """
    :param x1, y1: ndarray of coordinates 1, larger at initial
    :param x2, y2: ndarray of coordnitaes 2, smaller at initial
    :return: first crossing point of two graph
    """
    res = y2 - np.interp(x2, x1, y1)
    mask = res > 0
    if (np.all(mask)):
        mean_ini = np.mean([[x1[0], y1[0]], [x2[0], y2[0]]], axis=0)
        return tuple(mean_ini)
    if (np.all(~mask)):
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
        cut = arr[(bot <= value) & (value < top)]
        if (func is None):
            out.append(cut)
        else:
            out.append(func(cut))
    if (not return_centers):
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
        arr[..., i] = a
    return arr.reshape(-1, la)


def discrete_hist2d(shape, idx, use_long=False):
    if (use_long):
        idx = idx.astype('i8')
        shape = np.array(shape).astype('i8')
    idxs = idx[0] * shape[1] + idx[1]
    hist = np.bincount(idxs, minlength=shape[0] * shape[1])
    hist = np.reshape(hist, shape)
    return hist


def set_bins(bins, lims=None, values=None):
    if (isinstance(bins, int) and np.array(lims).ndim <= 1):
        if (lims is None):
            bins = np.quantile(values, np.linspace(0., 1., bins + 1))

            # Temp: add small offset at the end to include last sample
            bins[-1] += (bins[-1] - bins[-2]) * 0.5
            bins[0] -= (bins[1] - bins[0]) * 0.5
            return bins
        else:
            return np.linspace(*lims, bins + 1)
    else:
        bins = np.atleast_1d(bins)
        if (isinstance(bins[0], int) or isinstance(bins[0], np.int64) and np.array(lims).ndim == 2):
            binarr = []
            bins = np.atleast_1d(bins) * np.array([1, 1])
            for lim, nbin in zip(lims, bins):
                binarr.append(np.linspace(*lim, nbin + 1))
            bins = binarr
            return bins
    return bins


def digitize(points, bins, lims=None, single_idx=False):
    # n-dimensional version of np.digitize, but returns 0:size instead of 1:size+1
    points = np.array(points)
    ndim = points.ndim
    bins = set_bins(bins, lims, points)
    if (ndim == 1):
        return np.digitize(points, bins)

    points = points.T
    idx = []
    shape = []
    for idim in range(len(bins)):
        bin = bins[idim]
        idx.append(np.digitize(points[idim], bin) - 1)
        shape.append(len(bin) - 1)
    shape = np.array(shape)
    idx = np.array(idx)
    if (single_idx):
        idx_single = np.empty(points.shape[-1], dtype=idx.dtype)
        min_mask = np.any(idx < 0, axis=0)
        max_mask = np.any(idx >= shape[:, np.newaxis], axis=0)
        valid_mask = ~(max_mask | min_mask)
        idx_single[valid_mask] = np.ravel_multi_index(idx[..., valid_mask], shape)
        idx_single[min_mask] = -1
        idx_single[max_mask] = np.prod(shape)
        idx = idx_single
    return idx


def add_dims(arr, ndim, axis=-1):
    return np.expand_dims(arr, list(np.arange(axis, axis - ndim, -1)))


class discrete_stat(object):
    """A class to manage discrete set of data y (e.g. binned data, etc...)
    idx indicates the bin position of the data should be integer that gets within size
    statistics (such as mean, median, quantile, etc..) can be computed in binned data
    any idx outside (0, size) will be ignored
    this class only works with 1d discrete data. use inherited class to use it for higher dimensions.
    """

    def __init__(self, y, idx, grid_size=None, weights=None):
        # convert idx and y to numpy array
        y = np.array(y)
        idx = np.array(idx)

        # check dimensions and shapes
        if (idx.ndim != 1):
            raise ValueError("idx must have ndim==1 (received: idx.ndim=%d)" % idx.ndim)
        if (y.shape[0] != idx.size):
            raise ValueError(
                "Size mismatch: %d != %d\ny must have same shape with idx at axis 0!" % (y.shape[0], idx.size))

        # automatically set grid_size
        if (grid_size is None):
            grid_size = np.max(idx) + 1

        # set internal variables
        self.y = y

        # number of items => same as idx.size
        self.n = y.shape[0]

        self.grid_size = grid_size
        self.idx = idx

        # shape and dimension of one item
        self.value_shape = y.shape[1:]
        self.ndim = len(self.value_shape)

        self.output_shape = (self.grid_size,) + self.value_shape
        self.dtype = y.dtype

        # set weights to the correct shape (should be broadcastable to y)
        self.weights = weights
        if self.weights is not None:
            weights = np.atleast_1d(weights)
            try:
                self.weights = np.broadcast_to(weights, y.shape)
            except ValueError:
                raise ValueError("weights must be broadcastable to y!\n"
                                 "shape: %s -> %s" % (weights.shape, y.shape))

        self.init_boundary()
        self.cache = defaultdict(lambda: None)

        self.mode_alias = alias_dict()
        self.method_dict = {}
        self.set_dict()

    def init_boundary(self):
        # sort data by indices and precompute boundaries
        keys = np.argsort(self.idx)
        self.idx = self.idx[keys]
        self.y = self.y[keys]
        if self.weights is not None:
            self.weights = self.weights[keys]

        self.bounds = np.searchsorted(self.idx, np.arange(np.prod(self.grid_size) + 1))
        self.valid_slice = slice(self.bounds[0], self.bounds[-1])

    def apply_at_idx(self, value: np.ndarray, func=np.add, dtype=None, use_valid_slice=True):
        if (dtype is None):
            dtype = self.dtype
        value = np.atleast_1d(value)
        return_shape = (self.grid_size,) + value.shape[1:]

        # set default value of the output array
        if func == np.minimum:
            out = np.full(return_shape, np.inf, dtype=dtype)
        elif func == np.maximum:
            out = np.full(return_shape, -np.inf, dtype=dtype)
        else:
            out = np.full(return_shape, func.identity, dtype=dtype)

        if use_valid_slice:
            value = np.broadcast_to(value, self.idx.shape + value.shape[1:])[self.valid_slice]
        func.at(out, self.idx[self.valid_slice], value)
        return out

    def get(self, mode, method, *args, **kwargs):
        if self.cache[mode] is None:
            self.cache[mode] = method(*args, **kwargs)
        return self.cache[mode]

    def eval(self, mode, *args, **kwargs):
        mode = self.mode_alias[mode]
        try:
            method = self.method_dict[mode]
        except KeyError:
            raise ValueError("Unknown mode: ", mode)
        return self.get(mode, method, *args, **kwargs)

    def __call__(self, mode, *args, **kwargs):
        return self.eval(mode, *args, **kwargs)

    def eval_number(self):
        return self.apply_at_idx(1, dtype='i4')

    def eval_mean(self):
        return self.eval('sum') / self.eval('wsum')

    def eval_weights_sum(self):
        if self.weights is not None:
            wsum = self.apply_at_idx(self.weights)
        else:
            wsum = self.eval('num')
        return wsum

    def eval_minimum(self):
        return self.apply_at_idx(self.y, np.minimum)

    def eval_maximum(self):
        return self.apply_at_idx(self.y, np.maximum)

    def eval_sum(self, use_weights=True):
        y = self.y[self.valid_slice]
        if use_weights and self.weights is not None:
            to_add = y * self.weights[self.valid_slice]
        else:
            to_add = y.copy()
        return self.apply_at_idx(to_add, use_valid_slice=False)

    def eval_variance(self):
        idx = self.idx[self.valid_slice]
        y = self.y[self.valid_slice]
        weights = self.weights
        if (weights.shape[0] != 1):
            weights = weights[self.valid_slice]

        means = self.eval_mean()
        to_add = weights * (y - means[idx]) ** 2
        return self.apply_at_idx(to_add, use_valid_slice=False) / self.eval('wsum')

    def eval_std(self):
        var = self.eval('var')
        return np.sqrt(var)

    def eval_moment(self, k=1):
        idx = self.idx[self.valid_slice]
        y = self.y[self.valid_slice]
        weights = self.weights
        if (weights.shape[0] != 1):
            weights = weights[self.valid_slice]

        means = self.eval('mean')
        stds = self.eval('std')

        to_add = weights * ((y - means[idx]) / stds[idx]) ** k
        moments = self.apply_at_idx(to_add, use_valid_slice=False) / self.eval('wsum')

        return moments

    def eval_skew(self):
        if (self.cache['skew'] is None):
            self.cache['skew'] = self.eval('mom', k=3)
        return self.cache['skew']

    def eval_kurt(self):
        if (self.cache['kurt'] is None):
            self.cache['kurt'] = self.eval('mom', k=4)
        return self.cache['kurt']

    def eval_median(self, *args, **kwargs):
        if (self.cache['med'] is None):
            self.cache['med'] = self.eval('quant', 0.5, *args, **kwargs)
        return self.cache['med']

    def eval_quantile(self, q, use_weights=None):
        y = self.y
        use_weights = use_weights and self.weights is not None
        if (use_weights):
            weights = np.broadcast_to(self.weights, y.shape)
        slices = [slice(low, upp) for low, upp in zip(self.bounds[:-1], self.bounds[1:])]
        out = []
        for sl in slices:
            if (sl.stop - sl.start > 0):
                if (use_weights):
                    qua = np.empty(np.prod(self.value_shape))
                    for sidx in range(qua.size):
                        target_value = y[(sl,) + np.unravel_index(sidx, self.value_shape)]
                        target_weights = weights[(sl,) + np.unravel_index(0, self.value_shape)]
                        qua[sidx] = weighted_quantile(target_value, q, target_weights)
                    qua = np.reshape(qua, self.value_shape)
                else:
                    qua = np.quantile(y[sl], q, axis=0)
                out.append(qua)
            else:
                out.append(np.full(self.value_shape, np.nan))
        return np.array(out)

    def eval_pdf(self, bins_arr):
        idx = self.idx
        y = self.y
        weights = self.weights
        shape = self.output_shape
        # shape should be equal or smaller than 2d

        bins_arr = np.array(bins_arr)

        nvar = shape[1]
        pdf = np.zeros((bins_arr.shape[-1] - 1,) + shape, dtype='f8')

        for i in range(nvar):
            idx_pdf = np.digitize(y[:, i], bins=bins_arr[i]) - 1
            np.add.at(pdf[..., i], (idx_pdf, idx), weights[..., 0])
        return pdf

    def set_dict(self):
        # set defaultdict object for aliasing
        mode_alias = {
            'number': 'num',
            'average': 'mean',
            'weights': 'wsum',
            'minumum': 'min',
            'maximum': 'max',
            'stdev': 'std',
            'standard_deviation': 'std',
            'variance': 'var',
            'skewness': 'skew',
            'skw': 'skew',
            'kurtosis': 'kurt',
            'krt': 'kurt',
            'moment': 'mom',
            'quantile': 'quant',
            'median': 'med'
        }
        self.mode_alias.update(mode_alias)

        # dictionary object to return function
        self.method_dict = {
            'num': self.eval_number,
            'mean': self.eval_mean,
            'wsum': self.eval_weights_sum,
            'sum': self.eval_sum,
            'min': self.eval_minimum,
            'max': self.eval_maximum,
            'std': self.eval_std,
            'var': self.eval_variance,
            'skew': self.eval_skew,
            'kurt': self.eval_kurt,
            'mom': self.eval_moment,
            'quant': self.eval_quantile,
            'med': self.eval_median,
        }


class binned_stat(discrete_stat):
    def __init__(self, x, y, bins, lims=None, weights=None):
        self.bins = set_bins(bins, lims, y)
        idx = digitize(x, self.bins, lims, single_idx=True)
        self.grid_shape = tuple([len(bin) - 1 for bin in self.bins])
        grid_size = np.prod(self.grid_shape)

        super().__init__(y, idx, grid_size, weights)

    def __call__(self, mode, *args, **kwargs):
        out = super().__call__(mode, *args, **kwargs)
        return np.reshape(out, tuple(self.grid_shape) + out.shape[1:], order='F')


class kde_stat(object):

    def __init__(self, points, value, coord, bandwidth, weights=None, sigma_limit=2.):
        self.value = value
        self.coord = np.atleast_2d(coord)
        self.points = np.atleast_2d(points)
        self.tree_data = KDTree(self.coord)
        self.tree_points = KDTree(self.coord)
        self.shape = points.shape[:1] + value.shape[1:]

        if not self.coord.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        self.d, self.n = self.coord.shape

        if weights is not None:
            self.weights = weights / np.sum(weights)
        else:
            self.weights = np.ones(self.n) / self.n

        # compute the normalised residuals
        self.chi2 = cdist(points, self.coord) ** 2
        self.bandwidth = bandwidth
        self.norm_factor = (2 * np.pi * bandwidth ** 2) ** -0.5
        self.kde_weights = np.exp(-.5 * self.chi2 / bandwidth ** 2) * self.weights / self._norm_factor
        # compute the pdf
        self.weights_sums = np.sum(self.kde_weights, axis=-1)

        self.clear()

    def kernel(self, dist):
        return self.norm_factor * np.exp(-0.5 * dist ** 2 / self.bandwidth ** 2)

    def clear(self):
        self.sum = None
        self.mean = None
        self.var = None

    def evaluate(self):
        pass

    def eval_sum(self):
        if self.sum is None:
            points = self.points
            self.sum = np.sum(self.kde_weights * self.var, axis=-1)
        return self.sum

    def eval_mean(self):
        if self.mean is None:
            self.mean = self.eval_sum() / self.weights_sums
        return self.mean

    def eval_var(self):
        mean = self.eval_mean()
        var = np.sum()

    def eval_std(self):
        pass

    __call__ = evaluate


def k_partitioning(centers_init, points, weights,
                   gamma=2.0, scale=1.0, iterations=10, target_std=0.,
                   fail_threshold=(0.05, 5), n_nei=6, n_jobs=-1, verbose=False):
    # finds an effective solution for Voronoi binning with equal weights

    def replace(centers):
        centers = centers.copy()
        tree = KDTree(centers)
        dists, idx_closest = tree.query(points, k=1)
        sums = np.zeros(centers.shape[0], dtype='f8')
        np.add.at(sums, idx_closest, weights)

        idx_min = np.argmin(sums)
        pidx_max = np.argmax(dists * sums[idx_closest])
        centers[idx_min] = points[pidx_max]

        return centers

    def relax(centers):
        centers = centers.copy()
        # get closest centers in each points
        tree = KDTree(centers)
        dists, idx = tree.query(points, k=n_nei)
        idx_closest = idx[:, 0]

        # get number of points in each Voronoi bin
        sums = np.zeros(centers.shape[0], dtype='f8')
        np.add.at(sums, idx_closest, weights)

        multiplier = ((sums[idx] / (np.sum(sums) / centers.shape[0])) ** gamma)
        idx_assign = idx[np.arange(points.shape[0]), np.argmin(dists * multiplier, axis=-1)]

        vectors = np.zeros(centers.shape, dtype='f8')
        np.add.at(vectors, idx_assign, expand_shape(weights, [0], 2) * points)

        sums = np.zeros(centers.shape[0], dtype='f8')
        np.add.at(sums, idx_assign, weights)

        centers[sums > 0] = (vectors[sums > 0] / expand_shape(sums[sums > 0], [0], 2))
        return centers

    def perturb(centers):
        centers = centers.copy()
        tree = KDTree(centers)
        dists, idx_closest = tree.query(points)

        sums = np.zeros(centers.shape[0], dtype='f8')
        np.add.at(sums, idx_closest, weights)

        mdists = np.zeros(centers.shape[0], dtype='f8')
        np.add.at(mdists, idx_closest, weights * np.sqrt(np.sum((points - centers[idx_closest]) ** 2, axis=-1)))
        mdists = mdists / sums
        mdists[sums == 0] = 0.

        offset = np.random.normal(size=centers.shape, scale=1.)
        centers += offset * expand_shape(mdists, [0], 2) * scale

        return centers

    centers_min = centers_init
    sums_min = voronoi_binning(centers_min, points)
    std_min = np.std(sums_min) / np.mean(sums_min)

    for niter in range(iterations):
        if (niter > 0):
            centers = perturb(centers_min)
        else:
            centers = centers_min

        sums = voronoi_binning(centers, points)
        std = np.std(sums) / np.mean(sums)

        nfail = 0
        centers_new = centers
        while (nfail < fail_threshold[1]):
            centers_new = relax(centers_new)
            while (True):
                sums_new = voronoi_binning(centers_new, points)
                std_new = np.std(sums_new) / np.mean(sums_new)
                if (np.all(sums_new > 0.)):
                    break
                centers_new = replace(centers_new)
                if (verbose):
                    print('replace', std_new)

            if (std_new < std):
                if (std_new > std * (1 - fail_threshold[0])):
                    nfail += 1
                centers = centers_new
                std = std_new
                if (verbose):
                    print('iter', std)
            else:
                nfail += 1

        if (std < std_min):
            std_min = std
            centers_min = centers
            if (verbose):
                print('best', std)
            if (std_min < target_std):
                break

    return centers_min, std_min


def voronoi_binning(centers, points, weights=1., n_jobs=-1):
    idx_closest = find_closest(centers, points, n_jobs=n_jobs)

    sums = np.zeros(centers.shape[0], dtype='f8')
    np.add.at(sums, idx_closest, weights)

    return sums


def find_closest(centers, points):
    tree = KDTree(centers)
    idx_closest = tree.query(points, k=1)[1]
    return idx_closest


def format_bytes(size, format='{:#.4g}'):
    power = 1024
    n = 0
    power_labels = {0: '', 1: 'Ki', 2: 'Mi', 3: 'Gi', 4: 'Ti', 5: 'Pi'}
    while size > power:
        size /= power
        n += 1
    return (format.format(size)) + ' ' + power_labels[n] + 'B'


# not working for now
def ugenfromtxt(fname, comments='#', delimeter=' ', skip_header=0, dtype_int='i4', dtype_float='f8', *kwargs):
    with open(fname) as fi:
        fi.readline(skip_header)
        while (True):
            line = fi.readline()
            if (not line.startswith('#')):
                break
        elements = line.split(delimeter)
        dtype = []
        for element in elements:
            if (isint(element)):
                dtype.append(dtype_int)
            elif (isfloat(element)):
                dtype.append(dtype_float)
            else:
                dtype.append('U%d' % len(element))
    print(dtype)

    np.genfromtxt(fname, dtype=dtype, comments=comments, delimiter=delimeter, skip_header=skip_header, names=True,
                  *kwargs)


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def project_data(data, normal, origin=np.array([0., 0., 0.]), prefix='', copy=False):
    """
    :param pos: coordinates of original points
    :param normal: normal vector
    :return: rotated coordinate
    """
    pos = get_vector(data, prefix) - origin
    pos_r = align_to_vector(pos, normal) + origin
    set_vector(data, pos_r, prefix=prefix, copy=copy)
    return data


def rotate_data(data, angles, origin, prefix='', order='ZXZ', copy=False):
    # applies euler rotation for data
    pos = get_vector(data, prefix) - origin
    pos_r = euler_angle(pos, angles, order) + origin  # + euler_angle(origin, angles, order)
    if (copy):
        return set_vector(data, pos_r, prefix=prefix, copy=copy)
    else:
        set_vector(data, pos_r, prefix=prefix, copy=copy)
        return data


def weighted_std(values, weights, axis=0):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=axis)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights, axis=axis)
    return np.sqrt(variance)


class dtfe(object):

    def __init__(self, dataset, weights=None, smooth=0):
        self.dataset = np.atleast_2d(dataset).T
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        self.n, self.d = self.dataset.shape
        # center = np.median(self.dataset, axis=0)
        center = 0

        self.tri = Delaunay(self.dataset - center, qhull_options='')

        simplices = self.tri.simplices
        vertices = self.dataset[simplices]

        matrices = np.insert(vertices, 2, 1., axis=-1)
        matrices = np.swapaxes(matrices, -1, -2)
        tri_areas = np.abs(det(matrices)) / np.math.factorial(self.d + 1)

        hull_areas = np.zeros(self.n, dtype='f8')
        np.add.at(hull_areas, simplices, np.expand_dims(tri_areas, -1))

        if (weights is not None):
            hull_areas /= weights

        if (smooth > 0):
            indptr, neighbor_indices = self.tri.vertex_neighbor_vertices
            neighbor_nums = np.full(self.n, 0, dtype='i8')
            center_indices = np.repeat(np.arange(neighbor_nums.size), np.diff(indptr))
            np.add.at(neighbor_nums, center_indices, 1)

            for _ in range(smooth):
                hull_areas_new = np.zeros(hull_areas.shape, dtype='f8')
                np.add.at(hull_areas_new, center_indices, hull_areas[neighbor_indices])
                hull_areas = hull_areas_new / neighbor_nums

        densities = 1 / hull_areas

        self.lip = LinearNDInterpolator(self.tri, densities, fill_value=0)

    def evaluate(self, points):
        return self.lip(*points)

    __call__ = evaluate


def metropolis_hastings(loglike, params_ini, bandwidth, return_like=False, burn_in=10, n_points=100, n_sample=10000,
                        show_progress=False):
    n_pars = len(params_ini)
    loglike_ini = loglike(*params_ini)
    params_arr = np.tile(params_ini, (n_points, 2, 1))
    loglike_arr = np.tile(loglike_ini, (n_points, 2))
    loglike_mean = loglike_arr[:, 0].copy()
    n_iter = np.zeros(n_points, 'i8')
    mask_burn = np.full(n_points, False)

    if (return_like):
        params_out = np.zeros((n_sample, n_pars + 1))
    else:
        params_out = np.zeros((n_sample, n_pars))
    n_fill = 0
    if (show_progress):
        iterator = tqdm(total=n_sample)
    while (n_fill < n_sample):
        offset = np.random.normal(scale=bandwidth, size=(n_points, n_pars))

        params_arr[:, 1] = params_arr[:, 0] + offset
        loglike_arr[:, 1] = loglike(*params_arr[:, 1].T)

        alpha = np.exp(loglike_arr[:, 1] - loglike_arr[:, 0])
        rand = np.random.rand(n_points)
        mask = (rand < alpha)

        params_arr[mask, 0] = params_arr[mask, 1]
        loglike_arr[mask, 0] = loglike_arr[mask, 1]
        n_iter[mask] += 1

        mask_burn = mask_burn | (n_iter > burn_in) & (loglike_mean - loglike_arr[:, 0] > 0.)
        loglike_mean[mask] = (loglike_mean[mask] * (burn_in - 1) + loglike_arr[mask, 0]) / burn_in

        mask = mask & mask_burn
        n_add = np.minimum(np.sum(mask), n_sample - n_fill)
        params_out[n_fill:n_fill + n_add, :n_pars] = params_arr[mask, 0][:n_add]
        if (return_like):
            params_out[n_fill:n_fill + n_add, -1] = loglike_arr[mask, 0][:n_add]
        n_fill += n_add
        if (show_progress):
            iterator.update(n_add)
    if (show_progress):
        iterator.close()
    return params_out


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

        if (weights is None):
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
        distances, indices = self.tree.query(points, self.nsearch)
        normpdf = lambda x, b: (2 * np.pi * b ** 2) ** -(0.5 * self.d) * np.exp(-0.5 * (x / b) ** 2)
        bandwidth = distances[:, -1] / np.sqrt(self.nsearch) * self.smooth_factor

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


def weighted_median(values, *args, **kwargs):
    return weighted_quantile(values, quantiles=0.5, *args, **kwargs)


def multiproc(param_arr, func, n_proc=None, n_chunk=1, wait_period_sec=0.01, ncols_tqdm=None,
              direct_input=True, priorities=None, kwargs_dict=None):
    """
    A simple multiprocessing tool similar to joblib, but independent to picklability.
    It runs function 'func' with parameters 'param_arr'.

    Parameters
    ----------
    param_arr : list of tuples
    func : function
    n_proc : int
    n_chunk : int
    wait_period_sec
    ncols_tqdm
    direct_input : bool
    priorities

    Returns
    -------
    list that stores returned value of each function result.

    """
    if (n_proc is None):
        n_proc = cpu_count()

    if (kwargs_dict is None):
        kwargs_dict = {}

    def worker(param_slice, idx_slice, output_arr):
        output_slice = []
        for param in param_slice:
            if (direct_input):
                output_slice.append(func(*param, **kwargs_dict))
            else:
                output_slice.append(func(param, **kwargs_dict))

        output_arr[idx_slice] = output_slice

    def finalize():
        for proc in procs:
            proc.join()
        for proc in procs:
            proc.close()

    procs = []
    output_size = len(param_arr)
    manager = Manager()
    output_arr = manager.list([None, ] * output_size)

    if (priorities is not None):
        keys = np.argsort(priorities)
        keys_inv = np.argsort(keys)
    else:
        keys = np.arange(output_size)
        keys_inv = keys
    param_arr = [param_arr[key] for key in keys]

    head_idxs = np.arange(0, output_size, n_chunk)
    idx_proc = 0
    iterator = tqdm(head_idxs, ncols=ncols_tqdm)
    try:
        for head_idx in iterator:
            idx_slice = slice(head_idx, np.minimum(head_idx + n_chunk, output_size))
            wait = 0.
            while (len(procs) >= n_proc):
                sleep(wait)
                for idx in range(len(procs))[::-1]:
                    if not procs[idx].is_alive():
                        procs.pop(idx)
                wait = wait_period_sec

            p = Process(target=worker, args=(param_arr[idx_slice], idx_slice, output_arr))
            procs.append(p)
            p.start()
            idx_proc += 1
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        return None
    finally:
        finalize()
        iterator.close()
        return [output_arr[key] for key in keys_inv]


def join_arrays(arrays):
    # Same as rf.merge_arrays, but does not break nested dtype structures
    # from https://stackoverflow.com/questions/5355744/numpy-joining-structured-arrays
    sizes = np.array([a.itemsize for a in arrays])
    offsets = np.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    joint = np.empty((n, offsets[-1]), dtype=np.uint8)
    for a, size, offset in zip(arrays, sizes, offsets):
        joint[:, offset:offset + size] = a.view(np.uint8).reshape(n, size)
    dtype = sum((a.dtype.descr for a in arrays), [])
    return joint.ravel().view(dtype)


def add_fields(table, dtype, fill_value=None, overwrite=True):
    # add field to an np.recarray / structured array or Table (and inherited classes, e.g. Cell, Particle, ...)
    if isinstance(table, Table):
        data = table.table
    else:
        data = table
    dtype = np.dtype(dtype)
    if (overwrite):
        names_to_drop = np.array(dtype.names)[np.isin(dtype.names, data.dtype.names)]
        data = rf.drop_fields(data, names_to_drop)
    if fill_value is None:
        to_add = np.zeros_like(data, dtype=dtype)
    else:
        to_add = np.full_like(data, fill_value=fill_value, dtype=dtype)
    data = join_arrays([data, to_add])

    if isinstance(table, Table):
        return table.__copy__(data)
    else:
        return data
