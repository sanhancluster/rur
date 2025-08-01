from pylab import *

from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
from rur.utool import *
from scipy.spatial import Delaunay, cKDTree as KDTree
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.stats import norm
from scipy.signal import convolve2d
from numpy.linalg import det
from skimage.transform import resize, rescale, warp, EuclideanTransform, AffineTransform
from scipy.sparse import csr_matrix

import string
import matplotlib.collections as mcoll
from os.path import dirname, join, abspath
import pkg_resources

def colorline(
        x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0, zorder=0, ax=None):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha, zorder=zorder)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def image_coord(x, y, lims):
    x_out, y_out = (x - lims[0][0]) / (lims[0][1] - lims[0][0]), (y - lims[1][0]) / (lims[1][1] - lims[1][0])
    mask = bmask(x_out, (0, 1)) & bmask(y_out, (0, 1))
    return x_out[mask], y_out[mask]

def remove_keys(dic, keys):
    dic = dic.copy()
    for key in keys:
        if(key in dic):
            dic.pop(key)
    return dic


def axlabel(text, pos='right top', loc=None, offset=0.03, ax=None, **kwargs):
    # ha: right, left
    # va: top (or upper), bottom (or lower)

    if loc is not None:
        pos = loc
    pos = tuple(pos.split(' '))
    offset = np.atleast_1d(offset) * [1, 1]
    if 'right' in pos:
        x = 1 - offset[0]
        ha = 'right'
    elif 'left' in pos:
        x = offset[0]
        ha = 'left'
    elif 'center' in pos:
        x = 0.5
        ha = 'center'
    else:
        raise ValueError("Unknown horizontal position")
    if 'top' in pos or 'upper' in pos:
        y = 1 - offset[1]
        va = 'top'
    elif 'bottom' in pos or 'lower' in pos:
        y = offset[1]
        va = 'bottom'
    elif 'center' in pos:
        y = 0.5
        va = 'center'
    else:
        raise ValueError("Unknown vertical position")
    if ax is None:
        ax = plt.gca()
    ax.text(x, y, text, ha=ha, va=va, transform=ax.transAxes, **kwargs)


def bmask(arr, bet):
    return (bet[0] <= arr) & (arr < bet[1])


def kde_imshow(x, y, lims=None, reso=100, weights=None, tree=True, **kwargs):
    if(lims is None):
        lims = [(np.nanquantile(x, 0.001), np.nanquantile(x, 0.999)),
                (np.nanquantile(y, 0.001), np.nanquantile(y, 0.999))]
        print('Automatically setting lims as ', lims)

    pdf = kde_img(x, y, lims, reso, weights=weights, tree=tree).T
    plt.imshow(pdf, origin='lower', extent=[lims[0][0], lims[0][1], lims[1][0], lims[1][1]], aspect='auto', **kwargs)


def hist_imshow(x, y, lims=None, reso=100, weights=None, filter_sigma=None, normalize=None, **kwargs):
    # similar to plt.hist2d, with additional options
    if(lims is None):
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if(weights is not None):
            weights = weights[mask]
        lims = [[np.nanquantile(x, 0.00001), np.nanquantile(x, 0.99999)],
                [np.nanquantile(y, 0.00001), np.nanquantile(y, 0.99999)]]
        print('Automatically setting lims as ', lims)

    pdi = np.histogram2d(x, y, range=lims, bins=reso, weights=weights, density=True)[0].T
    if(normalize is not None and filter_sigma is not None):
        print('Warning: using both filter and normalization may cause problem')
    if(normalize is not None):
        if(normalize == 'column'):
            sums = np.sum(pdi, axis=0)
            pdi = pdi/sums
        elif(normalize == 'row'):
            sums = np.sum(pdi, axis=1)
            pdi = np.swapaxes(np.swapaxes(pdi, 0, 1)/sums, 0, 1)

    if(filter_sigma is not None):
        pdi = gaussian_filter(pdi, sigma=filter_sigma)
        area_per_px = (lims[0][1]-lims[0][0]) * (lims[1][1]-lims[1][0]) / reso**2
        pdi = pdi/np.sum(pdi) / area_per_px

    plt.imshow(pdi, origin='lower', extent=[lims[0][0], lims[0][1], lims[1][0], lims[1][1]], aspect='auto', **kwargs)


def kde_contour(x, y, lims, reso=100, bw_method='silverman', weights=None, sig_arr=[1, 2], filled=False, **kwargs):

    pdi = kde_img(x, y, lims, reso, weights=weights, bw_method=bw_method, density=True)
    area_per_px = (lims[0][1]-lims[0][0])*(lims[1][1]-lims[1][0])/reso**2
    levels = np.append([sig_level(pdi, sig_arr, area_per_px)[::-1]], np.max(pdi))

    xarr = bin_centers(lims[0][0], lims[0][1], reso)
    yarr = bin_centers(lims[1][0], lims[1][1], reso)

    if(filled):
        return plt.contourf(xarr, yarr, pdi, levels=levels, **kwargs)
    else:
        return plt.contour(xarr, yarr, pdi, levels=levels, **kwargs)


def fun_contour(f, lims, reso=100, axis=-1, sig_arr=[1, 2], filled=False, **kwargs):

    pdi = fun_img(f, lims, reso, axis)
    area_per_px = (lims[0][1]-lims[0][0])*(lims[1][1]-lims[1][0])/reso**2
    levels = np.append([sig_level(pdi, sig_arr, area_per_px)[::-1]], np.max(pdi))

    xarr = bin_centers(lims[0][0], lims[0][1], reso)
    yarr = bin_centers(lims[1][0], lims[1][1], reso)

    if(filled):
        return plt.contourf(xarr, yarr, pdi, levels=levels, **kwargs)
    else:
        return plt.contour(xarr, yarr, pdi, levels=levels, **kwargs)


def hist_contour(x, y, lims, reso=100, weights=None, sig_arr=[1, 2], filled=False, filter_sigma=None, **kwargs):

    pdi = np.histogram2d(x, y, range=lims, bins=reso, weights=weights, density=True)[0].T
    if filter_sigma is not None:
        pdi = gaussian_filter(pdi, sigma=filter_sigma)
        area_per_px = (lims[0][1]-lims[0][0]) * (lims[1][1]-lims[1][0]) / reso**2
        pdi = pdi/np.sum(pdi) / area_per_px

    area_per_px = (lims[0][1]-lims[0][0])*(lims[1][1]-lims[1][0])/reso**2
    levels = np.append([sig_level(pdi, sig_arr, area_per_px)[::-1]], np.max(pdi))

    xarr = bin_centers(lims[0][0], lims[0][1], reso)
    yarr = bin_centers(lims[1][0], lims[1][1], reso)

    if(filled):
        return plt.contourf(xarr, yarr, pdi, levels=levels, **kwargs)
    else:
        return plt.contour(xarr, yarr, pdi, levels=levels, **kwargs)


def sig_level(pdi, sig_arr, area_per_px=1, normed=True):
    arr = np.sort(pdi.ravel())[::-1]
    cs = np.cumsum(arr) * area_per_px
    if(normed):
        cs = cs/np.max(cs)

    prob_arr = (norm.cdf(sig_arr)-0.5)*2
    levels = []

    for prob in prob_arr:
        idx = np.flatnonzero(cs>prob)[0]
        if(idx < 1):
            print("ERROR: Too low sigma")
            return None
        frac = (prob - cs[idx-1]) / (cs[idx] - cs[idx-1])
        levels.append(arr[idx-1] + (arr[idx] - arr[idx-1]) * frac)
    return levels


def hist_img_adaptive(x, y, lims, reso=100, weights=None, smooth=5, supergrids=5, border='wrap'):
    x = np.array(x)
    y = np.array(y)
    x, y = image_coord(x, y, lims)
    if(np.isscalar(reso)):
        reso = np.repeat(reso, 2)
    reso = np.array(reso)
    lims = np.array(lims)

    size_weights = np.full(x.shape, 1.)

    for ulvl in np.arange(1, supergrids+1):
        supereso = (reso * 0.5**ulvl).astype(int)
        superhist = histogram2d(x, y, bins=supereso, range=[[0, 1], [0, 1]])[0]
        den = superhist[(x*supereso[0]).astype(int), (y*supereso[1]).astype(int)]
        size_weights[den == 1] *= 0.25

    size_hist = histogram2d(x, y, bins=reso, range=[[0, 1], [0, 1]], weights=size_weights)[0]
    if(weights is not None):
        weight_map = histogram2d(x, y, bins=reso, range=[[0, 1], [0, 1]], weights=weights)[0]/histogram2d(x, y, bins=reso, range=[[0, 1], [0, 1]])[0]
    else:
        weight_map = 1


    vals = size_hist[size_hist>0]
    minval = np.min(vals)
    maxval = np.max(vals)

    image = np.zeros(size_hist.shape)
    lvls = np.unique(vals)
    step = np.max([int(lvls.size/10), 1])
    lvls_bot = np.concatenate([lvls[lvls<1], lvls[lvls>=1][:-step:step], [lvls[-1]]])
    lvls_top = np.concatenate([lvls[lvls<1]*1.01, lvls[lvls>=1][step::step], [lvls[-1]*1.01]])
    lvls_cen = np.sqrt(lvls_bot * lvls_top)
    print('Number of adaptive kernel sizes: %d' % lvls_cen.size)

    n = 0

    for bot, cen, top in zip(lvls_bot, lvls_cen, lvls_top):
        mask = (bot <= size_hist) & (size_hist < top)

        # arbitrary sigma function that best produces a nice picture
        sigma = smooth * (cen**-(0.5))

        image += gaussian_filter(size_hist*mask*weight_map, sigma=sigma, mode=border)
        n += sum(mask)

    cell_size = np.prod(reso)

    return image

def gaussian_filter_border(image, sigma, **kwargs):
    sigma_int = int(sigma)
    fraction = sigma - sigma_int
    return gaussian_filter(image, sigma_int, **kwargs) * fraction + gaussian_filter(image, sigma_int+1, **kwargs) * (1-fraction)

def estimate_density_2d(x, y, lims, shape=100, weights=None, density=False, method='hist', **kwargs):
    """
    Estimate the density of points in 2D space and return a 2D image.
    """
    # apply kde-like image convolution using gaussian filter
    x, y = np.array(x), np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # set up shapes and ranges
    if np.isscalar(shape):
        shape = np.repeat(shape, 2)
    shape_array = np.asarray(shape)
    range_array = np.asarray(lims)

    if(weights is not None):
        weights = weights[mask]
    else:
        weights = np.ones_like(x)

    if method == 'hist':
        return hist_img(x, y, range_array, shape_array, weights=weights, density=density, **kwargs)
    elif method == 'kde':
        return kde_img(x, y, range_array, shape_array, weights=weights, density=density, **kwargs)
    elif method == 'cic':
        return cic_img(x, y, range_array, shape_array, weights=weights, density=density, **kwargs)
    elif method == 'gaussian':
        return gaussian_img(x, y, range_array, shape_array, weights=weights, density=density, **kwargs)
    elif method == 'dtfe':
        return dtfe_img(x, y, range_array, shape_array, weights=weights, density=density, **kwargs)
    elif method == 'hist_numpy':
        im = np.histogram2d(x, y, range=range_array, bins=shape_array, weights=weights, **kwargs)[0]
        if density:
            area_per_px = (range_array[0, 1] - range_array[0, 0]) * (range_array[1, 1] - range_array[1, 0]) / im.size
            im /= area_per_px
        return im
    else:
        raise ValueError("Unknown mode: %s. Use 'hist', 'kde', 'cic', 'gaussian', or 'dtfe'." % method)


def gaussian_img(x, y, lims, reso=100, weights=None, density=False, subdivide=4, kernel_size=1, sampling_sigma=2.5):
    # apply kde-like image convolution using gaussian filter
    x, y = np.array(x), np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # set up shapes and ranges
    if np.isscalar(reso):
        reso = np.repeat(reso, 2)
    lims = np.asarray(lims)
    shape_array = np.asarray(reso)
    range_array = np.asarray(lims)

    if(weights is not None):
        weights = weights[mask]
    else:
        weights = np.ones_like(x)

    kern_array_width = int(kernel_size*subdivide*sampling_sigma*2)

    arr = bin_centers(-kern_array_width/2, kern_array_width/2, kern_array_width)
    xm, ym = np.meshgrid(arr, arr)
    mesh = np.stack([xm, ym], axis=-1)
    dist = rss(mesh)

    kern = norm.pdf(dist, scale=kernel_size*subdivide)
    kern /= np.sum(kern)

    hist = np.histogram2d(x, y, bins=shape_array*subdivide, range=range_array, weights=weights)[0]
    hist = convolve2d(hist, kern, mode='same', boundary='symm')
    hist = resize(hist, shape_array, anti_aliasing=True) * subdivide**2
    if density:
        area_per_px = (range_array[0, 1] - range_array[0, 0]) * (range_array[1, 1] - range_array[1, 0]) / hist.size
        hist /= area_per_px

    return hist

def kde_img(x, y, lims, reso=100, weights=None, density=False, tree=True, bw_method='silverman', nsearch=30, smooth_factor=1):
    x, y = np.array(x), np.array(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # set up shapes and ranges
    if np.isscalar(reso):
        reso = np.repeat(reso, 2)
    lims = np.asarray(lims)
    shape_array = np.asarray(reso)
    range_array = np.asarray(lims)

    if(weights is not None):
        weights = weights[mask]
    else:
        weights = np.ones_like(x)

    if(tree):
        kde = gaussian_kde_tree(np.stack([x, y], axis=-1), weights=weights, nsearch=nsearch, smooth_factor=smooth_factor)
        grid = fun_img(kde, lims, shape_array, axis=-1)
    else:
        kde = gaussian_kde(np.stack([x, y], axis=0), weights=weights, bw_method=bw_method)
        grid = fun_img(kde, lims, shape_array, axis=0)

    if not density:
        area_per_px = (range_array[0, 1] - range_array[0, 0]) * (range_array[1, 1] - range_array[1, 0]) / grid.size
        grid *= area_per_px
    return grid

def cic_img(x, y, lims, reso=100, weights=None, density=False, full_vectorize=False):
    """
    Create a 2D image using Cloud-in-Cell (CIC) method.
    This method is useful for creating density maps from point data.
    Parameters:
    - x, y: 1D arrays of coordinates
    - lims: 2D array defining the limits of the image, shape (2, 2)
    - reso: resolution of the image, can be a scalar or a tuple/list of two values
    - weights: optional 1D array of weights for each point
    - full_vectorize: if True, uses a fully vectorized approach (memory-intensive)
    Returns:
    - pool: 2D numpy array representing the image, shape (reso[0], reso[1])
    Note: The function assumes that the input coordinates are finite and within the specified limits.
    If the coordinates are outside the limits, they will be ignored.
    """

    # set up shapes and ranges
    if np.isscalar(reso):
        reso = np.repeat(reso, 2)
    lims = np.asarray(lims)
    shape_array = np.asarray(reso)
    shape_pad = shape_array + 2
    range_array = np.asarray(lims)

    # mask calculation
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    mask &= (x >= lims[0, 0]) & (x < lims[0, 1])
    mask &= (y >= lims[1, 0]) & (y < lims[1, 1])

    x = x[mask]
    y = y[mask]

    points = np.stack([x, y], axis=-1)

    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = np.asarray(weights)[mask]

    # Normalize coordinates to [0, 1] range
    indices_float = (points - range_array[:, 0]) / (range_array[:, 1] - range_array[:, 0]) * shape_array + 0.5
    indices_float = indices_float.reshape(-1, 2)

    # Create zero array for accumulation
    dxs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    if not full_vectorize:
        pool = np.zeros(shape_array, dtype='f8')
        for dx in dxs:
            indices_int = np.floor(indices_float - dx).astype(np.int32)
            offsets = indices_float - indices_int
            areas = (1 - np.abs(offsets[:, 0] - 1)) * (1 - np.abs(offsets[:, 1] - 1))
            values = areas * weights

            indices_int += 1  # padding offset
            flat_indices = indices_int[:, 0] * shape_pad[1] + indices_int[:, 1]
            accum = np.bincount(flat_indices, weights=values, minlength=shape_pad[0] * shape_pad[1])
            pool += accum.reshape(shape_pad)[1:-1, 1:-1]
    else:
        # full vectorization: memory-intensive
        indices_int = np.floor(indices_float[None, :, :] - dxs[:, None, :]).astype(np.int32)
        offsets = indices_float[None, :, :] - indices_int
        areas = (1 - np.abs(offsets[..., 0] - 1)) * (1 - np.abs(offsets[..., 1] - 1))
        values = (areas * weights[None, :]).reshape(-1)

        indices_int += 1
        flat_indices = (indices_int[..., 0] * shape_pad[1] + indices_int[..., 1]).reshape(-1)
        accum = np.bincount(flat_indices, weights=values, minlength=shape_pad[0] * shape_pad[1])
        pool = accum.reshape(shape_pad)[1:-1, 1:-1]
    
    if density:
        area_per_px = (range_array[0, 1] - range_array[0, 0]) * (range_array[1, 1] - range_array[1, 0]) / pool.size
        pool /= area_per_px

    return pool

def hist_img(x, y, lims, reso=100, weights=None, density=False):
    """
    Create a 2D histogram image from x and y coordinates.
    Only works for the uniform grid and faster than np.histogram2d, but the result may slightly vary near the bin edges.
    Parameters:
    - x, y: 1D arrays of coordinates
    - lims: 2D array defining the limits of the image, shape (2, 2)
    - reso: resolution of the image, can be a scalar or a tuple/list of two values
    - weights: optional 1D array of weights for each point
    - density: if True, normalize the histogram to represent a probability density function
    - filter_sigma: if provided, apply Gaussian smoothing with this sigma value
    Returns:
    - pool: 2D numpy array representing the histogram image, shape (reso[0], reso[1])
    """
    
    # set up shapes and ranges
    if np.isscalar(reso):
        reso = np.repeat(reso, 2)
    lims = np.asarray(lims)
    shape_array = np.asarray(reso)
    shape_pad = shape_array + 2

    # mask calculation
    x = np.asarray(x)
    y = np.asarray(y)

    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = np.asarray(weights)

    xi = uniform_digitize(x, lims[0], shape_array[0])
    yi = uniform_digitize(y, lims[1], shape_array[1])

    flat_indices = xi * shape_pad[1] + yi
    accum = np.bincount(flat_indices, weights=weights, minlength=shape_pad[0] * shape_pad[1])
    pool = accum.reshape(shape_pad)[1:-1, 1:-1]
    
    if density:
        area_per_px = (lims[0][1] - lims[0][0]) * (lims[1][1] - lims[1][0]) / pool.size
        pool /= area_per_px

    return pool

def coo_img(lims, reso=100, axis=-1, ravel=False):
    # set up shapes and ranges
    if np.isscalar(reso):
        reso = np.repeat(reso, 2)
    lims = np.asarray(lims)
    shape_array = np.asarray(reso)
    range_array = np.asarray(lims)

    xarr = bin_centers(range_array[0][0], range_array[0][1], shape_array[0])
    yarr = bin_centers(range_array[1][0], range_array[1][1], shape_array[1])

    xm, ym = np.meshgrid(xarr, yarr)
    if(ravel):
        xm, ym = xm.ravel(), ym.ravel()

    mesh = np.stack([xm, ym], axis=axis)
    return mesh

def fun_img(f, lims, reso=100, axis=-1):
    # returns 2d numpy array image with function
    # axis: the axis that function accepts to separate each dimensions.
    # set up shapes and ranges
    if np.isscalar(reso):
        reso = np.repeat(reso, 2)
    shape_array = np.asarray(reso)

    mesh = coo_img(lims, shape_array, axis=axis, ravel=True)

    pdi = f(mesh)
    pdi = np.reshape(pdi, shape_array)

    return pdi.T

def kde_scatter(x, y, bw_method='silverman', cmap=plt.cm.jet, xlog=False, ylog=False, weights=None, **kwargs):
    if(xlog):
        x = np.log10(x)
    if(ylog):
        y = np.log10(y)

    mask = np.isfinite(x) & np.isfinite(y)
    coo = np.array([x[mask], y[mask]])

    kde = gaussian_kde_tree(coo, bw_method=bw_method, weights=weights)
    density = kde(coo)
    maxd = np.max(density)

    if(xlog):
        coo[0] = 10**coo[0]
    if(ylog):
        coo[1] = 10**coo[1]

    return plt.scatter(*coo, color=cmap(density/maxd), **kwargs)

def voronoi_img(centers, lims, reso=500):
    f = lambda x: find_closest(centers, x)
    return fun_img(f, lims, reso, axis=-1)

def dtfe_img(x, y, lims, reso=100, weights=None, density=False, smooth=0, interpolator='linear'):
    """
    Create a 2D image using Delaunay triangulation and area-based density estimation.
    This method computes the density of points in a 2D space by triangulating the points
    and calculating the area of the triangles formed. The density is then interpolated
    onto a grid defined by the specified limits and resolution.
    """
    if (np.isscalar(reso)):
        reso = np.repeat(reso, 2)

    # set up shapes and ranges
    range_array = np.array(lims)
    shape_array = np.array(reso)
    
    points = np.stack([x, y], axis=-1)
    center = np.median(points, axis=0)
    n_points = points.shape[0]

    if(smooth is None):
        smooth = int(0.05 * n_points**0.6)

    # For some "Complex Geometrical Reasons", Qhull does not work properly without options???
    # Even with blank option, the result is different.
    tri = Delaunay(points-center, qhull_options='Qbb Qc Qx')
    
    simplices = tri.simplices
    vertices = points[simplices]
        
    v1 = vertices[:, 1] - vertices[:, 0]
    v2 = vertices[:, 2] - vertices[:, 0]
    tri_areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

    flat_simplices = simplices.reshape(-1)
    area_repeat = np.repeat(tri_areas, 3)
    hull_areas = np.bincount(flat_simplices, weights=area_repeat, minlength=n_points) / 3

    if(smooth>0):
        indptr, neighbor_indices = tri.vertex_neighbor_vertices
        rows = np.repeat(np.arange(n_points), np.diff(indptr))
        cols = neighbor_indices
        data = np.ones_like(cols, dtype=np.float64)

        neighbor_nums = np.bincount(rows, minlength=n_points)
        data = data / neighbor_nums[rows]

        W = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

        for _ in range(smooth):
            hull_areas = (hull_areas + W @ hull_areas) / 2

    densities = 1 / hull_areas

    if(weights is not None):
        densities *= weights

    xarr = bin_centers(range_array[0][0], range_array[0][1], shape_array[0])
    yarr = bin_centers(range_array[1][0], range_array[1][1], shape_array[1])

    xm, ym = np.meshgrid(xarr, yarr)

    if interpolator == 'nearest':
        ip = NearestNDInterpolator(points, densities)
    elif interpolator == 'linear':
        ip = LinearNDInterpolator(points, densities)
    grid = ip(xm, ym).T
    grid = np.nan_to_num(grid, nan=0.0)

    if not density:
        area_per_px = (range_array[0, 1] - range_array[0, 0]) * (range_array[1, 1] - range_array[1, 0]) / grid.size
        grid *= area_per_px

    return grid


def mosaic_stat(x, y, v, weights=None, bins=10, lims=None, minnum=0, statmode=None, show_number=False, statmin=1, fmt="%.3f", fontsize=8, contour=False, **kwargs):
    if statmode is None:
        statmode = 'mean'
    bins = np.atleast_1d(bins)
    stat = binned_stat(np.stack([x, y], axis=-1), v, bins, lims, weights=weights)

    num = stat('num')
    arr = stat(statmode)

    mask = num < minnum
    arr = np.ma.masked_array(arr, mask)

    ims = plt.imshow(arr, origin='lower', extent=np.array(lims).flatten(), **kwargs)
    extent = ims.get_extent()

    if(contour):
        hist_contour(x, y, lims, cmap=plt.cm.Greys, color='none', reso=100, sig_arr=[0.5, 1.5], filter_sigma=5, alpha=0.25, filled=True)

    if(show_number):
        for i in np.arange(bins[0]):
            for j in np.arange(bins[1]):
                if(num[j, i] >= statmin):
                    tx, ty = (extent[0] + (extent[1] - extent[0]) * i / bins)[0], \
                             (extent[2] + (extent[3] - extent[2]) * j / bins)[-1]
                    if(statmode == 'mean'):
                        std = stat('std')
                        text = (fmt+"±"+fmt+"\nN = %d") % (arr[j, i], std[j, i], num[j, i])
                        plt.text(tx, ty, text, fontsize=fontsize, ha='left', va='bottom')
                    elif(statmode == 'median'):
                        if(num[j, i] == 0):
                            print(arr[j, i])
                        u, l = stat('quantile', 0.75) - stat(statmode), stat(statmode) - stat('quantile', 0.25)
                        text = (fmt+"$^{+"+fmt+"}_{-"+fmt+"}"+"$\nN = %d") % (arr[j, i], u[j, i], l[j, i], num[j, i])
                        plt.text(tx, ty, text, fontsize=fontsize, ha='left', va='bottom')

    return ims


def draw_function(fun, xlim, invert=False, **kwargs):
    xarr = np.linspace(*xlim, num=1000)
    if(not invert):
        plt.plot(xarr, fun(xarr), **kwargs)
    else:
        plt.plot(fun(xarr), xarr, **kwargs)

def linear_regression(x, y, err=None, xarr=[-1000, 1000], invert=False, **kwargs):
    def chisq(y, a, b):
        return a * y + b
    xarr = np.array(xarr)

    if(invert):
        cof = curve_fit(chisq, y, x, sigma=err)[0]
        plt.plot(chisq(xarr, *cof), xarr, **kwargs)
    else:
        cof = curve_fit(chisq, x, y, sigma=err)[0]
        plt.plot(xarr, chisq(xarr, *cof), **kwargs)


def medplot(x, y, binarr, minnum=1, xyinv=False, line='med', face='qua', errbar=None, color=None, **kwargs):
    # outdated, use binned_plot
    if(xyinv):
        x, y = np.array(y), np.array(x)
    else:
        x, y = np.array(x), np.array(y)
    table = []
    for bot, top in zip(binarr[:-1], binarr[1:]):
        mask = (bot <= x) & (x < top)
        if(np.sum(mask)>=minnum):
            xbet = (bot+top)/2
            yseg = y[mask]
            med = np.median(yseg)
            mean = np.mean(yseg)

            std = np.std(yseg)
            stdm = std / np.sqrt(np.sum(mask))

            uqua = np.percentile(yseg, 75)
            lqua = np.percentile(yseg, 25)

            table.append([xbet, med, mean, uqua, lqua, std, stdm])

    table = np.array(table).T
    if line is not None:
        if(line == 'med'):
            c = table[1]
        elif(line == 'mean'):
            c = table[2]
        else:
            raise ValueError("Unknown line mode: ", line)
        xc = table[0]

        if(xyinv):
            p = plt.plot(c, xc, color=color, **kwargs)
        else:
            p = plt.plot(xc, c, color=color, **kwargs)
        if(color is None):
            color = p[0].get_color()

    if face is not None:
        if(face == 'qua'):
            le, ue = table[3], table[4]
        elif(face == 'std'):
            le, ue = c - table[5], c + table[5]
        elif(face == 'stdm'):
            le, ue = c - table[6], c + table[6]
        else:
            raise ValueError("Unknown face mode:", face)

        kwargs_cen = remove_keys(kwargs, ['alpha', 'lw', 'marker', 'label', 'zorder'])
        if(xyinv):
            p = plt.fill_betweenx(xc, le, ue, alpha=0.2, lw=0, zorder=-10, color=color, **kwargs_cen)
        else:
            p = plt.fill_between(xc, le, ue, alpha=0.2, lw=0, zorder=-10, color=color, **kwargs_cen)

        color = p[0].get_color()

    if errbar is not None:
        if(errbar == 'qua'):
            le, ue = c - table[3], table[4] - c
        elif(errbar == 'std'):
            le, ue = table[5], table[5]
        elif(errbar == 'stdm'):
            le, ue = table[6], table[6]
        else:
            raise ValueError("Unknown errorbar mode:", errbar)

        kwargs_cen = remove_keys(kwargs, ['alpha', 'lw', 'label', 'zorder'])
        if(xyinv):
            p = plt.errorbar(c, xc, yerr=None, xerr=[le, ue], color=color, **kwargs_cen)
        else:
            p = plt.errorbar(xc, c, yerr=[le, ue], color=color, **kwargs_cen)
        if(color is None):
            color = p[0].get_color()

    return xc, c

def binned_plot(x, y, weights=None, errors=None, bins=10, weighted_binning=False, mode=None, xmode='mean',
                errmode=None, xerrmode=None, error_dict=None, min_stat=1, **kwargs):
    if error_dict is None:
        error_dict = {}
    if errmode is None:
        errmode = ['sigma', 'face']
    if mode is None:
        mode = ['median', 'line']
    if(weights is None):
        weights = np.full_like(y, 1.)
    key = np.argsort(x)
    x = x[key]
    y = y[key]
    weights = weights[key]

    if(isinstance(bins, int)):
        q = np.linspace(0, 1, bins+1)
        if(weighted_binning):
            bins = weighted_quantile(x, q, sample_weight=None)
        else:
            bins = np.quantile(x, q)

    bins = np.array(bins)
    nbins = bins.size-1

    bins_idx = np.searchsorted(x, bins)
    bins_idx[-1] += 1

    xarr, yarr = [], []
    xerr, yerr = [], []
    for ibin, ibot, itop in zip(np.arange(0, nbins), bins_idx[:-1], bins_idx[1:]):
        if(itop - ibot < min_stat):
            continue
        x_slice = x[ibot:itop]
        y_slice = y[ibot:itop]
        w_slice = weights[ibot:itop]
        if(errors is not None):
            e_slice = errors[ibot:itop]
        if(np.sum(w_slice) == 0):
            continue

        if(xmode == 'mean'):
            xbin = np.average(x_slice, weights=w_slice)
        elif(xmode == 'center'):
            xbin = (bins[ibin] + bins[ibin+1])/2
        else:
            raise ValueError("Unknown xmode: ", xmode)
        xarr.append(xbin)

        if(mode[0] == 'mean'):
            ybin = np.average(y_slice, weights=w_slice)
        elif(mode[0] == 'median'):
            ybin = weighted_quantile(y_slice, 0.5, sample_weight=w_slice)
        else:
            ybin = None
            raise ValueError("Unknown mode: ", mode[0])
        yarr.append(ybin)

        if(xerrmode == 'quatile'):
            xqua = weighted_quantile(x_slice, [0.25, 0.75], sample_weight=w_slice)
            xe = np.abs(xqua - xbin)
        elif(xerrmode in ['sigma', '1sigma']):
            sig = 0.682689492137086
            xqua = weighted_quantile(x_slice, [0.5-sig/2, 0.5+sig/2], sample_weight=w_slice)
            xe = np.abs(xqua - xbin)
        elif (xerrmode in ['2sigma']):
            sig = 0.954499736103642
            xqua = weighted_quantile(x_slice, [0.5 - sig / 2, 0.5 + sig / 2], sample_weight=w_slice)
            xe = np.abs(xqua - xbin)
        elif(xerrmode == 'std'):
            xstd = weighted_std(x_slice, weights=w_slice)
            xe = [xstd, xstd]
        else:
            xe = None
        if(xe is not None):
            xerr.append(xe)
        else:
            xerr = None

        if(errmode[0] == 'quatile'):
            yqua = weighted_quantile(y_slice, [0.25, 0.75], sample_weight=w_slice)
            ye = np.abs(yqua - ybin)
        elif(errmode[0] in ['sigma', '1sigma']):
            sig = 0.68269
            yqua = weighted_quantile(y_slice, [0.5-sig/2, 0.5+sig/2], sample_weight=w_slice)
            ye = np.abs(yqua - ybin)
        elif(errmode[0] == 'std'):
            ystd = weighted_std(y_slice, weights=w_slice)
            ye = [ystd, ystd]
        elif(errmode[0] == 'std_mean'):
            # standard deviation of mean
            ystd = weighted_std(y_slice, weights=w_slice)/np.sqrt(y_slice.size)
            ye = [ystd, ystd]
        elif (errmode[0] == 'std_binomial'):
            # standard deviation of binomial function.
            # y must be a value between 0 and 1, we assume mean(y) as p, ignores weights.
            ystd = np.sqrt(np.average(y_slice) * (1-np.average(y_slice)) / (itop - ibot))
            ye = [ystd, ystd]
        elif (errmode[0] == 'std_median'):
            # standard deviation of binomial function.
            # y must be a value between 0 and 1, we assume mean(y) as p, ignores weights.
            ystd = weighted_std(y_slice, weights=w_slice)/np.sqrt(y_slice.size) * 1.2533
            ye = [ystd, ystd]
        else:
            ye = None

        if(ye is not None):
            yerr.append(ye)
        else:
            yerr = None

    xarr = np.array(xarr)
    yarr = np.array(yarr)
    if(xerr is not None):
        xerr = np.array(xerr).T
    if(yerr is not None):
        yerr = np.array(yerr).T

    if(mode[1] == 'line'):
        p0 = plt.plot(xarr, yarr, **kwargs)
    elif(mode[1] in ['marker', 'scatter', 'point']):
        p0 = plt.scatter(xarr, yarr, **kwargs)
    else:
        p0 = None

    if(p0 is not None):
        if(mode[1] == 'line'):
            color = p0[0].get_color()
        elif(mode[1] in ['marker', 'scatter', 'point']):
            color = p0.get_edgecolor()[0]
    else:
        color = None

    if(errmode[1] in ['face', 'filled']):
        plt.fill_between(xarr, yarr-yerr[0], yarr+yerr[1], color=color, alpha=0.25, linewidth=0, **error_dict)
    elif(errmode[1] in ['bar', 'errorbar', 'errbar']):
        plt.errorbar(xarr, yarr, yerr=yerr, xerr=xerr, color=color, linewidth=0., **error_dict)
    elif (errmode[1] in ['arrow']):
        p, c, b = plt.errorbar(xarr, yarr, yerr=yerr, xerr=xerr, color=color, linewidth=0., **error_dict)
        c[0].set_marker('v')
        c[1].set_marker('^')
    elif(errmode[1] == 'line'):
        plt.plot(xarr, yarr-yerr[0], color=color, linewidth=0.5, **error_dict)
        plt.plot(xarr, yarr+yerr[1], color=color, linewidth=0.5, **error_dict)
    return p0

def avgplot(x, y, binarr, minnum=1, stdmean=False, face=True, **kwargs):
    x, y = np.array(x), np.array(y)
    avg, std = [], []
    xbet = []
    for bot, top in zip(binarr[:-1], binarr[1:]):
        mask = (bot <= x) & (x < top)
        masknum = np.sum(mask)
        if(masknum>=minnum):
            yseg = y[mask]
            avg.append(np.average(yseg))
            if(stdmean):
                std.append(np.std(yseg)/np.sqrt(masknum))
            else:
                std.append(np.std(yseg))
            xbet.append((bot+top)/2)
    avg, std = np.array(avg), np.array(std)

    if(face):
        kwargs_cen = remove_keys(kwargs, ['alpha', 'lw', 'label'])
        plt.fill_between(xbet, avg-std, avg+std, alpha=0.2, lw=0, zorder=-10, **kwargs_cen)

    plt.plot(xbet, avg, **kwargs)


def gridplot(nrows, ncols, xlims=None, ylims=None, xshow=[], yshow=[], log=None, nogrid=[], fig=None, xlabel=None, ylabel=None, numpanel=None, labpanel=None, panlabcolor='k', panlabsize=12, **kwargs):
    # use Gridplot more conviniently
    # xlims, ylims: common limits for each axis
    # xshow, yshow: list of indices to show ticklabels.
    # if fig=None graps current figure.
    # numpanel: attach the number/alphabet to each panel, ex) ['left', 'bottom', 'lower'] denotes the lowercase label on the left-bottom side of the panel.

    if fig is None:
        fig = plt.gcf()

    grid = plt.GridSpec(nrows, ncols, **kwargs)

    xshow, yshow = np.array(xshow), np.array(yshow)
    nogrid = np.array(nogrid)
    numlist = None

    if (numpanel is not None):
        if (len(numpanel) > 2):
            if (numpanel[2] == 'lower'):
                numlist = list(string.ascii_lowercase)
            elif (numpanel[2] == 'upper'):
                numlist = list(string.ascii_lowercase)
            elif (numpanel[2] == 'digits'):
                numlist = np.arange(1, 10)
            else:
                ('Error: numpanel cannot be understood.')
        else:
            numlist = list(string.ascii_lowercase)

    for ir in range(nrows):
        for ic in range(ncols):
            num = ir*ncols + ic

            if(np.any(nogrid==num)):
                ax = fig.add_subplot(grid[num], frameon=False)
                ax.tick_params(labelcolor='w', which='both', top='off', bottom='off', left='off', right='off')

            else:
                ax = fig.add_subplot(grid[num])
                ax.tick_params(which='both', direction='in', top='on', bottom='on', left='on', right='on')

                if(ic != 0 and np.all(yshow!=ic)):
                    ax.set_yticklabels([])
                if(ir != nrows-1 and np.all(xshow!=ir)):
                    ax.set_xticklabels([])

                if(log=='x'):
                    ax.set_xscale('log')
                elif(log=='y'):
                    ax.set_yscale('log')
                elif(log=='xy'):
                    ax.set_xscale('log')
                    ax.set_yscale('log')

                if(xlims is not None):
                    ax.set_xlim(xlims[ic])
                if(ylims is not None):
                    ax.set_ylim(ylims[ir])

                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                if(numpanel is not None):
                    if(numpanel[0] == 'left'):
                        tx = xlim[0] + (xlim[1] - xlim[0]) * 0.05
                    else:
                        tx = xlim[1] - (xlim[1] - xlim[0]) * 0.05

                    if(numpanel[1] == 'bottom'):
                        ty = ylim[0] + (ylim[1] - ylim[0]) * 0.05
                    else:
                        ty = ylim[1] - (ylim[1] - ylim[0]) * 0.05

                    ax.text(tx, ty, '(%s)' % numlist[num], ha=numpanel[0], va=numpanel[1], fontsize=panlabsize, color=panlabcolor)

                if(labpanel is not None):
                    if(labpanel[0] == 'left'):
                        tx = xlim[0] + (xlim[1] - xlim[0]) * 0.05
                    else:
                        tx = xlim[1] - (xlim[1] - xlim[0]) * 0.05

                    if(labpanel[1] == 'bottom'):
                        ty = ylim[0] + (ylim[1] - ylim[0]) * 0.05
                    else:
                        ty = ylim[1] - (ylim[1] - ylim[0]) * 0.05

                    ax.text(tx, ty, '%s' % labpanel[2][num], ha=labpanel[0], va=labpanel[1], fontsize=panlabsize, color=panlabcolor)


    if(xlabel is not None or ylabel is not None):
        base = fig.add_subplot(111, frameon=False)

        base.tick_params(which='both', top='off', bottom='off', left='off', right='off')
        base.set_xticklabels([])
        base.set_yticklabels([])
        print(xlabel)

        if(xlabel is not None):
            base.set_xlabel(xlabel)
        if(ylabel is not None):
            base.set_ylabel(ylabel)

    return grid

def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    isalpha = len(colors[0])==4
    cdict = {'red':[], 'green':[], 'blue':[]}
    if isalpha: cdict['alpha'] = []
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
        if isalpha: cdict['alpha'].append((pos, color[3], color[3]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

def load_cmap(path):
    data = np.loadtxt(path, delimiter=",", skiprows=4).tolist()
    center_emph = mpl.colors.ListedColormap((data[:256])[10:-10], name="center_emph")
    return center_emph

def dark_cmap(color):
    color_bright = np.array(color)+0.25
    color_bright[color_bright>1] = 1
    return make_cmap([[0, 0, 0], color, color_bright], position=[0, 0.5, 1])        

def grid_projection(centers, levels=None, quantities=None, weights=None, shape=None, lims=None,
                    mode='sum', plot_method='hist', projector_kwargs={}, projection=['x', 'y'],
                    interp_order=0, crop_mode='subpixel', output_dtype=np.float64, padding=0,
                    type='particle'):
    """
    Generate a 2D projection plot of a quantity using particle or AMR data.

    Parameters:
    -----------
    centers : np.ndarray
        Array of shape (N, 3) containing the coordinates of the cell centers or particles.
    levels : np.ndarray, optional
        Array of shape (N,) containing the refinement levels of the cells. Required for AMR data.
    quantities : np.ndarray, optional
        Array of shape (N,) containing the quantity values to be projected. Default is None.
    weights : np.ndarray, optional
        Array of shape (N,) containing the weights for each cell or particle. If None, all weights are set to 1. Default is None.
    shape : int or tuple of int, optional
        Shape of the output grid. If an integer is provided, it is used for both dimensions. Default is None.
    lims : list of list of float, optional
        Limits for the projection in the form [[xmin, xmax], [ymin, ymax], [zmin, zmax]]. If None, defaults to [[0, 1], [0, 1], [0, 1]]. Default is None.
    mode : str, optional
        Mode of projection. Options are 'sum', 'mean', 'min', 'max'. Default is 'sum'.
    plot_method : str, optional
        Method for plotting. Options are 'hist' for histogram and 'cic' for Cloud-In-Cell. Default is 'hist'.
    projection : list of str, optional
        Axes to project onto. Default is ['x', 'y'].
    interp_order : int, optional
        Order of interpolation for rescaling. Default is 0.
    crop_mode : str, optional
        Mode for cropping the image to fit the output to be within desired limits. Options are 'grid', 'pixel', 'subpixel'. Default is 'subpixel'.
        'grid': crop the image based on the current minimum resolution of the grid.
        'pixel': crop the image based on the pixel size of the drawing image.
        'subpixel': crop the image with allowing subpixel cropping.
    output_dtype : dtype, optional
        Data type of the output grid. Default is np.float64.
    padding : int, optional
        Number of padding pixels for drawing the data. Default is 0.
    type : str, optional
        Type of data. Options are 'particle' or 'amr'. Default is 'particle'.

    Returns:
    --------
    grid : np.ndarray
        2D array representing the projected quantity.
    """
    def apply_projection(grid, grid_weight, x, y, quantity, weights, lims_2d, projector, mode='sum'):
        shape = grid.shape
        if mode in ['sum', 'mean']:
            # do a weighted sum over projection
            grid += projector(x, y, quantity*weights, lims_2d, shape)
            if mode in ['mean']:
                grid_weight += projector(x, y, weights, lims_2d, shape)
        elif mode in ['min', 'max']:
            xi = uniform_digitize(x, lims_2d[0], shape[0]) - 1
            yi = uniform_digitize(y, lims_2d[1], shape[1]) - 1
            if mode in ['min']:
                # do a minimum over projection
                np.minimum.at(grid, (xi, yi), quantity)
            elif mode in ['max']:
                # do a maximum over projection
                np.maximum.at(grid, (xi, yi), quantity)

    ndim = 3
    levelmin, levelmax = None, None
    # if lims is None, set all limits to [0, 1]
    if lims is None:
        lims = [[0, 1],] * ndim
    lims = np.array(lims)
    # if quantities is None, set all quantities to 1
    if quantities is None:
        quantities = np.ones(centers.shape[0])
    # if weights is None, set all weights to 1
    if weights is None:
        weights = np.ones_like(quantities)
    # if shape is scalar, make it a tuple
    if np.isscalar(shape):
        shape = tuple(np.repeat(shape, 2))
    # if number of arrays does not match the number of quantities, raise an error
    if centers.shape[0] != len(quantities):
        raise ValueError("The number of centers and quantities do not match.")
    # if number of arrays does not match the number of weights, raise an error
    if weights is not None and centers.shape[0] != len(weights):
        raise ValueError("The number of centers and weights do not match.")
    if plot_method != 'hist' and mode in ['min', 'max']:
        warnings.warn("plot_method is disabled when mode is min, max")

    if type in ['particle', 'part']:
        type = 'part'
    elif type in ['amr', 'cell', 'grid']:
        type = 'amr'

    dim_keys = np.array(['x', 'y', 'z'][:ndim])
    proj = [np.arange(ndim)[dim_keys==p][0] for p in projection]
    
    # get the z-axis that is not in the projection
    proj_z = np.setdiff1d(np.arange(ndim), proj)[0]
    lims_2d = np.array([lims[proj[0]], lims[proj[1]]])

    if mode in ['sum', 'mean']:
        fill_value = 0
    elif mode in ['min']:
        fill_value = np.inf
    elif mode in ['max']:
        fill_value = -np.inf

    pixel_size = (lims_2d[:, 1] - lims_2d[:, 0]) / np.array(shape)
    if padding != 0:
        padding_size = np.concatenate([pixel_size * padding, [0]])
    else:
        padding_size = 0

    if type == 'part':
        if shape is None:
            shape = (100, 100)
        # get mask that indicates particles to draw
        mask_draw = box_mask(centers, lims, size=padding_size)
        shape_grid = shape

    elif type == 'amr':
        if levels is None:
            raise ValueError("Levels must be provided for AMR data.")
        if centers.shape[0] != len(levels):
            raise ValueError("The number of centers and levels do not match.")
        levelmin, levelmax = np.min(levels), np.max(levels)

        if shape is not None:
            # get the levels of the grid to draw the desired resolution
            dx_min = np.minimum((lims_2d[0, 1] - lims_2d[0, 0]) / shape[0], (lims_2d[1, 1] - lims_2d[1, 0]) / shape[1])
            levelmax_draw = np.minimum(np.ceil(-np.log2(dx_min)).astype(int), levelmax)
        else:
            # if shape is not specified, draw with the full resolution
            levelmax_draw = levelmax
            dx_min = 2. ** -levelmax_draw
            shape = (lims_2d[0, 1] - lims_2d[0, 0]) // dx_min, (lims_2d[1, 1] - lims_2d[1, 0]) // dx_min
            if np.prod(shape) >= 1E8:
                warnings.warn("The shape of the grid is too large: {shape}, it may cause memory issues.")
        levelmin_draw = levelmin

        # get the smallest levelmin grid space that covers the whole region
        i_lims_levelmin = lims_2d * 2**levelmin_draw
        i_lims_levelmin[:, 0] = np.floor(i_lims_levelmin[:, 0])
        i_lims_levelmin[:, 1] = np.ceil(i_lims_levelmin[:, 1])
        i_lims_levelmin = i_lims_levelmin.astype(int)

        shape_grid = tuple(i_lims_levelmin[:, 1] - i_lims_levelmin[:, 0])
        lims_2d_draw = i_lims_levelmin / 2**levelmin_draw

        # get mask that indicates cells to draw
        mask_draw = box_mask(centers, lims, size=(0.5**levels)[..., np.newaxis]+padding_size)
        ll = levels[mask_draw]
    else:
        raise ValueError("Unknown type: %s. Supported types are 'part' and 'amr'." % type)

    # initialize grid and grid_weight
    grid = np.full(shape_grid, fill_value, dtype=output_dtype)
    if mode in ['mean']:
        grid_weight = np.zeros(shape_grid, dtype=output_dtype)
    else:
        grid_weight = None

    # apply mask based on the current drawing scope
    cc = centers[mask_draw]
    qq = quantities[mask_draw]
    ww = weights[mask_draw]

    # get the projected coordinates
    xx, yy = cc[:, proj[0]], cc[:, proj[1]]
    zz = cc[:, proj_z]

    # set the projector lambda based on the plot method
    projector = lambda x, y, w, lims, shape: estimate_density_2d(x, y, lims=lims, weights=w, shape=shape, density=False, method=plot_method, **projector_kwargs)

    # do projection
    if type == 'part':
        pixel_area = np.prod((lims_2d[:, 1] - lims_2d[:, 0]) / np.array(shape))

        # do projection onto current grid
        apply_projection(grid=grid, grid_weight=grid_weight, x=xx, y=yy, quantity=qq, weights=ww, lims_2d=lims_2d, projector=projector, mode=mode)
        grid /= pixel_area

        if mode == 'mean' and grid_weight is not None:
            grid /= grid_weight

    elif type == 'amr' and levelmin is not None and levelmax is not None:
        for grid_level in range(levelmin, levelmax+1):
            mask_level = ll == grid_level
            shape_now = grid.shape
            x, y, z, q = xx[mask_level], yy[mask_level], zz[mask_level], qq[mask_level]

            volume_weight = 1
            # get weight for the current level with depending on the line-of-sight depth within the limit.
            # e.g., the weight is 0.5 if the z-coordinate is in the middle of any z-limits 
            volume_weight += np.clip((z - lims[proj_z][0]) * 2**grid_level - 0.5, -1, 0) + np.clip((lims[proj_z][1] - z) * 2**grid_level - 0.5, -1, 0)
            # multiply the weight by the depth of the projected grid. The weight is doubled per each decreasing level except for the levels larger than the grid resolution.
            volume_weight *= 0.5**grid_level
            # give additional weight to the projection if cell is smaller than the grid resolution
            volume_weight *= 0.25**np.maximum(0, grid_level - levelmax_draw)
            w = ww[mask_level] * volume_weight

            # do projection onto current level grid
            apply_projection(grid=grid, grid_weight=grid_weight, x=x, y=y, quantity=q, weights=w, lims_2d=lims_2d_draw, projector=projector, mode=mode)

            # increase grid size if necessary
            if grid_level >= levelmin_draw and grid_level < levelmax_draw:
                grid = rescale(grid, 2, order=interp_order)
                if mode in ['mean']:
                    grid_weight = rescale(grid_weight, 2, order=interp_order)

        if mode == 'mean' and grid_weight is not None:
            grid /= grid_weight
        # resize and crop image to the desired shape
        if shape is not None:
            if crop_mode == 'grid':
                grid = resize(grid, output_shape=shape, order=interp_order)
            elif crop_mode in ['pixel', 'subpixel']:
                subpixel = crop_mode == 'subpixel'
                lims_crop = (lims_2d - lims_2d_draw[:, 0, np.newaxis]) / (lims_2d_draw[:, 1] - lims_2d_draw[:, 0])[:, np.newaxis]
                grid = crop(grid, range=lims_crop, output_shape=shape, subpixel=subpixel, order=interp_order)

    return grid

def part_projection(centers, quantities=None, weights=None, shape=100, lims=None,
                    mode='sum', plot_method='hist', projection=['x', 'y'], output_dtype=np.float64, padding=0):
    """
    Generate a 2D projection plot of a quantity using particle data.

    Parameters:
    -----------
    centers : np.ndarray
        Array of shape (N, 3) containing the coordinates of particles.
    quantities : np.ndarray
        Array of shape (N,) containing the quantity values to be projected.
    weights : np.ndarray, optional
        Array of shape (N,) containing the weights for each cell. If None, all weights are set to 1. Default is None.
    shape : int or tuple of int, optional
        Shape of the output grid. If an integer is provided, it is used for both dimensions. Default is 100.
    lims : list of list of float, optional
        Limits for the projection in the form [[xmin, xmax], [ymin, ymax], [zmin, zmax]]. If None, defaults to [[0, 1], [0, 1], [0, 1]]. Default is None.
    mode : str, optional
        Mode of projection. Options are 'sum', 'mean', 'min', 'max'. Default is 'sum'.
    plot_method : str, optional
        Method for plotting. Options are 'hist' for histogram and 'cic' for Cloud-In-Cell. Default is 'hist'.
    projection : list of str, optional
        Axes to project onto. Default is ['x', 'y'].
    interp_order : int, optional
        Order of interpolation for rescaling. Default is 0.

    Returns:
    --------
    grid : np.ndarray
        2D array representing the projected quantity.
    """
    return grid_projection(centers=centers, levels=None, quantities=quantities, weights=weights, shape=shape, lims=lims, mode=mode, plot_method=plot_method, projection=projection, output_dtype=output_dtype, padding=padding, type='particle')

def amr_projection(centers, levels, quantities=None, weights=None, shape=None, lims=None,
                   mode='sum', plot_method='hist', projection=['x', 'y'], interp_order=0, output_dtype=np.float64, padding=0, crop_mode='subpixel'):
    """
    Generate a 2D projection plot of a quantity using Adaptive Mesh Refinement (AMR) data.

    Parameters:
    -----------
    centers : np.ndarray
        Array of shape (N, 3) containing the coordinates of the cell centers. Should be within the range of (0, 1).
    levels : np.ndarray
        Array of shape (N,) containing the refinement levels of the cells.
    quantities : np.ndarray
        Array of shape (N,) containing the quantity values to be projected.
    weights : np.ndarray, optional
        Array of shape (N,) containing the weights for each cell. If None, all weights are set to 1. Default is None.
    shape : int or tuple of int, optional
        Shape of the output grid. If an integer is provided, it is used for both dimensions. Default is None.
    lims : list of list of float, optional
        Limits for the projection in the form [[xmin, xmax], [ymin, ymax], [zmin, zmax]]. If None, defaults to [[0, 1], [0, 1], [0, 1]]. Default is None.
    mode : str, optional
        Mode of projection. Options are 'sum', 'mean', 'min', 'max'. Default is 'sum'.
    plot_method : str, optional
        Method for plotting. Options are 'hist' for histogram and 'cic' for Cloud-In-Cell. Default is 'hist'.
    projection : list of str, optional
        Axes to project onto. Default is ['x', 'y'].
    interp_order : int, optional
        Order of interpolation for rescaling. Default is 0.

    Returns:
    --------
    grid : np.ndarray
        2D array representing the projected quantity.
    """
    return grid_projection(centers=centers, levels=levels, quantities=quantities, weights=weights, shape=shape, lims=lims, mode=mode, plot_method=plot_method, projection=projection, interp_order=interp_order, crop_mode=crop_mode, output_dtype=output_dtype, padding=padding, type='amr')


def crop(img, range, output_shape=None, subpixel=True, **kwargs):
    """
    Crop an image to a specified range.    
    """
    range = np.array(range)
    shape = np.array(img.shape)
    idx_true = shape[:, np.newaxis] * range
    if not subpixel:
        idx_int = np.array(np.round(idx_true), dtype=int)
        #idxs = np.array([np.round(shape[0] * range[0] - 0.5), np.round(shape[1] * range[1] - 0.5)], dtype=int)
        img = img[idx_int[0, 0]:idx_int[0, 1], idx_int[1, 0]:idx_int[1, 1]]
        if output_shape is not None:
            img = resize(img, output_shape, **kwargs)
    else:
        # subpixel crop using pixel interpolation
        # only works for square image and ranges
        true_shape = idx_true[:, 1] - idx_true[:, 0]

        if output_shape is None:
            output_shape = np.round(true_shape).astype(int)
        else:
            output_shape = np.array(output_shape)

        scale = true_shape / output_shape

        tform1 = EuclideanTransform(translation=idx_true[:, 0] - 0.5)
        tform2 = AffineTransform(scale=scale)
        # need to trnspose the image to apply the transformation correctly
        img = warp(img.T, tform2+tform1, output_shape=output_shape[::-1], **kwargs).T

    return img

class ccm:
    # write custom colormaps here
    TrueRed = make_cmap([(1, 1, 1), (1, 0, 0)])
    TrueGreen = make_cmap([(1, 1, 1), (0, 1, 0)])
    TrueBlue = make_cmap([(1, 1, 1), (0, 0, 1)])

    DarkRed = dark_cmap([1, 0, 0])
    DarkBlue = dark_cmap([0, 0, 1])
    DarkGreen = dark_cmap([0, 1, 0])

    DarkCyan = dark_cmap([0, 1, 1])
    DarkMagenta = dark_cmap([1, 0, 1])
    DarkYellow = dark_cmap([1, 1, 0])

    DarkOcean = dark_cmap([1, 1, 0])

    darkmatter = make_cmap([[0, 0, 0], [0, 0, 0.5], [0.5, 0, 1], [0.75, 0.5, 1], [1, 1, 1]], position=[0, 0.5, 0.75, 0.95, 1])
    forest = make_cmap([[0, 0, 0], [0.25, 0.25, 0.1], [0.25, 0.5, 0.1], [0.5, 1, 0.5], [0.25, 1, 0.75]], position=[0, 0.55, 0.7, 0.85, 1])
    oldstar = make_cmap([[0,0,0], [0.274, 0.239, 0.176], [0.588, 0.470, 0.380], [0.824, 0.729, 0.663], [0.996, 0.945, 0.949]], position=[0, 0.25, 0.5, 0.75, 1])

    mypath = dirname(abspath(__file__))

    cmap_dir = pkg_resources.resource_filename('rur', 'colormaps/data/')
    hesperia = load_cmap(join(cmap_dir, 'hesperia.csv'))
    laguna = load_cmap(join(cmap_dir, 'laguna.csv'))
    lacerta = load_cmap(join(cmap_dir, 'lacerta.csv'))
    mod_plasma = load_cmap(join(cmap_dir, 'mod_plasma.csv'))
    arctic = load_cmap(join(cmap_dir, 'arctic.csv'))
    sunburst = load_cmap(join(cmap_dir, 'sunburst.csv'))
    amber = load_cmap(join(cmap_dir, 'amber.csv'))