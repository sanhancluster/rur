from pylab import *

from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
from rur.utool import *
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import norm
from scipy.signal import convolve2d
from numpy.linalg import det
from skimage.transform import resize

import string
import matplotlib.collections as mcoll
from os.path import dirname, join, abspath

def colorline(
        x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0, zorder=0):
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


def axlabel(text, pos='right top', offset=0.03, **kwargs):
    # ha: right, left
    # va: top, bottom

    ha, va = tuple(pos.split(' '))
    if(ha == 'right'):
        x = 1 - offset
    elif(ha == 'left'):
        x = offset
    elif (ha == 'center'):
        x = 0.5
    else:
        raise ValueError("Unknown horizontal position %s" % ha)
    if(va == 'top'):
        y = 1 - offset
    elif(va == 'bottom'):
        y = offset
    elif(va == 'center'):
        y = 0.5
    else:
        raise ValueError("Unknown vertical position %s" % va)
    plt.text(x, y, text, ha=ha, va=va, transform=plt.gca().transAxes, **kwargs)


def bmask(arr, bet):
    return (bet[0] <= arr) & (arr < bet[1])


def kde_imshow(x, y, lims=None, reso=100, weights=None, tree=True, **kwargs):
    if(lims is None):
        lims = [(np.nanquantile(x, 0.001), np.nanquantile(x, 0.999)),
                (np.nanquantile(y, 0.001), np.nanquantile(y, 0.999))]
        print('Automatically setting lims as ', lims)

    pdf = kde_img(x, y, lims, reso, weights=weights, tree=tree)
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

    pdi = np.histogram2d(x, y, range=lims, bins=reso, weights=weights, normed=True)[0].T
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

    pdi = kde_img(x, y, lims, reso, bw_method, weights=weights)
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

    pdi = np.histogram2d(x, y, range=lims, bins=reso, weights=weights, normed=True)[0].T
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
    sigma_int = np.int(sigma)
    fraction = sigma - sigma_int
    return gaussian_filter(image, sigma_int, **kwargs) * fraction + gaussian_filter(image, sigma_int+1, **kwargs) * (1-fraction)

def gauss_img(x, y, lims, reso=100, weights=None, subdivide=3, kernel_size=1):
    x, y = np.array(x), np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if(weights is not None):
        weights = weights[mask]

    kern_size = int(kernel_size*subdivide*6) - 1

    arr = bin_centers(-kern_size/2, kern_size/2, kern_size)
    xm, ym = np.meshgrid(arr, arr)
    mesh = np.stack([xm, ym], axis=-1)
    dist = rss(mesh)

    kern = norm.pdf(dist, scale=kernel_size*subdivide)
    kern /= np.sum(kern)

    hist = np.histogram2d(x, y, bins=reso*subdivide, range=lims, weights=weights)[0]
    hist = convolve2d(hist, kern, mode='same')
    hist = resize(hist, reso)*subdivide**2

    return hist

def kde_img(x, y, lims, reso=100, weights=None, tree=True):
    x, y = np.array(x), np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if(weights is not None):
        weights = weights[mask]

    if(tree):
        kde = gaussian_kde_tree(np.stack([x, y], axis=-1), weights=weights, smooth_factor=3)
        return fun_img(kde, lims, reso, axis=-1)
    else:
        kde = gaussian_kde(np.stack([x, y], axis=0), weights=weights)
        return fun_img(kde, lims, reso, axis=0)

def fun_img(f, lims, reso=100, axis=-1):
    # returns 2d numpy array image with function
    # axis: the axis that function accepts to separate each dimensions.
    if(np.isscalar(reso)):
        reso = np.repeat(reso, 2)

    xarr = bin_centers(lims[0][0], lims[0][1], reso[0])
    yarr = bin_centers(lims[1][0], lims[1][1], reso[1])

    xm, ym = np.meshgrid(xarr, yarr)

    mesh = np.stack([xm.ravel(), ym.ravel()], axis=axis)

    pdi = f(mesh)
    pdi = np.reshape(pdi, xm.shape)

    return pdi

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


def dtfe_img(x, y, lims, reso=100, weights=None, smooth=0):
    if (np.isscalar(reso)):
        reso = np.repeat(reso, 2)

    points = np.stack([x, y], axis=-1)
    center = np.median(points, axis=0)
    n_points = points.shape[0]

    if(smooth is None):
        smooth = np.int(0.05 * n_points**0.6)

    # For some "Complex Geometrical Reasons", Qhull does not work properly without options???
    # Even with blank option, the result is different.
    tri = Delaunay(points-center, qhull_options='')

    simplices = tri.simplices
    vertices = points[simplices]

    matrices = np.insert(vertices, 2, 1., axis=-1)
    matrices = np.swapaxes(matrices, -1, -2)
    tri_areas = np.abs(det(matrices)) / 2

    hull_areas = np.zeros(n_points, dtype='f8')

    np.add.at(hull_areas, simplices, np.expand_dims(tri_areas, -1))
    hull_areas /= 3

    if(smooth>0):
        indptr, neighbor_indices = tri.vertex_neighbor_vertices
        neighbor_nums = np.full(n_points, 0, dtype='i8')
        center_indices = np.repeat(np.arange(neighbor_nums.size), np.diff(indptr))
        np.add.at(neighbor_nums, center_indices, 1)

        for _ in np.arange(smooth):
            hull_areas_new = np.zeros(hull_areas.shape, dtype='f8')
            np.add.at(hull_areas_new, center_indices, hull_areas[neighbor_indices])
            hull_areas = hull_areas_new / neighbor_nums

    densities = 1 / hull_areas

    if(weights is not None):
        densities *= weights

    xarr = bin_centers(lims[0][0], lims[0][1], reso[0]) - center[0]
    yarr = bin_centers(lims[1][0], lims[1][1], reso[1]) - center[1]

    xm, ym = np.meshgrid(xarr, yarr)

    ip = LinearNDInterpolator(tri, densities, fill_value=0)
    grid = ip(xm, ym).T

    return grid


def mosaic_mean(x, y, v, bins=10, range=None, minnum=0, stat=False, statmin=1, fmt="%.3f", **kwargs):
    weig = np.histogram2d(x, y, bins, range, weights=v)[0].T
    norm = np.histogram2d(x, y, bins, range)[0].T
    mask = norm < minnum
    arr = np.ma.masked_array((weig/norm), mask)

    ims = plt.imshow(arr, origin='lower', extent=np.array(range).flatten(), **kwargs)
    extent = ims.get_extent()

    if(stat):
        for i in np.arange(bins):
            for j in np.arange(bins)[::-1]:
                if(norm[j, i] >= statmin):
                    plt.text(extent[0] + (extent[1]-extent[0]) * i/bins, extent[2] + (extent[3]-extent[2]) * j/bins, (fmt+"\n%d") % ((weig/norm)[j, i], norm[j, i]), fontsize=8, ha='left', va='bottom')


    return ims


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


def medplot(x, y, binarr, minnum=1, xyinv=False, face=True, err=False, **kwargs):
    if(xyinv):
        x, y = np.array(y), np.array(x)
    else:
        x, y = np.array(x), np.array(y)
    med, usig, lsig = [], [], []
    xbet = []
    for bot, top in zip(binarr[:-1], binarr[1:]):
        mask = (bot <= x) & (x < top)
        if(np.sum(mask)>=minnum):
            yseg = y[mask]
            med.append(np.percentile(yseg, 50))
            usig.append(np.percentile(yseg, 75))
            lsig.append(np.percentile(yseg, 25))
            xbet.append((bot+top)/2)

    med, usig, lsig, xbet = np.array(med), np.array(usig), np.array(lsig), np.array(xbet)

    if(face):
        kwargs_cen = remove_keys(kwargs, ['alpha', 'lw', 'label', 'zorder'])
        if(xyinv):
            plt.fill_betweenx(xbet, lsig, usig, alpha=0.2, lw=0, zorder=-10, **kwargs_cen)
        else:
            plt.fill_between(xbet, lsig, usig, alpha=0.2, lw=0, zorder=-10, **kwargs_cen)

    if(err):
        #kwargs_cen = remove_keys(kwargs, [])
        if(xyinv):
            plt.errorbar(med, xbet, yerr=None, xerr=[med-lsig, usig-med], **kwargs)
        else:
            plt.errorbar(xbet, med, yerr=[med-lsig, usig-med], **kwargs)

    if(xyinv):
        plt.plot(med, xbet, **kwargs)
    else:
        plt.plot(xbet, med, **kwargs)



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
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

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

import pkg_resources
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
