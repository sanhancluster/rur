import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import rescale, resize, warp, EuclideanTransform, AffineTransform
from rur import drawer as dr, utool
from rur.drawer import ccm
from rur import uri, uhmi
from rur.utool import Timer, get_vector, bin_centers, rss, los
from matplotlib.patches import RegularPolygon, Rectangle
from scipy.ndimage.filters import gaussian_filter1d
from collections.abc import Iterable
from itertools import repeat
import os

verbose = 1
timer = Timer(verbose=verbose)

default_box = np.array([[0, 1], [0, 1], [0, 1]])

def crop(img, range):
    range = np.array(range)
    shape = np.array(img.shape)
    idxs = np.array([np.ceil(shape[0] * range[0]), np.floor(shape[1] * range[1])], dtype=int)
    return img[idxs[0, 0]:idxs[0, 1], idxs[1, 0]:idxs[1, 1]]

def crop_float(img, range, output_shape=None):
    # subpixel crop using pixel interpolation
    # only works for square image and ranges
    range = np.array(range)
    shape = np.array(img.shape)

    idx_true = np.expand_dims(shape, -1) * range

    true_shape = idx_true[:, 1] - idx_true[:, 0]

    if output_shape is None:
        output_shape = np.round(true_shape).astype(int)
    else:
        output_shape = np.array(output_shape)

    scale = true_shape/output_shape

    # transform, rescale, crop in stupid skimage interface
    tform1 = EuclideanTransform(translation=idx_true[:, 0])
    tform2 = AffineTransform(scale=scale)
    img = warp(img.T, tform2+tform1, output_shape=output_shape[::-1], cval=1000).T

    return img

def get_box_proj(box, proj):
    if box is None:
        box = default_box
    box = np.array(box)
    box_proj = box[proj]

    return box_proj

def set_bins(known_lvls, minlvl, maxlvl, box_proj, shape):

    if(minlvl is None):
        minlvl = np.min(known_lvls)
    else:
        known_lvls = known_lvls[known_lvls >= minlvl]
        minlvl = np.max([minlvl, np.min(known_lvls)])

    if(maxlvl is None):
        maxlvl = np.max(known_lvls)
        if(shape is not None):
            pixlvl = np.max(-np.log2((box_proj[:, 1]-box_proj[:, 0]) / np.array(shape)))
            maxlvl = np.min([maxlvl, int(pixlvl)+1])
    else:
        maxlvl = np.min([maxlvl, np.max(known_lvls)])

    mingrid = np.linspace(0, 1, 2**minlvl + 1)

    # find the smallest grid box that encloses projected box region
    edgeidx = np.stack([np.searchsorted(mingrid, box_proj[:, 0], 'right')-1,
                        np.searchsorted(mingrid, box_proj[:, 1], 'left')], axis=-1)
    basebin = edgeidx[:, 1] - edgeidx[:, 0]
    edge = mingrid[edgeidx]

    if(verbose>=2):
        print('box', list(box_proj))
        print('edge', list(edge))
        print('basebin', basebin)
        print('Known levels: ', known_lvls)

    return minlvl, maxlvl, basebin, edge


def lvlmap(cell, box=None, proj=[0, 1], shape=500, minlvl=None, maxlvl=None, subpx_crop=True):
    if(box is None and isinstance(cell, uri.RamsesSnapshot.Cell)):
        box = cell.snap.box

    lvl = cell['level']

    box_proj = get_box_proj(box, proj)

    if (np.isscalar(shape)):
        shape = np.repeat(shape, 2)

    known_lvls = np.unique(lvl)
    minlvl, maxlvl, basebin, edge = set_bins(known_lvls, minlvl, maxlvl, box_proj, shape)

    known_lvls = np.arange(minlvl, maxlvl+1)

    if(verbose>=1):
        print('MinLvl = %d, MaxLvl = %d, Initial Image Size: ' % (minlvl, maxlvl), basebin * 2**(maxlvl-minlvl))
    timer.start('Drawing Refinement Level Map... ', 1)

    if(shape is None):
        shape = basebin * 2 ** (maxlvl - minlvl)

    image = np.zeros(basebin)
    for ilvl in known_lvls:
        mask = lvl==ilvl
        cell_lvl = cell[mask]
        xm = get_vector(cell_lvl)
        lm = cell_lvl['level']

        binlvl = np.min([ilvl, maxlvl])
        binsize = basebin * 2**(binlvl-minlvl)

        # convert coordinates to map
        hist_map = np.histogram2d(xm[:, proj[0]], xm[:, proj[1]], bins=binsize, range=edge, weights=lm)[0]
        hist_num = np.histogram2d(xm[:, proj[0]], xm[:, proj[1]], bins=binsize, range=edge)[0]

        np.divide(hist_map, hist_num, out=hist_map, where=hist_num!=0)
        image = np.nanmax([image, hist_map], axis=0)

        if(ilvl < maxlvl):
            image = rescale(image, 2, order=0, multichannel=False, anti_aliasing=False)

    crop_range = ((box_proj.T - edge[:, 0]) / (edge[:, 1] - edge[:, 0])).T
    if(subpx_crop):
        image = crop_float(image, crop_range, output_shape=shape)
    else:
        image = crop(image, crop_range)
        if(shape is not None):
            image = resize(image, shape, mode='constant')

    timer.record()
    if(verbose>=1):
        print("Cropped Image Size: ", image.shape)

    return image.T

def draw_lvlmap(cell, box=None, proj=[0, 1], shape=None, minlvl=None, maxlvl=None, **kwargs):
    image = lvlmap(cell, box, proj, shape=shape, minlvl=minlvl, maxlvl=maxlvl)

    if box is None:
        box = default_box
    box = np.array(box)

    box_proj = np.array(box)[proj]

    draw_image(image, np.concatenate(box_proj), vmax=maxlvl, vmin=minlvl, normmode='linear', **kwargs)

def set_weights(mode, cell, unit, depth):
    quantity = None
    if (mode == 'v'):
        # averge mass-weighted velocity along LOS
        weights = cell['rho']
    elif (mode == 'T'):
        # averge mass-weighted temperature along LOS
        weights = cell['rho']
    elif (mode == 'metal'):
        # averge mass-weighted metallicity along LOS
        weights = cell['rho']
    elif (mode == 'mach'):
        # averge mass-weighted mach number along LOS
        weights = cell['rho']
    elif (mode == 'vel'):
        # averge mass-weighted velocity along LOS
        weights = cell['rho']
        quantity = rss(cell[mode, unit])
    elif (mode == 'dust'):
        # average dust density along LOS
        weights = cell['rho']
    elif (mode == 'zoom'):
        # cumulative refinement paramster along LOS
        weights = np.full(cell.size, 1)
    elif (mode == 'rho'):
        # average density along LOS
        weights = np.full(cell.size, 1)
    elif (mode == 'crho'):
        # column density along LOS
        weights = np.full(cell.size, 1)
        quantity = cell['rho', unit] * depth
    else:
        raise ValueError('Unknown gasmap mode.')
    if(quantity is None):
        quantity = cell[mode, unit]

    return quantity, weights

def gasmap(cell, box=None, proj=[0, 1], shape=500, mode='rho', unit=None, minlvl=None, maxlvl=None, subpx_crop=True, anti_aliasing=False, interp_order=0):
    if(box is None and isinstance(cell, uri.RamsesSnapshot.Cell)):
        box = cell.snap.box

    lvl = cell['level']

    box_proj = get_box_proj(box, proj)

    if (np.isscalar(shape)):
        shape = np.repeat(shape, 2)

    if(shape is not None):
        shape = np.array(shape)

    known_lvls = np.unique(lvl)
    minlvl, maxlvl, basebin, edge = set_bins(known_lvls, minlvl, maxlvl, box_proj, shape)

    known_lvls = np.arange(minlvl, np.max(known_lvls)+1)

    if(verbose>=1):
        print('MinLvl = %d, MaxLvl = %d, Initial Image Size: ' % (minlvl, maxlvl), basebin * 2**(maxlvl-minlvl))
    timer.start('Drawing gas map... ', 1)

    if(shape is None):
        shape = basebin * 2 ** (maxlvl - minlvl)

    image = np.zeros(basebin)
    depth_map = np.zeros(basebin)
    depth = np.diff(box[los(proj)])

    for ilvl in known_lvls:
        mask = lvl==ilvl
        cell_lvl = cell[mask]

        qm, wm = set_weights(mode, cell_lvl, unit, depth)

        xm = get_vector(cell_lvl)

        binlvl = np.min([ilvl, maxlvl])
        binsize = basebin * 2**(binlvl-minlvl)

        # convert coordinates to map
        hist_weight = np.histogram2d(xm[:, proj[0]], xm[:, proj[1]], bins=binsize, range=edge, weights=wm)[0]
        hist_map = np.histogram2d(xm[:, proj[0]], xm[:, proj[1]],
                                  bins=binsize, range=edge, weights=qm*wm)[0]
        # weighted average map of quantities
        hist_map = np.divide(hist_map, hist_weight, where=hist_weight!=0)

        if (ilvl < maxlvl):
            ibin = ilvl
        else:
            ibin = ilvl*3 - maxlvl*2

        # additional depth
        add_depth = hist_weight * 0.5 ** ibin

        # new depth
        depth_map_new = depth_map + add_depth
        mask_active = (hist_weight > 0) & (depth_map_new > 0)

        image[mask_active] = (np.divide(image * depth_map + hist_map * add_depth, depth_map_new,
                                        where=mask_active))[mask_active]
        depth_map = depth_map_new

        if(ilvl < maxlvl):
            image = rescale(image, 2, mode='constant', order=interp_order, multichannel=False, anti_aliasing=anti_aliasing)
            depth_map = rescale(depth_map, 2, mode='constant', order=interp_order, multichannel=False, anti_aliasing=anti_aliasing)

    crop_range = ((box_proj.T - edge[:, 0]) / (edge[:, 1] - edge[:, 0])).T
    if(subpx_crop):
        image = crop_float(image, crop_range, output_shape=shape)
    else:
        image = crop(image, crop_range)
        if(shape is not None):
            image = resize(image, shape)

    timer.record()
    return image.T


def draw_gasmap(cell, box=None, proj=[0, 1], shape=500, extent=None, mode='rho', unit=None, minlvl=None, maxlvl=None, subpx_crop=True, anti_aliasing=False, interp_order=0, **kwargs):
    if(box is None and isinstance(cell, uri.RamsesSnapshot.Cell)):
        box = cell.snap.box

    image = gasmap(cell, box, proj, mode=mode, unit=unit, shape=shape, minlvl=minlvl, maxlvl=maxlvl, subpx_crop=subpx_crop, anti_aliasing=anti_aliasing, interp_order=interp_order)

    box_proj = get_box_proj(box, proj)
    if extent is None:
        extent = np.concatenate(box_proj)

    draw_image(image, extent=extent, **kwargs)

def tracermap(tracer_part, box=None, proj=[0, 1], shape=500, mode='rho', unit=None, minlvl=None, maxlvl=None, subpx_crop=True, anti_aliasing=False):
    if(box is None and isinstance(tracer_part, uri.RamsesSnapshot.Particle)):
        box = tracer_part.snap.box

    lvl = tracer_part['level']
    box_proj = get_box_proj(box, proj)

    if (np.isscalar(shape)):
        shape = np.repeat(shape, 2)

    if(shape is not None):
        shape = np.array(shape)

    known_lvls = np.unique(lvl)
    minlvl, maxlvl, basebin, edge = set_bins(known_lvls, minlvl, maxlvl, box_proj, shape)

    known_lvls = np.arange(minlvl, np.max(known_lvls)+1)

    if(verbose>=1):
        print('MinLvl = %d, MaxLvl = %d, Initial Image Size: ' % (minlvl, maxlvl), basebin * 2**(maxlvl-minlvl))
    timer.start('Drawing tracer map... ', 1)

    if(shape is None):
        shape = basebin * 2 ** (maxlvl - minlvl)

    image = np.zeros(basebin)

    depth = np.diff(box[los(proj)])

    for ilvl in known_lvls:
        mask = lvl==ilvl
        cell_lvl = tracer_part[mask]

        binlvl = np.min([ilvl, maxlvl])
        binsize = basebin * 2**(binlvl-minlvl)

        xm = get_vector(cell_lvl)
        qm = cell_lvl['m', unit] / 0.5**(binlvl*2)
        if(mode == 'crho'):
            qm *= depth

        # convert coordinates to map
        hist_map = np.histogram2d(xm[:, proj[0]], xm[:, proj[1]], bins=binsize, range=edge, weights=qm)[0]

        ibin = ilvl - binlvl
        image += hist_map * 0.5 ** ibin

        if(ilvl < maxlvl):
            image = rescale(image, 2, mode='constant', order=0, multichannel=False, anti_aliasing=anti_aliasing)
    image /= depth

    crop_range = ((box_proj.T - edge[:, 0]) / (edge[:, 1] - edge[:, 0])).T
    if(subpx_crop):
        image = crop_float(image, crop_range, output_shape=shape)
    else:
        image = crop(image, crop_range)
        if(shape is not None):
            image = resize(image, shape)

    timer.record()
    return image.T

def draw_tracermap(tracer_part, box=None, proj=[0, 1], shape=500, extent=None, mode='rho', unit=None, minlvl=None, maxlvl=None, subpx_crop=True, anti_aliasing=False, **kwargs):
    if(box is None and isinstance(tracer_part, uri.RamsesSnapshot.Particle)):
        box = tracer_part.snap.box

    image = tracermap(tracer_part, box, proj, mode=mode, unit=unit, shape=shape, minlvl=minlvl, maxlvl=maxlvl, subpx_crop=subpx_crop, anti_aliasing=anti_aliasing)

    box_proj = get_box_proj(box, proj)
    if extent is None:
        extent = np.concatenate(box_proj)

    draw_image(image, extent=extent, **kwargs)

def partmap(part, box=None, proj=[0, 1], shape=1000, weights=None, unit=None, method='hist', x=None, smooth=16, crho=False):
    if(box is None and isinstance(part, uri.RamsesSnapshot.Particle)):
        box = part.snap.box

    # Compute the column density map along the LOS
    if(x is None):
        x = get_vector(part)

    box_proj = get_box_proj(box, proj)

    dims = np.arange(3)
    los = dims[np.isin(dims, proj, invert=True, assume_unique=True)][0]
    depth = np.diff(box[los])

    if (np.isscalar(shape)):
        shape = np.repeat(shape, 2)
    shape = np.array(shape)

    if(weights is None):
        weights = part['m']

    px_area = np.multiply(*((box_proj[:, 1] - box_proj[:, 0])/shape))

    timer.start('Computing particle map of %d particles... ' % part.size, 1)

    if(method == 'hist'):
        image = np.histogram2d(x[:, proj[0]], x[:, proj[1]], bins=shape, range=box_proj, weights=weights)[0]
        image /= px_area
    elif(method == 'kde'):
        image = dr.kde_img(x[:, proj[0]], x[:, proj[1]], reso=shape, lims=box_proj, weights=weights, tree=True)
    elif(method == 'dtfe'):
        image = dr.dtfe_img(x[:, proj[0]], x[:, proj[1]], reso=shape, lims=box_proj, weights=weights, smooth=smooth)
    else:
        raise ValueError('Unknown estimator.')

    if(not crho):
        image /= depth

    if(unit is not None):
        image /= part.snap.unit[unit]

    timer.record()

    return image.T


def draw_partmap(part, box=None, proj=[0, 1], shape=500, extent=None, weights=None, unit=None, method='hist', smooth=16, crho=False, **kwargs):
    if(box is None and isinstance(part, uri.RamsesSnapshot.Particle)):
        box = part.snap.box

    image = partmap(part, box, proj, shape, weights, unit, method, smooth=smooth, crho=crho)

    box_proj = get_box_proj(box, proj)

    if extent is None:
        extent = np.concatenate(box_proj)

    draw_image(image, extent=extent, **kwargs)


def rgb_image(image,  vmin=None, vmax=None, qscale=3., normmode='log', nanzero=False, imfilter=None, cmap=dr.ccm.laguna):
    if(not len(image.shape)>2):
        image = norm(image, vmin, vmax, qscale=qscale, mode=normmode, nanzero=nanzero)

    if(imfilter is not None):
        image = imfilter(image)

    return cmap(image)


def draw_image(image, extent=None, vmin=None, vmax=None, qscale=3., normmode='log', nanzero=False, imfilter=None, cmap=dr.ccm.laguna, **kwargs):
    if(not len(image.shape)>2):
        image = norm(image, vmin, vmax, qscale=qscale, mode=normmode, nanzero=nanzero)

    if(imfilter is not None):
        image = imfilter(image)

    ims = plt.imshow(image, extent=extent, vmin=0, vmax=1, origin='lower', zorder=1, cmap=cmap, **kwargs)
    return ims


def save_image(image, fname, cmap=dr.ccm.laguna, vmin=None, vmax=None, qscale=3., normmode='log', nanzero=False, make_dir=False):
    fname = os.path.expanduser(fname)
    if(make_dir):
        os.makedirs(os.path.dirname(fname), exist_ok=True)

    if(len(image.shape)>2):
        plt.imsave(fname, image, origin='lower')
    else:
        image_norm = cmap(norm(image, vmin, vmax, qscale, mode=normmode, nanzero=nanzero))

        plt.imsave(fname, image_norm, origin='lower')

def save_figure(fname, make_dir=True, **kwargs):
    fname = os.path.expanduser(fname)
    if(make_dir):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, **kwargs)

def draw_contour(image, extent, vmin=None, vmax=None, qscale=None, normmode='log', nlevel=3, **kwargs):
    image = norm(image, vmin, vmax, qscale=qscale, mode=normmode)

    xarr = bin_centers(extent[0], extent[1], image.shape[0])
    yarr = bin_centers(extent[2], extent[3], image.shape[1])

    return plt.contour(xarr, yarr, image, **kwargs)


def draw_points(points, box=None, proj=[0, 1], color=None, label=None, fontsize=None, fontcolor=None, **kwargs):
    x = get_vector(points)
    if(box is None and isinstance(points, uri.RamsesSnapshot.Particle)):
        box = points.snap.box
    else:
        box = default_box
    mask = uri.box_mask(x, box)
    x = x[mask]

    plt.scatter(x[:, proj[0]], x[:, proj[1]], color=color, zorder=50, **kwargs)

    if(label is not None):
        label = np.array(label)[mask]
        if(not isinstance(label, Iterable)):
            label = repeat(label)
        if (fontsize is not None and not isinstance(fontsize, Iterable)):
            fontsize = repeat(fontsize)
        ax = plt.gca()
        if(fontcolor is None):
            fontcolor = color
        if (isinstance(fontcolor, str) or not isinstance(fontcolor, Iterable)):
            fontcolor = repeat(fontcolor)

        for pos, lb, fs, fc in zip(x, label, fontsize, fontcolor):
            ax.annotate(lb, (pos[proj[0]], pos[proj[1]]), xytext=(5, 5), textcoords='offset points', color=fc, fontsize=fs, zorder=100)

def draw_smbhs(smbh, box=None, proj=[0, 1], s=30, cmap=None, color='k', mass_range=None, zorder=100, labels=None, fontsize=10, fontcolor='lightyellow', **kwargs):
    if(box is None and isinstance(smbh, uri.RamsesSnapshot.Particle)):
        box = smbh.snap.box
        mass = smbh['m', 'Msol']
        if(mass_range is None):
            m_max = np.max(mass)
            m_min = np.min(mass)
        else:
            m_min = 10.**mass_range[0]
            m_max = 10.**mass_range[1]

        mass_scale = norm(mass, m_min, m_max)

        ss = (mass_scale)**2 * s + 1
    else:
        ss = np.repeat(10, smbh.size)
    box_proj = get_box_proj(box, proj)

    poss = uri.get_vector(smbh)
    mask = uri.box_mask(poss, box)
    smbh = smbh[mask]

    if(cmap is not None):
        colors = cmap(mass_scale)
        color = colors
    plt.scatter(poss[:, proj[0]], poss[:, proj[1]], s=ss, color=color, zorder=zorder, **kwargs)

    plt.xlim(box_proj[0])
    plt.ylim(box_proj[1])

    if(labels is not None):
        labels = np.array(labels)[mask]
        ax = plt.gca()
        for i, pos, label, s in zip(np.arange(smbh.size), poss, labels, ss):
            #ax.text(pos[proj[0]], pos[proj[1]], label, color='white', ha='center', va='top', fontsize=fontsize, zorder=zorder, transform=ax.transAxes)
            ax.annotate(label, (pos[proj[0]], pos[proj[1]]), xytext=(5, 5), textcoords='offset points', color=fontcolor, fontsize=fontsize, zorder=zorder)



def draw_halos(halos, box=None, ax=None, proj=[0, 1], mass_range=None, cmap=plt.cm.jet, colors=None, labels=None, size_key='rvir', shape='circle', fontsize=10, extents=None, **kwargs):
    proj_keys = np.array(['x', 'y', 'z'])[proj]

    if ax is None:
        ax = plt.gca()

    if(not isinstance(halos, Iterable)):
        halos = np.array([halos], dtype=halos.dtype)
    mask = uri.box_mask(get_vector(halos), box=box)

    if(labels is None):
        labels = np.full(halos.size, None)
    if(not isinstance(extents, Iterable)):
        extents = np.full(halos.size, extents)
    else:
        extents = extents

    if(colors is None):
        colors = repeat(None)

    halos = np.array(halos)[mask]
    labels = np.array(labels)[mask]
    extents = np.array(extents)[mask]

    timer.start('Drawing %d halos...' % halos.size, 1)

    if mass_range is None:
        mass_range = np.log10(np.array([np.min(halos['mvir']), np.max(halos['mvir'])]))

    for halo, label, extent, color in zip(halos, labels, extents, colors):
        if(color is not None):
            color_cmp = color
        else:
            color_cmp = cmap((np.log10(halo['mvir']) - mass_range[0]) / (mass_range[1] - mass_range[0]))
        x, y, r = halo[proj_keys[0]], halo[proj_keys[1]], halo[size_key]
        if(extent is not None):
            r = extent/2
        if(shape == 'circle'):
            ax.add_artist(plt.Circle([x, y], radius=r, linewidth=0.5, edgecolor=color_cmp, facecolor='none', zorder=10, **kwargs))
        elif(shape == 'pentagon'):
            ax.add_artist(RegularPolygon([x, y], 5, radius=r, linewidth=0.5, edgecolor=color_cmp, facecolor='none', zorder=10, **kwargs))
        elif(shape == 'square'):
            ax.add_artist(Rectangle([x-r, y-r], r*2, r*2, linewidth=0.5, edgecolor=color_cmp, facecolor='none', zorder=10, **kwargs))
        if(label is not None):
            ax.text(x, y-r*1.1, label, color=color_cmp, ha='center', va='top', fontsize=fontsize)

    timer.record()

    if(box is not None):
        ax.set_xlim(box[proj[0]])
        ax.set_ylim(box[proj[1]])


def draw_grid(cell, box=None, ax=None, proj=[0, 1], minlvl=None, maxlvl=None, color='white', cmap=None, linewidth=0.5, draw_threshold=0, **kwargs):
    if ax is None:
        ax = plt.gca()
    if(box is None and isinstance(cell, uri.RamsesSnapshot.Cell)):
        box = cell.snap.box

    lvl = cell['level']

    box_proj = get_box_proj(box, proj)

    known_lvls = np.unique(lvl)
    minlvl, maxlvl, basebin, edge = set_bins(known_lvls, minlvl, maxlvl, box_proj, None)

    known_lvls = np.arange(minlvl, maxlvl+1)

    if(verbose>=1):
        print('MinLvl = %d, MaxLvl = %d, Initial Image Size: ' % (minlvl, maxlvl), basebin * 2**(maxlvl-minlvl))
    timer.start('Drawing grids... ', 1)

    for ilvl in known_lvls:
        mask = lvl==ilvl
        cell_lvl = cell[mask]

        xm = get_vector(cell_lvl)

        binlvl = np.min([ilvl, maxlvl])
        binsize = basebin * 2**(binlvl-minlvl)

        # convert coordinates to map
        hist_map = np.histogram2d(xm[:, proj[0]], xm[:, proj[1]], bins=binsize, range=edge)[0].T / 2**ilvl

        xarr = bin_centers(edge[0, 0], edge[0, 1], binsize[0])
        yarr = bin_centers(edge[1, 0], edge[1, 1], binsize[1])

        xm, ym = np.meshgrid(xarr, yarr)
        mesh = np.stack([xm, ym], axis=-1)
        size = 0.5**ilvl
        coords = mesh[hist_map > draw_threshold] - size/2
        progress = (ilvl-minlvl)/(maxlvl-minlvl)
        alpha = (1.-progress)/2+0.5
        if(cmap is not None):
            color=cmap(progress)

        for xy in coords:
            ax.add_patch(Rectangle(xy, size, size, edgecolor=color, facecolor='None', linewidth=linewidth, alpha=alpha, zorder=100, **kwargs))

    if(box is not None):
        ax.set_xlim(box_proj[0])
        ax.set_ylim(box_proj[1])


def draw_vector(pos, vec, box=None, ax=None, proj=[0, 1], length=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if(length is not None):
        vec = vec / utool.rss(vec) * length

    origin = pos[proj]
    direc = vec[proj]

    #ax.arrow(*origin, *direc, **kwargs)
    ax.annotate(s='', xy=tuple(origin+direc), xytext=tuple(origin), arrowprops=dict(arrowstyle='->'), **kwargs)

    if(box is not None):
        ax.set_xlim(box[proj[0]])
        ax.set_ylim(box[proj[1]])


def draw_mergertree(tree, root):

    brches = np.unique(tree['brch_id'])

    brch_ordering = np.empty(brches.size, dtype='i4')
    brch_dir = 1

    def search_prog(halo, idx):
        brch_ordering[idx] = halo['brch_id']
        progs = tree[tree['desc_id'] == halo['id']]
        for prog in progs:
            search_prog(prog)

    search_prog(root, 0)


def draw_mergertree_space(tree, box=None, proj=[0, 1], mass_range=None, alpha=0.3):
    if box is None:
        box = default_box

    brches = np.unique(tree['brch_id'])
    main_brch_id = tree[tree['scale'] == np.max(tree['scale'])][0]['brch_id']
    main_brch = tree[tree['brch_id'] == main_brch_id]
    main_brch = np.sort(main_brch, order='scale')

    a_start, a_end = 0.2, 1.0

    start = np.median(uri.get_vector(tree[tree['scale'] == a_start]), axis=0)
    end = uri.get_vector(main_brch[-1])

    center = lambda a: start+(end-start)*(a-a_start)/(a_end-a_start)-end

    if mass_range is None:
        mass_range = np.log10(np.array([np.min(tree['mvir']), np.max(tree['mvir'])]))

    for brch_id in brches:
        if(brch_id == main_brch_id):
            continue
        halos = tree[tree['brch_id']==brch_id]
        halos = np.sort(halos, order='scale')
        desc_id = halos[-1]['desc_id']
        desc_mask = tree['id'] == desc_id

        maxmass = np.log10(np.max(halos['mvir']))
        halos = halos[np.log10(halos['mvir'])>=maxmass-1]

        if(maxmass<mass_range[0]):
            continue

        if(np.sum(desc_mask)>0):
            desc = tree[desc_mask]
            halos = np.concatenate([halos, desc])

        pos = uri.get_vector(halos)
        for i in np.arange(halos.size):
            pos[i] = pos[i]-center(halos[i]['scale'])
        dr.colorline(gaussian_filter1d(pos[:, proj[0]], 1),
                     gaussian_filter1d(pos[:, proj[1]], 1),
                     np.log10(halos['mvir']),
                     cmap=plt.cm.rainbow, linewidth=((maxmass-10)/2)**1.5,
                     norm=plt.Normalize(*mass_range), alpha=alpha, zorder=10)

    plt.xlim(box[proj[0]])
    plt.ylim(box[proj[1]])


def combine_image(rgbs, mode='screen', weights=None):
    rgbs = np.array(rgbs)
    if(mode == 'average'):
        image = np.average(rgbs, axis=0, weights=weights)
    elif(mode == 'multiply'):
        image = np.product(rgbs, axis=0)
    elif(mode == 'max'):
        image = np.max(rgbs, axis=0)
    elif(mode == 'screen'):
        image = 1.-np.prod(1.-rgbs, axis=0)
    return image


def composite_image(images, cmaps, weights=None, vmins=None, vmaxs=None, qscales=3., mode='average', normmodes=None):
    rgbs = []

    nimg = len(images)

    if(vmins is None):
        vmins = np.full(nimg, None)
    if(vmaxs is None):
        vmaxs = np.full(nimg, None)
    if(isinstance(qscales, float)):
        qscales = np.full(nimg, qscales)
    if(normmodes is None):
        normmodes = np.full(nimg, 'log')

    if(verbose>=2):
        print('vmins:', vmins)
        print('vmaxs:', vmaxs)

    for image, cmap, vmin, vmax, qscale, normmode in zip(images, cmaps, vmins, vmaxs, qscales, normmodes):
        rgbs.append(cmap(norm(image, vmin, vmax, qscale=qscale, mode=normmode)))
    rgbs = np.array(rgbs)

    image = combine_image(rgbs, mode, weights)
    return image


def norm(v, vmin=None, vmax=None, qscale=3., mode='log', nanzero=False):
    v = v.copy()
    if (vmax is None):
        vmax = np.nanmax(v)
    if (qscale is None):
        if (vmin is not None):
            qscale = np.log10(vmax - vmin)
        else:
            qscale = np.log10(vmax) - np.log10(np.nanmin(v[v > 0]))
    if (vmin is None):
        vmin = 10. ** (np.log10(vmax) - qscale)

    if(mode == 'log'):
        v[v<vmin] = vmin
        v = np.log10(v/vmin) / np.log10(vmax/vmin)
    elif(mode == 'linear'):
        v = (v - vmin) / (vmax - vmin)

    elif(mode == 'asinh'):
        asinh = lambda x: np.arcsinh(10*x)/3
        v = asinh((v - vmin) / (vmax - vmin))

    elif(mode == 'sinh'):
        sinh = lambda x: np.sinh(3*x)/10
        v = sinh((v - vmin) / (vmax - vmin))

    elif(mode == 'sqrt'):
        sqrt = lambda x: np.sqrt(x)
        v = sqrt((v - vmin) / (vmax - vmin))

    elif(mode == 'pow'):
        a = 1000
        pow = lambda x: (a**x-1)/a
        v = pow((v - vmin) / (vmax - vmin))


    if(verbose>=2):
        print('vmin: %f' % vmin)
        print('vmax: %f' % vmax)
        print('qscale: %f' % qscale)

    if(not nanzero):
        v[np.isnan(v)] = 0
    return v


def draw_partmap_polar(part, pos=[0.5, 0.5, 0.5], radius=0.5, qscale=3):
    coo = utool.get_polar_coord(part['pos'], pos)
    coo = coo[coo[:, 0]<radius]
    x = coo[:, 1]/np.pi*np.cos(coo[:, 2])
    y = coo[:, 1]/np.pi*np.sin(coo[:, 2])

    #image = np.histogram2d(x[:, proj[0]], x[:, proj[1]], bins=shape, range=box_proj, weights=weights)[0]
    dr.hist_imshow(x, y, lims=[[-1, 1], [-1, 1]], reso=1000, weights=None)

def set_ticks_unit(snap, proj=[0, 1], unit='kpc', nticks=4, centered=True):
    box_proj = get_box_proj(snap.box, proj)
    xr = np.array(box_proj[0])
    yr = np.array(box_proj[1])

    if(centered):
        xc = np.mean(xr)
        yc = np.mean(yr)
    else:
        xc = 0.
        yc = 0.

    lunit = snap.unit[unit]

    xticks = get_tickvalues((xr-xc)/lunit, nticks)
    yticks = get_tickvalues((yr-yc)/lunit, nticks)

    plt.xticks(xticks*snap.unit[unit]+xc, labels=xticks)
    plt.yticks(yticks*snap.unit[unit]+yc, labels=yticks)

    chars = ['X', 'Y', 'Z']
    xchar = chars[proj[0]]
    ychar = chars[proj[1]]

    plt.xlabel('%s (%s)' % (xchar, unit))
    plt.ylabel('%s (%s)' % (ychar, unit))


def get_tickvalues(range, nticks=4):
    # selects "appropriate" tick value intervals for a given range of numbers and number of ticks
    diff = np.diff(range)
    order = np.log10(diff/nticks)
    order_int = int(np.floor(order))
    res = order - order_int
    ticksize = 10**order_int
    if(0.3 <= res < 0.4):
        ticksize *= 2
    elif(0.4 <= res < 0.7):
        ticksize *= 2.5
        order_int -= 1
    elif(0.7 <= res):
        ticksize *= 5

    ticks = (np.arange(range[0]//ticksize, range[1]//ticksize, 1) + 1) * ticksize
    ticks = np.round(ticks, -order_int)
    if(order_int>=0):
        ticks = ticks.astype(int)
    return ticks

def viewer(snap, hf='GalaxyMaker', rank=1, radius=10, mode='star',
           savefile=None, part_method='hist', align=True, age_cut=None, center=None):
    cell = None

    if(hf == 'GalaxyMaker'):
        gal = uhmi.HaloMaker.load(snap, path_in_repo='galaxy', galaxy=True)
        gal = np.sort(gal, order='m')
    else:
        gal = None

    if (gal is None):
        snap.set_box(center, radius * snap.unit['kpc'] * 2)
    else:
        snap.set_box_halo(gal[-rank], radius=radius * snap.unit['kpc'], use_halo_radius=False)
    snap.get_part()
    if (mode == 'gas' or mode == 'dust' or mode == 'metal' or mode == 'temp'):
        snap.get_cell()
        # cell = snap.cell
        if (gal is not None and align):
            cell = uri.align_axis_cell(snap.cell, gal[-rank])
        else:
            cell = snap.cell
    if (gal is not None and align):
        part = uri.align_axis(snap.part, gal[-rank])
    else:
        part = snap.part

    smbh = part['smbh']

    plt.figure(figsize=(20, 10))

    plt.subplot(121)
    proj = [0, 1]
    if (mode == 'star'):
        star = part['star']
        if (age_cut is not None):
            star = star[star['age', 'Gyr'] < age_cut]
        draw_partmap(star, proj=proj, shape=1000, qscale=4, vmax=3E5, crho=True, method=part_method,
                             unit='Msol/pc2')
    if (mode == 'dm'):
        dm = part['dm']
        draw_partmap(dm, proj=proj, shape=1000, qscale=4, vmax=1E4, crho=True, method=part_method,
                             unit='Msol/pc2')
    elif (mode == 'gas'):
        draw_gasmap(cell, proj=proj, shape=1000, qscale=4, vmax=3E3, mode='crho', cmap=ccm.hesperia,
                            interp_order=1, unit='Msol/pc2')
    elif (mode == 'temp'):
        draw_gasmap(cell, proj=proj, shape=1000, qscale=4, mode='T', cmap=ccm.hesperia, vmax=1E8, unit='K')
    elif (mode == 'dust'):
        draw_gasmap(cell, proj=proj, shape=1000, qscale=2, vmax=3E-2, mode='dust', cmap=ccm.lacerta,
                            interp_order=1)
    elif (mode == 'metal'):
        draw_gasmap(cell, proj=proj, shape=1000, qscale=3, vmax=1E-1, mode='metal', cmap=ccm.lacerta,
                            interp_order=1)
    draw_smbhs(smbh, proj=proj, labels=['%.3f' % m for m in np.log10(smbh['m', 'Msol'])], color='gray',
                       fontsize=10, mass_range=[4, 8])
    set_ticks_unit(snap, proj, 'kpc')
    if (gal is not None):
        dr.axlabel('M$_*$ = %.3e M$_{sol}$' % gal[i]['m'], 'left top', color='white', fontsize=20)
    dr.axlabel('z = %.3f' % snap.z, 'right top', color='white', fontsize=20)

    plt.subplot(122)
    proj = [0, 2]
    if (mode == 'star'):
        star = part['star']
        if (age_cut is not None):
            star = star[star['age', 'Gyr'] < age_cut]
        draw_partmap(star, proj=proj, shape=1000, qscale=4, vmax=3E5, crho=True, method=part_method,
                             unit='Msol/pc2')
    if (mode == 'dm'):
        dm = part['dm']
        draw_partmap(dm, proj=proj, shape=1000, qscale=4, vmax=1E4, crho=True, method=part_method,
                             unit='Msol/pc2')
    elif (mode == 'gas'):
        draw_gasmap(cell, proj=proj, shape=1000, qscale=3, vmax=3E3, mode='crho', cmap=ccm.hesperia,
                            interp_order=1, unit='Msol/pc2')
    elif (mode == 'temp'):
        draw_gasmap(cell, proj=proj, shape=1000, qscale=4, mode='T', cmap=ccm.hesperia, vmax=1E8, unit='K')
    elif (mode == 'dust'):
        draw_gasmap(cell, proj=proj, shape=1000, qscale=2, vmax=3E-2, mode='dust', cmap=ccm.lacerta,
                            interp_order=1)
    elif (mode == 'metal'):
        draw_gasmap(cell, proj=proj, shape=1000, qscale=3, vmax=1E-1, mode='metal', cmap=ccm.lacerta,
                            interp_order=1)
    draw_smbhs(smbh, proj=proj, labels=['%.3f' % m for m in np.log10(smbh['m', 'Msol'])], color='gray',
                       fontsize=10, mass_range=[4, 8])
    set_ticks_unit(snap, proj, 'kpc')

    if (savefile is not None):
        save_figure(savefile)