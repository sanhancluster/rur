import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import rescale, resize, warp, EuclideanTransform, AffineTransform
from rur import drawer as dr, utool
from rur.drawer import ccm
from rur import uri, uhmi
from rur.sci import photometry as phot
from rur.sci.kinematics import align_axis, align_axis_cell
from rur.utool import Timer, get_vector, bin_centers, rss, los
from matplotlib.patches import RegularPolygon, Rectangle
from scipy.ndimage.filters import gaussian_filter1d
from collections.abc import Iterable
from itertools import repeat
from PIL import Image
from warnings import warn
from rur.sci import geometry as geo
import os
from astropy.visualization import make_lupton_rgb
from rur.config import default_path_in_repo, timer
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import cycle

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

    if(timer.verbose>=2):
        print('box', list(box_proj))
        print('edge', list(edge))
        print('basebin', basebin)
        print('Known levels: ', known_lvls)

    return minlvl, maxlvl, basebin, edge


def lvlmap(cell, box=None, proj=[0, 1], shape=500, minlvl=None, maxlvl=None, subpx_crop=True):
    if(box is None and isinstance(cell, uri.Cell)):
        box = cell.snap.box

    lvl = cell['level']

    box_proj = get_box_proj(box, proj)

    if (np.isscalar(shape)):
        shape = np.repeat(shape, 2)

    known_lvls = np.unique(lvl)
    minlvl, maxlvl, basebin, edge = set_bins(known_lvls, minlvl, maxlvl, box_proj, shape)

    known_lvls = np.arange(minlvl, maxlvl+1)

    if(timer.verbose>=1):
        print('MinLvl = %d, MaxLvl = %d, Initial Image Size: ' % (minlvl, maxlvl), (basebin * 2.**(maxlvl-minlvl)).astype(int))
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
            image = rescale(image, 2, order=0, multichannel=False)

    crop_range = ((box_proj.T - edge[:, 0]) / (edge[:, 1] - edge[:, 0])).T
    if(subpx_crop):
        image = crop_float(image, crop_range, output_shape=shape)
    else:
        image = crop(image, crop_range)
        if(shape is not None):
            image = resize(image, shape, mode='constant')

    timer.record()
    if(timer.verbose>=1):
        print("Cropped Image Size: ", image.shape)

    return image.T

def draw_lvlmap(cell, box=None, proj=[0, 1], shape=None, minlvl=None, maxlvl=None, **kwargs):
    image = lvlmap(cell, box, proj, shape=shape, minlvl=minlvl, maxlvl=maxlvl)

    if box is None:
        box = default_box
    box = np.array(box)

    box_proj = np.array(box)[proj]

    draw_image(image, np.concatenate(box_proj), vmax=maxlvl, vmin=minlvl, normmode='linear', **kwargs)

def set_weights(mode, cell, unit, depth, weights=None, quantity=None):
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
    elif (mode == 'refmask'):
        # cumulative refinement paramster along LOS
        weights = np.full(cell.size, 1)
    elif (mode == 'rho'):
        # average density along LOS
        weights = np.full(cell.size, 1)
    elif (mode == 'crho'):
        # column density along LOS
        weights = np.full(cell.size, 1)
        quantity = cell['rho', unit] * depth
    elif (mode != 'custom'):
        weights = np.full(cell.size, 1)
    if(quantity is None and mode != 'custom'):
        quantity = cell[mode, unit]

    return quantity, weights

def gasmap(cell, box=None, proj=[0, 1], shape=500, mode='rho', unit=None, minlvl=None, maxlvl=None, subpx_crop=True,
           interp_order=0, weights=None, quantity=None, method='hist'):
    if(box is None and hasattr(cell, 'snap')):
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

    if(timer.verbose>=1):
        print('MinLvl = %d, MaxLvl = %d, Initial Image Size: ' % (minlvl, maxlvl), (basebin * 2.**(maxlvl-minlvl)).astype(int))
    timer.start('Drawing gas map... ', 1)

    if(shape is None):
        shape = basebin * 2 ** (maxlvl - minlvl)

    image = np.zeros(basebin)
    depth_map = np.zeros(basebin)
    depth = np.diff(box[los(proj)])

    qc, wc = set_weights(mode, cell, unit, depth, weights, quantity)

    for ilvl in known_lvls:
        mask = lvl==ilvl
        cell_lvl = cell[mask]
        qm = qc[mask]
        wm = wc[mask]

        xm = get_vector(cell_lvl)

        binlvl = np.min([ilvl, maxlvl])
        binsize = basebin * 2**(binlvl-minlvl)

        # convert coordinates to map
        if(method == 'hist'):
            hist_weight = np.histogram2d(xm[:, proj[0]], xm[:, proj[1]], bins=binsize, range=edge, weights=wm)[0]
            hist_map = np.histogram2d(xm[:, proj[0]], xm[:, proj[1]],
                                      bins=binsize, range=edge, weights=qm*wm)[0]
        elif(method == 'cic'):
            # apply cic when measuring density map, only useful when the line of view is not aligned to x, y, z axis.
            hist_weight = dr.cic_img(xm[:, proj[0]], xm[:, proj[1]], reso=binsize, lims=edge, weights=wm)
            hist_map = dr.cic_img(xm[:, proj[0]], xm[:, proj[1]],
                                      reso=binsize, lims=edge, weights=qm*wm)

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
            image = rescale(image, 2, mode='constant', order=interp_order, multichannel=False)
            depth_map = rescale(depth_map, 2, mode='constant', order=interp_order, multichannel=False)

    crop_range = ((box_proj.T - edge[:, 0]) / (edge[:, 1] - edge[:, 0])).T
    if(subpx_crop):
        image = crop_float(image, crop_range, output_shape=shape)
    else:
        image = crop(image, crop_range)
        if(shape is not None):
            image = resize(image, shape)

    timer.record()
    return image.T

def velmap(data, box=None, proj=[0, 1], shape=500, unit=None, minlvl=None, maxlvl=None, subpx_crop=True, interp_order=0):
    if(box is None):
        box = data.snap.box

    if(isinstance(data, uri.Cell)):
        cell = data

        lvl = cell['level']

        box_proj = get_box_proj(box, proj)

        if (np.isscalar(shape)):
            shape = np.repeat(shape, 2)

        if(shape is not None):
            shape = np.array(shape)

        known_lvls = np.unique(lvl)
        minlvl, maxlvl, basebin, edge = set_bins(known_lvls, minlvl, maxlvl, box_proj, shape)

        known_lvls = np.arange(minlvl, np.max(known_lvls)+1)

        if(timer.verbose>=1):
            print('MinLvl = %d, MaxLvl = %d, Initial Image Size: ' % (minlvl, maxlvl), basebin * 2**(maxlvl-minlvl))
        timer.start('Drawing gas velocity map... ', 1)

        ndim = cell['vel'].shape[-1]
        if(shape is None):
            shape = basebin * 2 ** (maxlvl - minlvl)

        image = np.zeros((ndim,) + tuple(basebin))
        depth_map = np.zeros(basebin)
        depth = np.diff(box[los(proj)])

        for ilvl in known_lvls:
            mask = lvl==ilvl
            cell_lvl = cell[mask]

            xm = get_vector(cell_lvl)
            wm = cell_lvl['rho']
            qm = cell_lvl['vel']

            binlvl = np.min([ilvl, maxlvl])
            binsize = basebin * 2**(binlvl-minlvl)

            # convert coordinates to map
            hist_weight = np.histogram2d(xm[:, proj[0]], xm[:, proj[1]], bins=binsize, range=edge, weights=wm)[0]
            hist_map = []
            for idim in range(ndim):
                hist_map.append(
                    np.histogram2d(xm[:, proj[0]], xm[:, proj[1]], bins=binsize, range=edge, weights=qm[:, idim]*wm)[0])

            # weighted average map of quantities
            hist_map = np.array(hist_map)
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

            for idim in range(ndim):
                image[idim, mask_active] = (np.divide(image[idim] * depth_map + hist_map[idim] * add_depth, depth_map_new,
                                            where=mask_active))[mask_active]
            depth_map = depth_map_new

            if(ilvl < maxlvl):
                image = np.moveaxis(image, 0, -1)
                image = rescale(image, 2, mode='constant', order=interp_order, multichannel=True)
                image = np.moveaxis(image, -1, 0)
                depth_map = rescale(depth_map, 2, mode='constant', order=interp_order, multichannel=False)

        crop_range = ((box_proj.T - edge[:, 0]) / (edge[:, 1] - edge[:, 0])).T

        image_out = []
        for idim in range(ndim):
            if(subpx_crop):
                image_out.append(crop_float(image[idim], crop_range, output_shape=shape))
            else:
                image_out.append(crop(image[idim], crop_range))
                if(shape is not None):
                    image_out[idim] = resize(image_out[idim], shape)
        image_out = np.moveaxis(image_out, 0, -1)
        image_out = np.swapaxes(image_out, 0, 1)
    timer.record()
    if(unit is not None):
        image_out /= data.snap.unit[unit]
    return image_out

def draw_gasmap(cell, box=None, proj=[0, 1], shape=500, extent=None, mode='rho', unit=None, minlvl=None, maxlvl=None,
                subpx_crop=True, interp_order=0, weights=None, quantity=None, method='hist', **kwargs):
    if(box is None and hasattr(cell, 'snap')):
        box = cell.snap.box

    image = gasmap(cell, box, proj, mode=mode, unit=unit, shape=shape, minlvl=minlvl, maxlvl=maxlvl,
                   subpx_crop=subpx_crop, interp_order=interp_order, weights=weights, quantity=quantity, method=method)

    box_proj = get_box_proj(box, proj)
    if extent is None:
        extent = np.concatenate(box_proj)

    return draw_image(image, extent=extent, **kwargs)

def tracermap(tracer_part, box=None, proj=[0, 1], shape=500, mode='rho', unit=None, minlvl=None, maxlvl=None, subpx_crop=True):
    if(box is None and hasattr(tracer_part, 'snap')):
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

    if(timer.verbose>=1):
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
            image = rescale(image, 2, mode='constant', order=0, multichannel=False)
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

def draw_tracermap(tracer_part, box=None, proj=[0, 1], shape=500, extent=None, mode='rho', unit=None, minlvl=None, maxlvl=None, subpx_crop=True, **kwargs):
    if(box is None and hasattr(tracer_part, 'snap')):
        box = tracer_part.snap.box

    image = tracermap(tracer_part, box, proj, mode=mode, unit=unit, shape=shape, minlvl=minlvl, maxlvl=maxlvl, subpx_crop=subpx_crop)

    box_proj = get_box_proj(box, proj)
    if extent is None:
        extent = np.concatenate(box_proj)

    return draw_image(image, extent=extent, **kwargs)

def partmap(part, box=None, proj=[0, 1], shape=1000, weights=None, unit=None, method='hist', x=None, smooth=16,
            crho=False, angles=None, **kwargs):
    if(box is None and isinstance(part, uri.Particle)):
        box = part.snap.box

    # Compute the column density map along the LOS
    if(x is None):
        x = get_vector(part)
    if(angles is not None):
        focus = np.mean(box, axis=-1)
        x = x - focus
        x = geo.euler_angle(x, angles) + focus

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

    if(part is not None):
        timer.start('Computing particle map of %d particles... ' % part.size, 1)

    if(method == 'hist'):
        image = np.histogram2d(x[:, proj[0]], x[:, proj[1]], bins=shape, range=box_proj, weights=weights, **kwargs)[0]
        image /= px_area
    elif (method == 'gaussian'):
        image = dr.gauss_img(x[:, proj[0]], x[:, proj[1]], reso=shape, lims=box_proj, weights=weights, **kwargs)
        image /= px_area
    elif(method == 'kde'):
        image = dr.kde_img(x[:, proj[0]], x[:, proj[1]], reso=shape, lims=box_proj, weights=weights, tree=True, **kwargs)
    elif(method == 'dtfe'):
        image = dr.dtfe_img(x[:, proj[0]], x[:, proj[1]], reso=shape, lims=box_proj, weights=weights, smooth=smooth)
    elif (method == 'cic'):
        image = dr.cic_img(x[:, proj[0]], x[:, proj[1]], reso=shape, lims=box_proj, weights=weights)
        image /= px_area
    else:
        raise ValueError('Unknown estimator.')

    if(not crho):
        image /= depth

    if(unit is not None):
        image /= part.snap.unit[unit]

    timer.record()

    return image.T


def draw_partmap(part, box=None, proj=[0, 1], shape=500, extent=None, weights=None, unit=None, method='hist',
                 smooth=16, crho=False, angles=None, kwargs_partmap={}, **kwargs):
    if(box is None and hasattr(part, 'snap')):
        box = part.snap.box

    image = partmap(part, box, proj, shape, weights, unit, method, smooth=smooth, crho=crho, angles=angles, **kwargs_partmap)

    box_proj = get_box_proj(box, proj)

    if extent is None:
        extent = np.concatenate(box_proj)

    return draw_image(image, extent=extent, **kwargs)


def rgb_image(image, vmin=None, vmax=None, qscale=3., normmode='log', nanzero=False, imfilter=None, cmap=dr.ccm.laguna):
    # returns rgb array of the target image using the given vmin / vmax
    if not len(image.shape)>2:
        image = norm(image, vmin, vmax, qscale=qscale, mode=normmode, nanzero=nanzero)

    if imfilter is not None:
        image = imfilter(image)

    return cmap(image)


def draw_image(image, extent=None, vmin=None, vmax=None, qscale=3., normmode='log', nanzero=False, imfilter=None,
               cmap=dr.ccm.laguna, colorbar=False, colorbar_kw={}, **kwargs):

    if imfilter is not None:
        image = imfilter(image)
    ax = kwargs.pop('ax', plt.gca())
    if not len(image.shape)>2:
        if qscale is not None and vmax is None:
            vmax = np.nanmax(image)
        #image = norm(image, vmin, vmax, qscale=qscale, mode=normmode, nanzero=nanzero)
        nm = get_norm(vmin, vmax, qscale=qscale, mode=normmode)
        sm = mpl.cm.ScalarMappable(norm=nm, cmap=cmap)
        ims = ax.imshow(image, extent=extent, norm=sm.norm, origin='lower', zorder=1, cmap=sm.cmap, **kwargs)
        if colorbar:
            plt.colorbar(mappable=sm, ax=ax, **colorbar_kw)
    else:
        ims = ax.imshow(image, extent=extent, origin='lower', zorder=1, **kwargs)

    return ims


def save_image(image, fname, cmap=dr.ccm.laguna, vmin=None, vmax=None, qscale=3., normmode='log',
               nanzero=False, make_dir=False, grayscale=False, img_mode='RGB', bit=8):
    fname = os.path.expanduser(fname)
    if(make_dir):
        os.makedirs(os.path.dirname(fname), exist_ok=True)

    image = norm(image, vmin, vmax, qscale, mode=normmode, nanzero=nanzero)
    if(not grayscale and len(image.shape)<3):
        image = cmap(image)
    if(bit == 16):
        im = Image.fromarray(np.uint16(image * 65535))
    elif(bit == 8):
        im = Image.fromarray(np.uint8(image * 255))
    else:
        raise ValueError("Unknown bit size")
    im.save(fname, format='png', compression=None)


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


def draw_points(points, box=None, proj=[0, 1], color=None, label=None, fontsize=None, fontcolor=None, s=None, **kwargs):
    x = get_vector(points)
    if(box is None and hasattr(points, 'snap')):
        box = points.snap.box
    else:
        box = default_box
    mask = uri.box_mask(x, box)
    x = x[mask]

    if(isinstance(s, Iterable)):
        s = np.array(s)[mask]

    plt.scatter(x[:, proj[0]], x[:, proj[1]], color=color, zorder=50, s=s, **kwargs)

    if(label is not None):
        label = np.array(label)
        if(not isinstance(label, Iterable)):
            label = repeat(label)
        else:
            label = np.array(label)[mask]

        if (fontsize is not None and not isinstance(fontsize, Iterable)):
            fontsize = repeat(fontsize)
        else:
            fontsize = np.array(fontsize)[mask]
        ax = plt.gca()
        if(fontcolor is None):
            fontcolor = color
        if (isinstance(fontcolor, str) or not isinstance(fontcolor, Iterable)):
            fontcolor = repeat(fontcolor)
        else:
            fontcolor = np.array(fontcolor)[mask]

        for pos, lb, fs, fc in zip(x, label, fontsize, fontcolor):
            ax.annotate(lb, (pos[proj[0]], pos[proj[1]]), xytext=(5, 5), textcoords='offset points', color=fc, fontsize=fs, zorder=100)

def draw_smbhs(smbh, box=None, proj=[0, 1], s=30, cmap=None, color='k', mass_range=None, zorder=100,
               labels=None, fontsize=10, fontcolor='lightyellow', **kwargs):
    if(box is None and isinstance(smbh, uri.Particle)):
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
        for i, pos, label, s in zip(np.arange(smbh.size), poss[mask], labels, ss[mask]):
            #ax.text(pos[proj[0]], pos[proj[1]], label, color='white', ha='center', va='top', fontsize=fontsize, zorder=zorder, transform=ax.transAxes)
            ax.annotate(label, (pos[proj[0]], pos[proj[1]]), xytext=(3, 3), textcoords='offset points',
                        color=fontcolor, fontsize=fontsize, zorder=zorder)



def draw_halos(halos, box=None, ax=None, proj=[0, 1], mass_range=None, cmap=plt.cm.jet, colors=None, labels=None,
               size_key='rvir', radius=1, shape='circle', fontsize=10, extents=None, **kwargs):
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
    if(not isinstance(radius, Iterable)):
        radius = np.full(halos.size, radius)

    if(colors is None or isinstance(colors, Iterable)):
        colors = repeat(colors)

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
        x, y = halo[proj_keys[0]], halo[proj_keys[1]]
        if(size_key is not None):
            r = halo[size_key]
        else:
            r = radius
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
    if(box is None and isinstance(cell, uri.Cell)):
        box = cell.snap.box

    lvl = cell['level']

    box_proj = get_box_proj(box, proj)

    known_lvls = np.unique(lvl)
    minlvl, maxlvl, basebin, edge = set_bins(known_lvls, minlvl, maxlvl, box_proj, None)

    known_lvls = np.arange(minlvl, maxlvl+1)

    if(timer.verbose>=1):
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

def blend(image1, image2, mode='average'):
    # based on https://en.wikipedia.org/wiki/Blend_modes
    if mode == 'average':
        return (image1 + image2) / 2
    if mode == 'multiply':
        return image1 * image2
    elif mode == 'overlay':
        return np.select([image1 < 0.5, True], [2*image1*image2, 1-2*(1-image1)*(1-image2)])
    elif mode == 'hard light':
        return blend(image2, image1, 'overlay')
    elif mode == 'soft light':
        return np.select([image2 < 0.5, True],
                         [2 * image1 * image2 + image1**2 * (1 - 2 * image2),
                          2 * image1 * (1 - image2) + np.sqrt(image1) * (2 * image2 - 1)])
    else:
        raise ValueError("Unknown blend mode: %s" % mode)

def combine_image(images_to_combine, mode='screen', weights=None):
    images_to_combine = np.array(images_to_combine)
    if mode == 'average':
        image = np.average(images_to_combine, axis=0, weights=weights)
    elif mode == 'sum':
            image = np.sum(images_to_combine, axis=0)
    elif mode == 'multiply':
        image = np.product(images_to_combine, axis=0)
    elif mode == 'max':
        image = np.max(images_to_combine, axis=0)
    elif mode == 'screen':
        image = 1.-np.prod(1.-images_to_combine, axis=0)
    elif mode == 'overlay':
        image1 = images_to_combine[0]
        image = np.select([image1 < 0.5, True], [2 * np.prod(images_to_combine), 1 - 2 * np.prod(1 - images_to_combine)])
    elif mode in ('soft light', 'hard light'):
        image = images_to_combine[0]
        for image2 in images_to_combine[1:]:
            image = blend(image, image2, 'overlay')
    else:
        raise ValueError("Unknown blend mode: %s" % mode)

    return image


def composite_image(images, cmaps, weights=None, vmins=None, vmaxs=None, qscales=3., mode='average', normmodes=None):
    images_to_combine = []

    nimg = len(images)

    if(vmins is None):
        vmins = np.full(nimg, None)
    if(vmaxs is None):
        vmaxs = np.full(nimg, None)
    if(isinstance(qscales, float)):
        qscales = np.full(nimg, qscales)
    if(normmodes is None):
        normmodes = np.full(nimg, 'log')

    if(timer.verbose>=2):
        print('vmins:', vmins)
        print('vmaxs:', vmaxs)

    for image, cmap, vmin, vmax, qscale, normmode in zip(images, cmaps, vmins, vmaxs, qscales, normmodes):
        images_to_combine.append(cmap(norm(image, vmin, vmax, qscale=qscale, mode=normmode)))
    images_to_combine = np.array(images_to_combine)

    image = combine_image(images_to_combine, mode, weights)
    return image


class ZeroLogNorm(mpl.colors.LogNorm):
    # LogNorm that returns 0 (instead of 'masked') for values <=0. when clip is True.
    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip
        if clip:
            value = np.array(value)
            value[value <= 0.] = self.vmin
        return super().__call__(value, clip)


def get_norm(vmin=None, vmax=None, qscale=3., mode='log', clip=True):
    if (qscale is None):
        if (vmin is not None):
            qscale = np.log10(vmax - vmin)
    if (vmin is None):
        vmin = 10. ** (np.log10(vmax) - qscale)

    if mode in ['linear', 'lin', 'norm']:
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=clip)
    elif mode in ['log', 'logscale']:
        return ZeroLogNorm(vmin=vmin, vmax=vmax, clip=clip)
    else:
        raise ValueError("Unknown normalization mode: ", mode)


def norm(v, vmin=None, vmax=None, qscale=3., mode='log', nanzero=False):
    # vmin overrides qscale.
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


    if(timer.verbose>=2):
        print('vmin: %f' % vmin)
        print('vmax: %f' % vmax)
        print('qscale: %f' % qscale)

    if(not nanzero):
        v[np.isnan(v)] = 0
    return v


def draw_partmap_polar(part, pos=None, radius=0.5, qscale=3):
    if pos is None:
        pos = [0.5, 0.5, 0.5]
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

def quick_target_box(snap:uri.RamsesSnapshot, center=None, target=None, catalog=None, source='GalaxyMaker',
                 rank=None, rank_order=None, id=None, id_name='id', radius=None, radius_unit='kpc', drag_part=True):
    # complex parameter handling comes here...
    if (radius is None):
        radius = 10.
    if (radius_unit in snap.unit):
        radius = radius * snap.unit[radius_unit]
    elif (target is not None and radius_unit in target.dtype.names):
        radius = radius * target[radius_unit]
    elif radius_unit is not None:
        warn("Unknown radius_unit, assuming as code unit...")

    if (center is not None):
        # if center is specified, use it
        pass
    elif (rank is None and id is None and target is None
          and not (np.array_equal(snap.box, default_box) or snap.box is None)):
        # if there is predefined box in snap, use it
        pass
    else:
        if(target is None):
            # if target is not specified, get one from catalog using rank / id
            if(catalog is None):
                # if catalog is not specified, get it using source
                if (source == 'GalaxyMaker'):
                    catalog = uhmi.HaloMaker.load(snap, path_in_repo=default_path_in_repo['GalaxyMaker'], galaxy=True, double_precision=True)
                elif(source == 'SINKPROPS'):
                    catalog = snap.read_sinkprop(drag_part=drag_part)
                elif (source == 'sink'):
                    snap.get_sink(all=True)
                    catalog = snap.sink.table
                else:
                    raise ValueError("Unknown source: %s" % source)

            if rank is None:
                rank = 1

            catalog = np.sort(catalog, order=rank_order)
            if(id is not None):
                # use id if specified
                target = catalog[catalog[id_name] == id]
                if(target.size == 0):
                    raise ValueError("No target found with the matching id")
                elif(target.size > 1):
                    # print("Multiple targets with same id are selected, using rank to select 1 target")
                    target = catalog[catalog[id_name] == id][-rank]
                else:
                    target = catalog[catalog[id_name] == id][0]
            else:
                # use rank instead, default value is 1
                target = catalog[-rank]
        center = uri.get_vector(target)
    box = uri.get_box(center, extent=radius * 2)

    return box, target


def viewer(snap:uri.RamsesSnapshot, box=None, center=None, target=None, catalog=None, source='GalaxyMaker',
           rank=None, rank_order='m', id=None, id_name='id', radius=None, radius_unit='kpc', mode=['star', 'gas'],
           show_smbh=True, savefile=None, part_method='cic', cell_method=None, align=False, age_cut=None, proj=[0, 1],
           smbh_minmass=1E4, interp_order=1, subplots_adjust_kw=None,
           smbh_labels=True, figsize=None, dpi=150, vmaxs=None, qscales=None, phot_filter='SDSS_u', shape=1000,
           drag_part=True, colorbar=False, colorbar_kw=None, colorbar_size=0.15,
           axis=False, ruler=True, ruler_size_in_radius_unit=None, props=['contam', 'sfr', 'fedd', 'mbh'], fontsize=8,
           nrows=1, ncols=-1, size_panel=3.5, title=None):
    """Simple galaxy viewer integrated with GalaxyMaker / SINKPROPS data.
    parameters are used in following priorities, inputs with lower priorities are ignored
    - box
    - center
    - target
    - catalog
    - source
    """

    cell = None
    part = None
    smbh = None

    subplots_adjust_kw_default = dict(
        wspace=0.05, hspace=0.05, bottom=0.05, top=0.95, left=0.05, right=0.95)

    if subplots_adjust_kw is None:
        subplots_adjust_kw = subplots_adjust_kw_default
    else:
        subplots_adjust_kw_default.update(subplots_adjust_kw)
        subplots_adjust_kw = subplots_adjust_kw_default

    colorbar_kw_default = dict(
        orientation='vertical')

    if colorbar_kw is None:
        colorbar_kw = colorbar_kw_default
    else:
        colorbar_kw_default.update(colorbar_kw)
        colorbar_kw = colorbar_kw_default


    proj = np.atleast_2d(proj)
    if cell_method is None:
        if align:
            cell_method='cic'
        else:
            cell_method='hist'

    if(isinstance(mode, Iterable)):
        npans = len(mode)
    else:
        npans = proj.shape[0]

    if proj.shape[0] == 1:
        proj = np.repeat(proj, npans, axis=0)

    if ncols == -1:
        ncols = int(np.ceil(npans//nrows))
    elif nrows == -1:
        nrows = int(np.ceil(npans//ncols))

    if not isinstance(mode, Iterable):
        mode = np.repeat(mode, npans)
    if not isinstance(show_smbh, Iterable):
        show_smbh = np.repeat(show_smbh, npans)
    if not isinstance(smbh_labels, Iterable):
        smbh_labels = np.repeat(smbh_labels, npans)

    vmax_dict = {
        'star':  3E5,
        'gas':   3E3,
        'dm':    1E4,
        'metal': 1E-1,
        'dust':  3E-2,
        'temp':  1E8,
        'phot':  1E19,
        'sdss':  1E17
    }

    qscale_dict = {
        'star':  5,
        'gas':   3,
        'dm':    4,
        'metal': 3,
        'dust':  2,
        'temp':  4,
        'phot':  5,
        'sdss':  None,
    }

    if (box is not None):
        # if box is specified, use it
        snap.box = box
    else:
        snap.box, target = quick_target_box(snap, center, target, catalog, source, rank, rank_order, id, id_name, radius, radius_unit, drag_part)

    if (np.any(np.isin(['star', 'dm', 'sdss', 'phot'], mode)) or True in show_smbh
            or np.any(np.isin(['sfr', 'contam'], mode))):
        snap.get_part()
        if target is not None and align:
            part = align_axis(snap.part, target)
        else:
            part = snap.part
        if True in show_smbh:
            smbh = part['smbh']
            if smbh is not None:
                smbh = smbh[smbh['m', 'Msol'] >= smbh_minmass]
    if np.any(np.isin(['gas', 'dust', 'metal', 'temp'], mode)):
        snap.get_cell()
        if target is not None and align:
            cell = align_axis_cell(snap.cell, target)
        else:
            cell = snap.cell

    if np.any(np.isin(['fedd', 'mbh'], props)):
        snap.get_sink()
        sink = snap.sink

    if figsize is None:
        figsize = [size_panel*ncols, size_panel*nrows]
        if colorbar:
            if colorbar_kw['orientation'] == 'horizontal':
                figsize[1] *= (1.05 + colorbar_size)
            if colorbar_kw['orientation'] == 'vertical':
                figsize[0] *= (1.05 + colorbar_size)
    fig, axes = plt.subplots(figsize=figsize, dpi=dpi, ncols=ncols, nrows=nrows, squeeze=False)
    if title is not None:
        plt.suptitle(title)

    for ipan in np.arange(ncols * nrows):
        irow, icol = ipan // ncols, ipan % ncols
        plt.sca(axes[irow, icol])
        plt.axis('off')

    for ipan in np.arange(npans):
        irow, icol = ipan // ncols, ipan % ncols
        plt.sca(axes[irow, icol])
        proj_now = proj[ipan]
        mode_now = mode[ipan]

        if(vmaxs is not None):
            vmax = vmaxs[ipan]
        else:
            vmax = vmax_dict[mode_now]

        if(qscales is not None):
            qscale = qscales[ipan]
        else:
            qscale = qscale_dict[mode_now]

        if (mode_now == 'star'):
            star = part['star']
            if (age_cut is not None):
                star = star[star['age', 'Gyr'] < age_cut]
            im = draw_partmap(star, proj=proj_now, shape=shape, qscale=qscale, vmax=vmax, crho=True, method=part_method,
                                 unit='Msol/pc2')
            colorbar_label = 'Stellar density\nM$_{\odot}$ pc$^{-2}$'
            mode_label = 'Stars'
        elif (mode_now == 'phot'):
            star = part['star']
            if (age_cut is not None):
                star = star[star['age', 'Gyr'] < age_cut]
            mags = phot.measure_magnitude(star, filter_name=phot_filter, total=False)
            lums = 10**(-mags/2.5)
            im = draw_partmap(star, proj=proj_now, shape=shape, qscale=qscale, vmax=vmax, crho=True, method=part_method,
                         weights=lums)
            colorbar_label = 'Stellar flux'
            mode_label = 'Stars'
        elif (mode_now == 'sdss'):
            star = part['star']
            filters = ['SDSS_i', 'SDSS_r', 'SDSS_g']
            images = []
            for filter_name in filters:
                mags = phot.measure_magnitude(star, filter_name=filter_name, total=False)
                lums = 10**(-mags/2.5)
                image = partmap(star, proj=proj_now, shape=shape, crho=True, method=part_method, weights=lums)
                images.append(image)

            images = np.array(images) / vmax
            rgb = make_lupton_rgb(*images, Q=10, stretch=0.5)
            im = draw_image(rgb, extent=np.concatenate(snap.box[proj_now]))
            mode_label = 'SDSS'
        elif (mode_now == 'dm'):
            dm = part['dm']
            im = draw_partmap(dm, proj=proj_now, shape=shape, qscale=qscale, vmax=vmax, crho=True, method=part_method,
                                 unit='Msol/pc2')
            mode_label = 'DM'
            colorbar_label = 'DM density\nM$_{\odot}$ pc$^{-2}$'
        elif (mode_now == 'gas' or mode_now == 'rho'):
            im = draw_gasmap(cell, proj=proj_now, shape=shape, qscale=qscale, vmax=vmax, mode='crho', cmap=ccm.hesperia,
                                interp_order=interp_order, unit='Msol/pc2', method=cell_method)
            mode_label = 'Gas - Density'
            colorbar_label = 'Gas density\nM$_{\odot}$ pc$^{-2}$'
        elif (mode_now == 'temp' or mode_now == 'T'):
            im = draw_gasmap(cell, proj=proj_now, shape=shape, qscale=qscale, vmax=vmax, mode='T', cmap=ccm.hesperia,
                        unit='K', method=cell_method)
            mode_label = 'Gas - Temperature'
            colorbar_label = 'Gas temperature\nK'
        elif (mode_now == 'dust'):
            im = draw_gasmap(cell, proj=proj_now, shape=shape, qscale=qscale, vmax=vmax, mode='dust', cmap=ccm.lacerta,
                                interp_order=interp_order, method=cell_method)
            mode_label = 'Gas - Dust'
            colorbar_label = 'Dust fraction'
        elif (mode_now == 'metal'):
            im = draw_gasmap(cell, proj=proj_now, shape=shape, qscale=qscale, vmax=vmax, mode='metal', cmap=ccm.lacerta,
                                interp_order=interp_order, method=cell_method)
            mode_label = 'Gas - Metallicity'
            colorbar_label = 'Metal fraction'
        else:
            raise ValueError('Unknown mode: ', mode_now)

        ax_label = ''
        if(show_smbh[ipan]):
            if(smbh_labels[ipan]):
                labels = ['%.2f' % m for m in np.log10(smbh['m', 'Msol'])]
            else:
                labels = None
            draw_smbhs(smbh, proj=proj_now, labels=labels, color='gray',
                       fontsize=fontsize, mass_range=[4, 8], facecolor='none', s=100)

        if(axis):
            plt.axis('on')
            set_ticks_unit(snap, proj_now, 'kpc')
        if(ipan == 0):
            if (target is not None):
                ax_label += 'ID = %d\n' % target['id']
                if(source == 'SINKPROPS'):
                    ax_label += 'log M$_{\\bullet}$ = %.2f' % np.log10(target['m']/snap.unit['Msol'])
                    ax_label += '\n'
                elif(source == 'GalaxyMaker'):
                    ax_label += 'log M$_{gal}$ = %.2f' % np.log10(target['m'])
                    ax_label += '\n'

            ax_label2 = ''
            for prop in props:
                if(prop == 'contam'):
                    dm = part['dm']
                    contam = np.sum(dm[dm['m'] > np.min(dm['m']) * 2]['m']) / np.sum(dm['m'])
                    f1 = contam
                    if(contam > 0):
                        ax_label2 += '\nf$_{low}$ = %.2f' % f1
                if(prop == 'fedd' and sink.size > 0):
                    if(source in ['SINKPROPS', 'sink']):
                        mms = sink[sink['id'] == target['id']]
                    else:
                        mms = sink[np.argmax(sink['m'])]
                    f1 = mms['dM'] / mms['dMEd']
                    ax_label2 += '\nf$_{Edd}$ = %.3f' % f1
                if(prop == 'sfr'):
                    f1 = np.sum(star[star['age', 'Myr'] < 100]['m', 'Msol'] / 1E8)
                    ax_label2 += '\nSFR$_{100 Myr}$ = %.2f' % f1
            dr.axlabel(ax_label2, 'right bottom', color='white', fontsize=fontsize, linespacing=1.5)

        if(mode_now in ['star', 'phot', 'sdss']):
            ax_label += 'log M$_*$ = %.2f\n' % np.log10(np.sum(star['m', 'Msol']))
        elif(mode_now in ['dm']):
            ax_label += 'log M$_{DM}$ = %.2f\n' % np.log10(np.sum(dm['m', 'Msol']))
        elif (mode_now in ['gas', 'rho']):
            ax_label += 'log M$_{gas}$ = %.2f\n' % np.log10(np.sum(cell['m', 'Msol']))

        dr.axlabel(ax_label, 'left top', color = 'white', fontsize=fontsize, linespacing=1.5)
        if(mode_label is not None):
            dr.axlabel(mode_label + ('\nz = %.3f' % snap.z), 'right top', color='white', fontsize=fontsize, linespacing=1.5)

        if(ruler):
            radius_in_unit = (snap.box[proj_now[0], 1] - snap.box[proj_now[0], 0]) * 0.5 / snap.unit[radius_unit]
            if(ruler_size_in_radius_unit is None):
                ruler_size_in_radius_unit = int(radius_in_unit / 2.5)
            bar_length = 0.5 / radius_in_unit * ruler_size_in_radius_unit
            rect = Rectangle([0.075, 0.1], bar_length, 0.02, transform=plt.gca().transAxes, color='white', zorder=100)
            plt.gca().add_patch(rect)
            plt.text(0.075+bar_length/2, 0.08, '%g %s' % (ruler_size_in_radius_unit, radius_unit), ha='center',
                     va='top', color='white', transform=plt.gca().transAxes, fontsize=8)

        if colorbar:
            divider = make_axes_locatable(plt.gca())
            if colorbar_kw['orientation'] == 'horizontal':
                cax = divider.append_axes('bottom', size=colorbar_size, pad=0.05)
                cbar = plt.colorbar(im, cax=cax, **colorbar_kw)
                subplots_adjust_kw['hspace'] += 0.05
            elif colorbar_kw['orientation'] == 'vertical':
                cax = divider.append_axes('right', size=colorbar_size, pad=0.05)
                cbar = plt.colorbar(im, cax=cax, **colorbar_kw)
                subplots_adjust_kw['wspace'] += 0.05
            else:
                raise ValueError("Unknown colorbar orientation: ", colorbar_kw['orientation'])
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label(label=colorbar_label, size=fontsize)
    plt.subplots_adjust(**subplots_adjust_kw)


    if (savefile is not None):
        save_figure(savefile)

def SDSS_rgb(star, filename=None, **kwargs):
    mags = phot.measure_magnitude(star, 'SDSS_g')
    lums = 10**(-mags/2.5)
    g = partmap(star, weights=lums, **kwargs)

    mags = phot.measure_magnitude(star, 'SDSS_r')
    lums = 10**(-mags/2.5)
    r = partmap(star, weights=lums, **kwargs)

    mags = phot.measure_magnitude(star, 'SDSS_i')
    lums = 10**(-mags/2.5)
    i = partmap(star, weights=lums, **kwargs)

    maxi = np.max([np.quantile(g, 0.99), np.quantile(r, 0.99), np.quantile(i, 0.99)])

    g = g / maxi
    r = r / maxi
    i = i / maxi

    rgb_default = make_lupton_rgb(i, r, g, Q=10, stretch=0.5, filename=filename)
    return rgb_default