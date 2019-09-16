import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import rescale, resize, warp, EuclideanTransform, AffineTransform
from rur import drawer as dr, utool
from rur import uri
from rur.utool import Timer, get_vector, bin_centers, rss
from matplotlib.patches import RegularPolygon, Rectangle
from scipy.ndimage.filters import gaussian_filter1d
from collections.abc import Iterable

if(__name__=='__main__'):
    verbose = 1
else:
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


def lvlmap(cell, box=None, proj=[0, 1], shape=None, minlvl=None, maxlvl=None, subpx_crop=True):
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

def set_weights(mode, cell, unit):
    quantity = None
    if(mode=='v'):
        # averge mass-weighted velocity along LOS
        weights = cell['rho']
    elif(mode=='T'):
        # averge mass-weighted temperature along LOS
        weights = cell['rho']
    elif(mode=='metal'):
        # averge mass-weighted metallicity along LOS
        weights = cell['rho']
    elif(mode=='mach'):
        # averge mass-weighted mach number along LOS
        weights = cell['rho']
    elif (mode == 'vel'):
        # averge mass-weighted velocity along LOS
        weights = cell['rho']
        quantity = rss(cell[mode, unit])
    elif(mode=='zoom'):
        # cumulative refinement paramster along LOS
        weights = np.full(cell.size, 1)
    elif(mode=='rho'):
        # average density along LOS
        weights = np.full(cell.size, 1)
    else:
        raise ValueError('Unknown gasmap mode.')
    if(quantity is None):
        if(unit is None):
            quantity = cell[mode]
        else:
            quantity = cell[mode, unit]

    return quantity, weights

def gasmap(cell, box=None, proj=[0, 1], shape=None, mode='rho', unit=None, minlvl=None, maxlvl=None, subpx_crop=True, anti_aliasing=False):
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
    for ilvl in known_lvls:
        mask = lvl==ilvl
        cell_lvl = cell[mask]

        xm = get_vector(cell_lvl)
        qm, wm = set_weights(mode, cell_lvl, unit)

        binlvl = np.min([ilvl, maxlvl])
        binsize = basebin * 2**(binlvl-minlvl)

        # convert coordinates to map
        hist_weight = np.histogram2d(xm[:, proj[0]], xm[:, proj[1]], bins=binsize, range=edge, weights=wm)[0]
        hist_map = np.histogram2d(xm[:, proj[0]], xm[:, proj[1]],
                                  bins=binsize, range=edge, weights=qm*wm)[0]
        hist_map = np.divide(hist_map, hist_weight, where=hist_weight!=0)

        if (ilvl < maxlvl):
            ibin = ilvl
        else:
            ibin = ilvl*3 - maxlvl*2

        depth_map_new = depth_map + hist_weight * 0.5 ** ibin
        mask_active = (hist_weight > 0) & (depth_map_new > 0)

        image[mask_active] = (np.divide(image * depth_map + hist_map * hist_weight * 0.5 ** ibin, depth_map_new,
                                        where=mask_active))[mask_active]
        depth_map = depth_map_new

        if(ilvl < maxlvl):
            image = rescale(image, 2, mode='constant', order=0, multichannel=False, anti_aliasing=anti_aliasing)
            depth_map = rescale(depth_map, 2, mode='constant', order=0, multichannel=False, anti_aliasing=anti_aliasing)


    crop_range = ((box_proj.T - edge[:, 0]) / (edge[:, 1] - edge[:, 0])).T
    if(subpx_crop):
        image = crop_float(image, crop_range, output_shape=shape)
    else:
        image = crop(image, crop_range)
        if(shape is not None):
            image = resize(image, shape)

    timer.record()
    return image.T


def draw_gasmap(cell, box=None, proj=[0, 1], shape=500, extent=None, mode='rho', unit=None, minlvl=None, maxlvl=None, subpx_crop=True, anti_aliasing=False, **kwargs):
    if(box is None and isinstance(cell, uri.RamsesSnapshot.Cell)):
        box = cell.snap.box

    image = gasmap(cell, box, proj, mode=mode, unit=unit, shape=shape, minlvl=minlvl, maxlvl=maxlvl, subpx_crop=subpx_crop, anti_aliasing=anti_aliasing)

    box_proj = get_box_proj(box, proj)
    if extent is None:
        extent = np.concatenate(box_proj)

    draw_image(image, extent=extent, **kwargs)

def tracermap(tracer_part, box=None, proj=[0, 1], shape=None, mode='rho', unit=None, minlvl=None, maxlvl=None, subpx_crop=True, anti_aliasing=False):
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

    dims = np.arange(3)
    los = dims[np.isin(dims, proj, invert=True, assume_unique=True)][0]
    depth = np.diff(box[los])

    for ilvl in known_lvls:
        mask = lvl==ilvl
        cell_lvl = tracer_part[mask]

        xm = get_vector(cell_lvl)
        qm = cell_lvl['m'] / cell_lvl['dx']**2

        binlvl = np.min([ilvl, maxlvl])
        binsize = basebin * 2**(binlvl-minlvl)

        # convert coordinates to map
        hist_map = np.histogram2d(xm[:, proj[0]], xm[:, proj[1]], bins=binsize, range=edge, weights=qm)[0]

        ibin = ilvl - binlvl

        image += hist_map * 0.5**ibin

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


def partmap(part, box=None, proj=[0, 1], shape=500, weights=None, unit=None, method='hist', x=None, smooth=16):
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

    image /= depth
    if(unit is not None):
        image /= part.snap.unit[unit]

    timer.record()

    return image.T


def draw_partmap(part, box=None, proj=[0, 1], shape=500, extent=None, weights=None, unit=None, method='hist', smooth=16, **kwargs):
    if(box is None and isinstance(part, uri.RamsesSnapshot.Particle)):
        box = part.snap.box

    image = partmap(part, box, proj, shape, weights, unit, method, smooth=smooth)

    box_proj = get_box_proj(box, proj)

    if extent is None:
        extent = np.concatenate(box_proj)

    draw_image(image, extent=extent, **kwargs)


def rgb_image(image,  vmin=None, vmax=None, qmin=None, qmax=None, qscale=3., normmode='log', nanzero=False, imfilter=None, cmap=plt.cm.viridis):
    if(not len(image.shape)>2):
        image = norm(image, vmin, vmax, qmin, qmax, qscale=qscale, mode=normmode, nanzero=nanzero)

    if(imfilter is not None):
        image = imfilter(image)

    return cmap(image)


def draw_image(image, extent=None, vmin=None, vmax=None, qmin=None, qmax=None, qscale=3., normmode='log', nanzero=False, imfilter=None, **kwargs):
    if(not len(image.shape)>2):
        image = norm(image, vmin, vmax, qmin, qmax, qscale=qscale, mode=normmode, nanzero=nanzero)

    if(imfilter is not None):
        image = imfilter(image)

    ims = plt.imshow(image, extent=extent, vmin=0, vmax=1, origin='lower', zorder=1, **kwargs)
    return ims



def save_image(image, fname, cmap=plt.cm.viridis, vmin=None, vmax=None, qmin=None, qmax=None, qscale=3., normmode='log', nanzero=False):
    if(len(image.shape)>2):
        plt.imsave(fname, image, origin='lower')
    else:
        image_norm = cmap(norm(image, vmin, vmax, qmin, qmax, qscale, mode=normmode, nanzero=nanzero))

        plt.imsave(fname, image_norm, origin='lower')

def draw_contour(image, extent, vmin=None, vmax=None, qmin=None, qmax=None, qscale=None, normmode='log', nlevel=3, **kwargs):
    image = norm(image, vmin, vmax, qmin, qmax, qscale=qscale, mode=normmode)

    xarr = bin_centers(extent[0], extent[1], image.shape[0])
    yarr = bin_centers(extent[2], extent[3], image.shape[1])

    return plt.contour(xarr, yarr, image, **kwargs)



def draw_points(points, box=None, proj=[0, 1], **kwargs):
    x = get_vector(points)
    if box is None:
        box = default_box
    box = np.array(box)
    x =x[uri.box_mask(x, box)]

    plt.scatter(x[:, proj[0]], x[:, proj[1]], zorder=100, **kwargs)


def draw_smbhs(smbh, box=None, proj=[0, 1], s=30, cmap=None, color='k', mass_range=None, **kwargs):
    if(box is None and isinstance(smbh, uri.RamsesSnapshot.Particle)):
        box = smbh.snap.box
    box_proj = get_box_proj(box, proj)
    smbh = smbh[uri.box_mask(uri.get_vector(smbh), box)]

    poss = uri.get_vector(smbh)
    mass = smbh['m', 'Msol']
    if(mass_range is None):
        m_max = np.max(mass)
        m_min = np.min(mass)
    else:
        m_min = 10.**mass_range[0]
        m_max = 10.**mass_range[1]

    mass_scale = norm(mass, m_min, m_max)

    s =  (mass_scale)**2 * s + 1

    if(cmap is not None):
        color = cmap(mass_scale)

    plt.scatter(poss[:, proj[0]], poss[:, proj[1]], s=s, color=color, **kwargs)

    plt.xlim(box_proj[0])
    plt.ylim(box_proj[1])



def draw_halos(halos, box=None, ax=None, proj=[0, 1], mass_range=None, cmap=plt.cm.jet, color=None, labels=None, size_key='rvir', shape='circle', fontsize=10, extents=None, **kwargs):
    mask = uri.box_mask(get_vector(halos), box=box)
    proj_keys = np.array(['x', 'y', 'z'])[proj]
    if ax is None:
        ax = plt.gca()
    if(labels is None):
        labels = np.full(halos.size, None)
    if(not isinstance(extents, Iterable)):
        extents = np.full(halos.size, extents)
    else:
        extents = extents

    halos = np.array(halos)[mask]
    labels = np.array(labels)[mask]
    extents = np.array(extents)[mask]

    print('Drawing %d halos...' % halos.size)

    if mass_range is None:
        mass_range = np.log10(np.array([np.min(halos['mvir']), np.max(halos['mvir'])]))

    for halo, label, extent in zip(halos, labels, extents):
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
        dr.colorline(gaussian_filter1d(pos[:, proj[0]], 1), gaussian_filter1d(pos[:, proj[1]], 1), np.log10(halos['mvir']), cmap=plt.cm.rainbow, linewidth=((maxmass-10)/2)**1.5, norm=plt.Normalize(*mass_range), alpha=alpha, zorder=10)


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


def composite_image(images, cmaps, weights=None, vmins=None, vmaxs=None, qmins=None, qmaxs=None, qscales=3., mode='average', normmodes=None):
    rgbs = []

    nimg = len(images)

    if(vmins is None):
        vmins = np.full(nimg, None)
    if(vmaxs is None):
        vmaxs = np.full(nimg, None)
    if(qmins is None):
        qmins = np.full(nimg, None)
    if(qmaxs is None):
        qmaxs = np.full(nimg, None)
    if(isinstance(qscales, float)):
        qscales = np.full(nimg, qscales)
    if(normmodes is None):
        normmodes = np.full(nimg, 'log')

    if(verbose>=2):
        print('vmins:', vmins)
        print('vmaxs:', vmaxs)

    for image, cmap, vmin, vmax, qmin, qmax, qscale, normmode in zip(images, cmaps, vmins, vmaxs, qmins, qmaxs, qscales, normmodes):
        rgbs.append(cmap(norm(image, vmin, vmax, qmin, qmax, qscale=qscale, mode=normmode)))
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


def draw_multicomponent(part, cell, snap, proj=[0, 1], maxlvl=14):
    fig = plt.figure(figsize=(18, 10))
    box = snap.box

    cfied = uri.part_classifier(part, snap.params['star'])

    dm = cfied['dm']
    star = cfied['star']

    gas_rho = gasmap(cell, maxlvl=maxlvl, box=box, proj=proj)
    part_dm = partmap(dm, bins=gas_rho.shape[::-1], box=box, proj=proj, adaptive=True)
    part_star = partmap(star, bins=gas_rho.shape[::-1], box=box, proj=proj, adaptive=False)

    gas_T = gasmap(cell, maxlvl=maxlvl, box=box, proj=proj, mode='T')
    gas_v = gasmap(cell, maxlvl=maxlvl, box=box, proj=proj, mode='v')
    gas_metal = gasmap(cell, maxlvl=maxlvl, box=box, proj=proj, mode='metal')
    #gas_ref = gasmap(cell, maxlvl=maxlvl, box=box, proj=proj, mode='ref')

    cim = composite_image([gas_rho, part_dm, part_star, gas_T], [dr.ccm.forest, dr.ccm.darkmatter, dr.ccm.DarkYellow, dr.ccm.DarkRed], weights=[1., 1., 1, 0.5], mode='product',
                                  qmaxs=[0.999, 0.999, 0.99, 0.999], qmins=[0.04, 0.04, 0.04, 0.04], normmodes=['log', 'log', 'log', 'linear'])

    grid = dr.gridplot(2, 4, wspace=0.07, hspace=0.07, labpanel=[
        'left', 'top',
        ['z=%.3f\nM=%.3e\nM/M*=%.2f' % (1 / snap.params['aexp'] - 1, np.sum(dm['m']) / uri.Msol, np.sum(dm['m']) / np.sum(star['m'])),
         'Gas', 'DM', 'Star', 'Gas-Temperature', 'Gas-Velocity', 'Gas-Metal', 'Refinement']],
                       xlims=[box[proj[0]], box[proj[0]], box[proj[0]], box[proj[0]]], ylims=[box[proj[1]], box[proj[1]]], panlabcolor='w', panlabsize=10)
    plt.subplot(grid[0])
    plt.imshow(cim, extent=np.concatenate([box[proj[0]], box[proj[1]]]), origin='lower')
    plt.xlim(box[proj[0]])
    plt.ylim(box[proj[1]])

    plt.subplot(grid[1])
    draw_image(gas_rho, extent=np.concatenate([box[proj[0]], box[proj[1]]]), cmap=plt.cm.viridis)
    plt.xlim(box[proj[0]])
    plt.ylim(box[proj[1]])

    plt.subplot(grid[2])
    draw_image(part_dm, extent=np.concatenate([box[proj[0]], box[proj[1]]]), cmap=dr.ccm.darkmatter)
    plt.xlim(box[proj[0]])
    plt.ylim(box[proj[1]])

    plt.subplot(grid[3])
    draw_image(part_star, extent=np.concatenate([box[proj[0]], box[proj[1]]]), cmap=plt.cm.afmhot, qmax=0.99)
    plt.xlim(box[proj[0]])
    plt.ylim(box[proj[1]])

    plt.subplot(grid[4])
    draw_image(gas_T, extent=np.concatenate([box[proj[0]], box[proj[1]]]), cmap=plt.cm.inferno)
    plt.xlim(box[proj[0]])
    plt.ylim(box[proj[1]])

    plt.subplot(grid[5])
    draw_image(gas_v, extent=np.concatenate([box[proj[0]], box[proj[1]]]), cmap=dr.ccm.forest)
    plt.xlim(box[proj[0]])
    plt.ylim(box[proj[1]])

    plt.subplot(grid[6])
    draw_image(gas_metal, extent=np.concatenate([box[proj[0]], box[proj[1]]]), cmap=plt.cm.winter)
    plt.xlim(box[proj[0]])
    plt.ylim(box[proj[1]])

    plt.subplot(grid[7])
    #draw_image(gas_ref, extent=np.concatenate([box[proj[0]], box[proj[1]]]), cmap=plt.cm.jet)
    plt.xlim(box[proj[0]])
    plt.ylim(box[proj[1]])

    fig.show()


def norm(v, vmin=None, vmax=None, qmin=None, qmax=None, qscale=3., mode='log', nanzero=False):
    v = v.copy()
    if(mode == 'log'):
        if(vmax is None):
            vmax = np.nanmax(v)
        if(qscale is None):
            if(vmin is not None):
                qscale = np.log10(vmax-vmin)
            else:
                qscale = np.log10(vmax)-np.log10(np.nanmin(v[v>0]))
        if(vmin is None):
            vmin = 10. ** (np.log10(vmax)-qscale)
        if(qmax is not None):
            vmax = 10. ** (np.log10(qmax)-qscale*(1-qmax))
        if(qmin is not None):
            vmin = 10. ** (np.log10(qmin) + qscale * qmin)

        v[v<vmin] = vmin
        v = np.log10(v/vmin) / np.log10(vmax/vmin)
    elif(mode == 'linear'):
        if(vmax is None):
            vmax = np.max(v)
        if(vmin is None):
            vmin = np.min(v)
        if(qmax is not None):
            vmax += (vmax-vmin)*qmax
        if(qmin is not None):
            vmin += (vmax-vmin)*qmin
        v = (v - vmin) / (vmax - vmin)

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

def set_ticks_unit(snap, proj=[0, 1], unit='kpc', nticks=4):
    box_proj = get_box_proj(snap.box, proj)
    xr = np.array(box_proj[0])
    yr = np.array(box_proj[1])

    xc = np.mean(xr)
    yc = np.mean(yr)

    lunit = snap.unit[unit]

    xticks = get_tickvalues((xr-xc)/lunit, nticks)
    yticks = get_tickvalues((yr-yc)/lunit, nticks)

    plt.xticks(xticks*snap.unit[unit]+xc, labels=xticks)
    plt.yticks(yticks*snap.unit[unit]+yc, labels=yticks)

    plt.xlabel('X (%s)' % unit)
    plt.ylabel('Y (%s)' % unit)


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
    if(order>=0):
        ticks = ticks.astype(int)
    return ticks
