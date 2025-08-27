from rur import drawer
import numpy as np

def get_amr(data):
    connectivity = data['connectivity']

    # n_vertices varies with ndim
    n_vertices = connectivity.shape[1]

    # coordinates are always in 3-dim
    coordinates = np.array(data['coordinates'])
    domain_length_max = np.max(np.max(coordinates, axis=0) - np.min(coordinates, axis=0))
    vertices = np.reshape(coordinates[connectivity], newshape=(-1, n_vertices, 3))
    centers = np.mean(vertices, axis=1)
    levels = np.round(-np.log2((vertices[:, 1, 0] - vertices[:, 0, 0]) / domain_length_max)).astype(int)
    return centers, levels


def amr_projection(data, quantity='rho', shape=100, lims=None, mode='sum', plot_method='cic', projection=['x', 'y'], ndim=3):
    # produce a projection plot of a quantity in the dyablo data
    amr = get_amr(data)
    centers, levels = amr

    return drawer.amr_projection(centers, levels, data[quantity], shape=shape, lims=lims, mode=mode, plot_method=plot_method, projection=projection, ndim=ndim)
