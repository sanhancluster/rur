from tqdm import tqdm
import numpy as np
from rur import utool

def match(a, b, search_radius, priority='m', inclusive=False):
    """
    position-based matching
    a, b: array that includes 'x', 'y', 'z' fields
    search_radius: radius to search matching target, can be either scalar or array with b.size
    inclusive: whether to match multiple b to a
    supports 1 to 1 match only
    use example 1: a is smbh and b is galaxy
    use example 2: a is galaxy and b is halo
    """
    free = np.full(b.size, True)
    matched = np.full(a.shape, -1, dtype='i8')
    key = np.argsort(a[priority])[::-1]
    for idx in key:
        target = a[idx]
        dists = utool.get_distance(target, b)
        mask = (dists < search_radius) & free
        if np.any(mask):
            matched_idx = np.argmax(b[priority]*mask)
            matched[idx] = matched_idx
            if not inclusive:
                free[matched_idx] = False

    return matched
