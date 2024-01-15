from tqdm import tqdm
import numpy as np
from rur import utool
from scipy.spatial import cKDTree as KDTree

def match(a, b, search_radius, radius_base='a', priority='m', inclusive=False, kw_query={}):
    """
    position-based matching
    a, b: array that includes 'x', 'y', 'z' fields
    search_radius: radius to search matching target, can be either scalar or array with b.size
    inclusive: whether to match multiple b to a
    use example 1: a is smbh and b is galaxy
    use example 2: a is galaxy and b is halo
    """
    matched = np.full(a.shape, -1, dtype='i8')
    match_pool = np.full(shape=(a.size, b.size), fill_value=False)

    akey = np.argsort(a[priority])[::-1]
    bkey = np.argsort(b[priority])[::-1]

    pos_a, pos_b = utool.get_vector(a[akey]), utool.get_vector(b[bkey])
    if radius_base == 'a':
        tree_b = KDTree(pos_b)
        query = tree_b.query_ball_point(pos_a, search_radius, **kw_query)
        for idx, q in enumerate(query):
            match_pool[idx, q] = True

    elif radius_base == 'b':
        tree_a = KDTree(pos_a)
        query = tree_a.query_ball_point(pos_b, search_radius, **kw_query)
        for idx, q in enumerate(query):
            match_pool[q, idx] = True
    else:
        raise ValueError("Unknown radius_base: %s" % radius_base)

    for aidx in range(a.size):
        row = match_pool[aidx]
        if np.any(row):
            bidx_max = np.argmax(row)
            matched[akey[aidx]] = bkey[bidx_max]
            if not inclusive:
                match_pool[:, bidx_max] = False
    return matched
