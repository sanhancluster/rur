from tqdm import tqdm
import numpy as np
from rur import utool
from scipy.spatial import KDTree

def match(a, b, search_radius, radius_base='a', priority='m', priority_b=None, inclusive=False, kw_query={}):
    """
    position-based matching
    a, b: array that includes 'x', 'y', 'z' fields
    search_radius: radius to search matching target, can be either scalar or array with b.size, if array, the matching is conducted in each element of search_radius sequentially.
    radius_base: 'a' or 'b', which array to use as base for search radius
    priority: field name to prioritize matching order
    inclusive: whether to match multiple b to a
    use example 1: a is smbh and b is galaxy
    use example 2: a is galaxy and b is halo
    """
    if np.isscalar(search_radius):
        search_radius = np.atleast_1d(search_radius)
    if inclusive:
        search_radius = search_radius[-1:]
    
    matched = np.full(a.shape, -1, dtype='i8')
    match_pool_base = np.full(shape=(b.size), fill_value=True)
    for r in search_radius:
        match_pool = np.full(shape=(a.size, b.size), fill_value=False)
        akey = np.argsort(a[priority])[::-1]

        if priority_b is not None:
            if priority_b != 'distance':
                priority_b = priority
                weight_b = b[priority_b]
                bkey = np.argsort(weight_b)[::-1]
        else:
            weight_b = b[priority]
            bkey = np.argsort(weight_b)[::-1]

        pos_a, pos_b = utool.get_vector(a[akey]), utool.get_vector(b[bkey])
        if radius_base == 'a':
            tree_b = KDTree(pos_b)
            query = tree_b.query_ball_point(pos_a, r, **kw_query)
            for idx, q in enumerate(query):
                match_pool[idx, q] = True

        elif radius_base == 'b':
            tree_a = KDTree(pos_a)
            query = tree_a.query_ball_point(pos_b, r, **kw_query)
            for idx, q in enumerate(query):
                match_pool[q, idx] = True
        else:
            raise ValueError("Unknown radius_base: %s" % radius_base)

        match_pool &= match_pool_base[np.newaxis, :]

        for aidx in range(a.size):
            # get the row of match pool for this aidx
            row = match_pool[aidx]
            if np.any(row):
                # get the first bidx that matches
                if priority_b == 'distance':
                    distances = np.linalg.norm(pos_b[row] - pos_a[aidx], axis=1)
                    bidx_max = np.argmin(distances)
                else:
                    bidx_max = np.argmax(row)

                matched[akey[aidx]] = bkey[bidx_max]
                if not inclusive:
                    match_pool_base[bidx_max] = False
                    match_pool[:, bidx_max] = False
    return matched