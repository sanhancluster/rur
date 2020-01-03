import os
from numpy.core.records import fromarrays
import numpy as np
from rur.utool import Timer, get_vector, type_of_script, dump, load, pairing, get_distance, rss, ss,\
    set_vector, discrete_hist2d, weighted_quantile
from rur.readhtm import readhtm as readh
from rur import uri
from scipy.stats import mode
from numpy.lib.recfunctions import append_fields, drop_fields
import gc
import string
from glob import glob
from parse import parse
if(type_of_script() == 'jupyter'):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


chars = string.ascii_lowercase

dtype_ct = [
    ('scale', 'f8'), ('id', 'i4'), ('desc_scale', 'f8'), ('desc_id', 'i4'), ('num_prog', 'i4'),
    ('pid', 'i4'), ('upid', 'i4'), ('desc_pid', 'i4'), ('phantom', 'i4'),
    ('sam_mvir', 'f8'), ('mvir', 'f8'), ('rvir', 'f8'), ('rs', 'f8'), ('vrms', 'f8'), ('mmp', 'i4'),
    ('scale_of_last_MM', 'f8'), ('vmax', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
    ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'), ('Jx', 'f8'), ('Jy', 'f8'), ('Jz', 'f8'), ('Spin', 'f8'),
    ('Breadth_first_ID', 'i4'), ('Depth_first_ID', 'i4'), ('Tree_root_ID', 'i4'),
    ('Orig_halo_ID', 'i4'),
    ('Snap_idx', 'i4'), ('Next_coprogenitor_depthfirst_ID', 'i4'),
    ('Last_progenitor_depthfirst_ID', 'i4'),
    ('Last_mainleaf_depthfirst_ID', 'i4'), ('Tidal_Force', 'f8'), ('Tidal_ID', 'i4'),
    ('mbound_vir', 'f8'), ('rvmax', 'f8'), ('E', 'f8'), ('PosUncertainty', 'f8'),
    ('VelUncertainty', 'f8'),
    ('bulk_vx', 'f8'), ('bulk_vy', 'f8'), ('bulk_vz', 'f8'), ('BulkVelUnc', 'f8'), ('n_core', 'i4'),
    ('m200b', 'f8'), ('m200c', 'f8'), ('m500c', 'f8'), ('m2500c', 'f8'), ('Xoff', 'f8'), ('Voff', 'f8'),
    ('spin_bullock', 'f8'), ('b_to_a', 'f8'), ('c_to_a', 'f8'), ('A[x]', 'f8'), ('A[y]', 'f8'), ('A[z]', 'f8'),
    ('b_to_a(500c)', 'f8'), ('c_to_a(500c)', 'f8'), ('A[x](500c)', 'f8'), ('A[y](500c)', 'f8'),
    ('A[z](500c)', 'f8'),
    ('Rs', 'f8'), ('Rs_Klypin', 'f8'), ('T/|U|', 'f8'), ('M_pe_Behroozi', 'f8'), ('M_pe_Diemer', 'f8'),
    ('Halfmass_Radius', 'f8'),
    ('idx', 'i4'), ('i_so', 'i4'), ('i_ph', 'i4'), ('num_cp', 'i4'), ('mmetric', 'f8'),
]

dtype_ct_old = [
    ('scale', '<f4'), ('id', '<i8'), ('desc_scale', '<f8'), ('desc_id', '<i8'), ('num_prog', '<i8'),
    ('pid', '<i8'), ('upid', '<i8'), ('desc_pid', '<i8'), ('phantom', '<i8'),
    ('sam_mvir', '<f8'), ('mvir', '<f8'), ('rvir', '<f8'), ('rs', '<f8'), ('vrms', '<f8'), ('mmp', '<i8'),
    ('scale_of_last_MM', '<f8'), ('vmax', '<f8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
    ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'), ('Jx', '<f8'), ('Jy', '<f8'), ('Jz', '<f8'), ('Spin', '<f8'),
    ('Breadth_first_ID', '<i8'), ('Depth_first_ID', '<i8'), ('Tree_root_ID', '<i8'), ('Orig_halo_ID', '<i8'),
    ('Snap_num', '<i8'), ('Next_coprogenitor_depthfirst_ID', '<i8'), ('Last_progenitor_depthfirst_ID', '<i8'),
    ('Last_mainleaf_depthfirst_ID', '<i8'), ('Tidal_Force', '<f8'), ('Tidal_ID', '<i8')]

timer = Timer()

def dtype_to_format(dtype):
    # Do something with stupid numpy dtype object formats
    fmt = []
    types = np.array(dtype.descr)[:, 1]
    for type in types:
        if('f' in type):
            fmt.append('%7e')
        elif('i' in type):
            fmt.append('%d')
    return " ".join(fmt)

def fix_out_of_order_fields(array):
    # why numpy does not have any solution about this?
    arrays = []
    names = array.dtype.names
    for name in names:
        arrays.append(array[name])

    output = fromarrays(arrays, names = names)
    return output

class HaloMaker:
    @staticmethod
    def unit_conversion(array, snap):
        # distance in code unit, mass in Msun
        mass_unit = 1E11
        array['m'] *= mass_unit
        array['mvir'] *= mass_unit
        boxsize_physical = snap['boxsize_physical']

        pos = get_vector(array)
        append_fields(array, names=['xp', 'yp', 'zp'], data=pos.T, usemask=False)
        array['x'] = array['x'] / boxsize_physical + 0.5
        array['y'] = array['y'] / boxsize_physical + 0.5
        array['z'] = array['z'] / boxsize_physical + 0.5
        array['rvir'] /= boxsize_physical
        array['r'] /= boxsize_physical
        array['rc'] /= boxsize_physical

        return array

    halo_dtype = [
        ('nparts', 'i4'), ('id', 'i4'), ('timestep', 'i4'), ('level', 'i4'),
        ('host', 'i4'), ('hostsub', 'i4'), ('nbsub', 'i4'), ('nextsub', 'i4'),
        ('aexp', 'f4'), ('m', 'f4'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'),
        ('Lx', 'f4'), ('Ly', 'f4'), ('Lz', 'f4'),
        ('r', 'f4'), ('a', 'f4'), ('b', 'f4'), ('c', 'f4'),
        ('ek', 'f4'), ('ep', 'f4'), ('et', 'f4'), ('spin', 'f4'),
        ('rvir', 'f4'), ('mvir', 'f4'), ('tvir', 'f4'), ('cvel', 'f4'),
        ('rho0', 'f4'), ('rc', 'f4')]

    galaxy_dtype = [
        ('nparts', 'i4'), ('id', 'i4'), ('timestep', 'i4'), ('level', 'i4'),
        ('host', 'i4'), ('hostsub', 'i4'), ('nbsub', 'i4'), ('nextsub', 'i4'),
        ('aexp', 'f4'), ('m', 'f4'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'),
        ('Lx', 'f4'), ('Ly', 'f4'), ('Lz', 'f4'),
        ('r', 'f4'), ('a', 'f4'), ('b', 'f4'), ('c', 'f4'),
        ('ek', 'f4'), ('ep', 'f4'), ('et', 'f4'), ('spin', 'f4'),
        ('sigma', 'f4'), ('sigma_bulge', 'f4'), ('m_bulge', 'f4'),
        ('rvir', 'f4'), ('mvir', 'f4'), ('tvir', 'f4'), ('cvel', 'f4'),
        ('rho0', 'f4'), ('rc', 'f4')]

    galaxy_dtype_dp = [
        ('nparts', 'i4'), ('id', 'i4'), ('timestep', 'i4'), ('level', 'i4'),
        ('host', 'i4'), ('hostsub', 'i4'), ('nbsub', 'i4'), ('nextsub', 'i4'),
        ('aexp', 'f8'), ('m', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'),
        ('Lx', 'f8'), ('Ly', 'f8'), ('Lz', 'f8'),
        ('r', 'f8'), ('a', 'f8'), ('b', 'f8'), ('c', 'f8'),
        ('ek', 'f8'), ('ep', 'f8'), ('et', 'f8'), ('spin', 'f8'),
        ('sigma', 'f8'), ('sigma_bulge', 'f8'), ('m_bulge', 'f8'),
        ('rvir', 'f8'), ('mvir', 'f8'), ('tvir', 'f8'), ('cvel', 'f8'),
        ('rho0', 'f8'), ('rc', 'f8')]


    @staticmethod
    def load(snap, path_in_repo=None, galaxy=False, full_path=None, load_parts=False, double_precision=None):
        # boxsize: comoving length of the box in Mpc
        repo = snap.repo
        start = snap.iout
        end = start+1
        if(double_precision is None):
            if(snap.mode=='nh' and galaxy):
                # NewHorizon galaxies uses dp, while halo data uses sp
                double_precision=True
            else:
                double_precision=False
        if (galaxy):
            if(path_in_repo is None):
                path_in_repo = 'GalaxyMaker/gal'
            if(not double_precision):
                dtype = HaloMaker.galaxy_dtype
            else:
                dtype = HaloMaker.galaxy_dtype_dp
        else:
            if(path_in_repo is None):
                path_in_repo = 'halo/DM'
            dtype = HaloMaker.halo_dtype
        path = os.path.join(repo, path_in_repo)

        if(full_path is not None):
            path = full_path

        #print("Searching for tree_brick in ", path)
        readh.read_bricks(path, galaxy, start, end, load_parts, double_precision)

        if(not double_precision):
            array = fromarrays([*readh.integer_table.T, *readh.real_table.T], dtype=dtype)
        else:
            array = fromarrays([*readh.integer_table.T, *readh.real_table_dp.T], dtype=dtype)
        array = HaloMaker.unit_conversion(array, snap)

        if(array.size==0):
            print("No tree_brick file found, or no halo found in %s" % path)

        if(load_parts):
            return array, readh.part_ids
        else:
            return array

    @staticmethod
    def cut_table(table, part_ids, mask):
        bound = np.concatenate([[0], np.cumsum(table['nparts'])])
        bounds = np.stack([bound[:-1][mask], bound[1:][mask]], axis=-1)

        table = table[mask]
        part_ids = np.concatenate([part_ids[idx[0]:idx[1]] for idx in bounds])
        return table, part_ids

    @staticmethod
    def match_halo(part, table, part_ids):
        full_ids = part['id']
        full_ids = np.abs(full_ids)
        idx = np.arange(full_ids.size)
        array_size = np.max(full_ids) + 1

        halo_ids = np.repeat(table['id'], table['nparts'])

        pool = np.zeros((2, array_size), dtype='i4')
        pool[0, full_ids] = idx
        pool[1, part_ids] = halo_ids
        mask = (pool[1] > 0)

        part_halo_ids = np.full(full_ids.size, fill_value=-1, dtype='i4')

        part_halo_ids[pool[0][mask]] = pool[1][mask]

        part.table = append_fields(part.table, names='halo_id', data=part_halo_ids, usemask=False)


class PhantomTree:
    path_in_repo = 'ptree'
    ptree_file = 'ptree.pkl'
    ptree_format = 'ptree_%05d.pkl'
    desc_format = 'desc%d%s'
    pass_format = 'pass%d%s'

    @staticmethod
    def from_halomaker(snap, lookup, rankup=1, path_in_repo=path_in_repo, max_part_size=None,
                       desc_format=desc_format, pass_format=pass_format, ptree_format=ptree_format, nparts_min=None,
                       part_array_buffer=1.1, skip_jumps=False, start_on_middle=False, **kwargs):
        print('Building PhantomTree from HaloMaker data in %s' % snap.repo)
        max_iout = snap.iout
        uri.timer.verbose = 0
        snap_iouts = np.arange(snap.iout, 0, -1)

        if (max_part_size is None):
            halo, part_ids = HaloMaker.load(snap, load_parts=True, **kwargs)
            max_part_size = int(np.max(part_ids) * part_array_buffer)

        part_pool = np.full((lookup, max_part_size), -1, dtype='i4')
        sizes = np.zeros(lookup, dtype='i4')
        halo_ids = []

        iterator = tqdm(snap_iouts, unit='snapshot')
        for iout in iterator:
            try:
                snap = uri.RamsesSnapshot(repo=snap.repo, iout=iout, mode=snap.mode)
            except FileNotFoundError:
                if(skip_jumps):
                    continue
                else:
                    iterator.close()
                    break

            halo, part_ids = HaloMaker.load(snap, load_parts=True, **kwargs)
            if(halo.size == 0):
                if(skip_jumps):
                    continue
                else:
                    iterator.close()
                    break
            if(nparts_min is not None):
                mask = halo['nparts'] >= nparts_min
                halo, part_ids = HaloMaker.cut_table(halo, part_ids, mask)
            if(halo.size == 0):
                iterator.close()
                break
            halo_idx = np.repeat(np.arange(halo.size), halo['nparts'])

            part_pool[1:lookup] = part_pool[0:lookup-1]
            sizes[1:lookup] = sizes[0:lookup-1]

            part_pool[0] = -1
            part_pool[0, part_ids] = halo_idx
            sizes[0] = halo.size

            buffer = max_iout - iout

            desc_ids = np.empty(shape=((lookup-1)*rankup, halo.size), dtype='i4')
            npass = np.empty(shape=((lookup-1)*rankup, halo.size), dtype='i4')

            for ilook in tqdm(np.arange(1, lookup), unit='lookup'):
                rank_range = slice((ilook - 1) * rankup, ilook * rankup)
                if(ilook<=buffer):
                    desc_idx, npass_arr = PhantomTree.find_desc(part_pool[np.array([0, ilook])], rankup=rankup)
                    desc_ids[rank_range] = halo_ids[ilook-1][desc_idx]
                    desc_ids[rank_range][desc_idx==-1] = -1
                    npass[rank_range] = npass_arr
                else:
                    desc_ids[rank_range] = -1
                    npass[rank_range] = 0

            halo_ids = [halo['id']] + halo_ids
            if(len(halo_ids)>lookup-1):
                halo_ids = halo_ids[:lookup-1]

            if(start_on_middle and sizes[-1]==0):
                print("Skipping output of iout = %d..." % iout)
                continue

            if(rankup==1):
                names = [desc_format % (ilook, "") for ilook in np.arange(1, lookup)] + [pass_format % (ilook, "") for ilook in np.arange(1, lookup)] + ['scale']
            elif(rankup>1):
                names = [desc_format % (ilook, rankchar) for ilook in np.arange(1, lookup) for rankchar in chars[:rankup]]\
                        + [pass_format % (ilook, rankchar) for ilook in np.arange(1, lookup) for rankchar in chars[:rankup]] + ['scale']
            scale = np.full(halo.size, snap['aexp'])
            halo = append_fields(halo, names=names, data=[*desc_ids, *npass, scale], usemask=False)
            path = os.path.join(snap.repo, path_in_repo, ptree_format % iout)
            dump(halo, path, msg=False)
        uri.timer.verbose = 1


    @staticmethod
    def find_desc(hid_arr, prog_n=None, next_n=None, rankup=1):
        """
        :param hid_arr: 2 * nparts array that specifies idx of halos for each particle.
        :param prog_n: number of progenitors.
        :param next_n: number of next halos.
        :return: list of descendent idx of each progentor, number of partices passed
        """
        hid_arr = hid_arr[:, (hid_arr[0] != -1) & (hid_arr[1] != -1)]
        if (prog_n is None):
            prog_n = np.max(hid_arr[0]) + 1
        if (next_n is None):
            next_n = np.max(hid_arr[1]) + 1
        """
        # performs discrete 2d histogram, which seems not available in numpy, much faster than np.histogram2d and np.add.at
        # multiprocessing implemented
        ncpu = mp.cpu_count()
        nparts = hid_arr.shape[-1]
        idxarr = nparts * np.arange(0, ncpu+1) // ncpu
        arrs = [hid_arr[:, bot:top] for bot, top in zip(idxarr[:-1], idxarr[1:])]
        
        pool = mp.Pool(processes=ncpu)
        hists = [pool.apply(PhantomTree.discrete_hist2d, args=((prog_n, next_n), hid_subarr)) for hid_subarr in arrs]
        pool.close()
        hist = np.sum(hists, axis=0)
        """
        hist = discrete_hist2d((prog_n, next_n), hid_arr, use_long=True)

        desc_idx = np.argpartition(hist, -(np.arange(rankup) + 1), axis=1)[:, -1:-(rankup + 1):-1].T
        npass = np.zeros(desc_idx.shape, dtype='i4')
        for irank in np.arange(rankup):
            npass[irank] = hist[(np.arange(prog_n), desc_idx[irank])]
        desc_idx[npass == 0] = -1

        return desc_idx, npass

    @staticmethod
    def merge_ptree(repo, iout_max, path_in_repo=path_in_repo, ptree_file=ptree_file, ptree_format=ptree_format, desc_format=desc_format, skip_jumps=False):
        dirpath = os.path.join(repo, path_in_repo)
        iout = iout_max
        ptree = []

        while(True):
            path = os.path.join(dirpath, ptree_format % iout)
            if(not os.path.exists(path)):
                if(skip_jumps):
                    if(iout == 1):
                        break
                    else:
                        iout -= 1
                        continue
                else:
                    break
            tree = load(path, msg=True)
            ptree.append(tree)
            iout -= 1
        if(len(ptree) == 0):
            raise FileNotFoundError('No ptree file found in %s' % dirpath)

        ptree = np.concatenate(ptree)
        ptree = PhantomTree.set_pairing_id(ptree, desc_format)

        dump(ptree, os.path.join(dirpath, ptree_file))

    @staticmethod
    def set_pairing_id(ptree, desc_format=desc_format, save_hmid=True, fix_range=10000):
        # fix range: temporary fix feature, if maximum id is smaller than this value, apply fix.
        names = ptree.dtype.names
        if(save_hmid):
            idx = names.index('id')
            names = np.array(names)
            names[idx] = 'hmid'
        else:
            ptree = drop_fields(ptree, 'id')

        halo_uid = pairing(ptree['timestep'], ptree['hmid'], ignore=-1)

        ilook = 1
        while(True):
            name1 = desc_format % (ilook, '')
            name2 = desc_format % (ilook, 'a')
            if(name1 in names):
                ptree[name1] = pairing(ptree['timestep'] + ilook, ptree[name1], ignore=-1)
                ilook += 1
            elif name2 in names:
                irank = 0
                while(True):
                    name2 = desc_format % (ilook, chars[irank])
                    if (name2 in names and np.max(ptree[name2])<fix_range):
                        ptree[name2] = pairing(ptree['timestep'] + ilook, ptree[name2], ignore=-1)
                        irank += 1
                    else:
                        break
                ilook += 1
            else:
                break

        ptree = append_fields(ptree, 'id', halo_uid, usemask=False)
        return ptree


    @staticmethod
    def process_tree(ptree, lookup=4, rankup=4, overwrite=False, purity_threshold=0.5, desc_format=desc_format, reduce=True):
        if(overwrite):
            ptree = drop_fields(ptree, ['desc', 'mainp', 'leaf', 'nprog', 'line'], usemask=False)

        id_ini = np.full(ptree.size, -1, dtype='i4')
        ptree = append_fields(
            ptree, ['desc', 'mainp', 'leaf', 'nprog', 'line'],
            [id_ini, id_ini, ptree['id'], np.zeros(ptree.size, dtype='i4'), np.zeros(ptree.size, dtype='f8')], usemask=False)
        ptree.sort(order='id')

        iout_max = np.max(ptree['timestep'])
        """
        names = ptree.dtype.names
        ilook = 1
        while('desc%da' % ilook in names):
            ilook += 1
        lookup = np.minimum(ilook, lookup)
        """
        if(rankup>1):
            desc_all = [ptree[desc_format % (ilook, 'a')] for ilook in np.arange(1, lookup)]
        else:
            desc_all = [ptree[desc_format % (ilook)] for ilook in np.arange(1, lookup)]
        desc_all = np.unique(np.concatenate(desc_all))

        leafs = ptree[np.isin(ptree['id'], desc_all, assume_unique=True, invert=True)]
        leafs_id = leafs['id']

        print('Number of leafs = %d' % leafs.size)
        PhantomTree.process_leafs(ptree, leafs_id, lookup, rankup, purity_threshold)

        rems = ptree[(ptree['desc'] == -1) & (ptree['timestep'] != iout_max)]
        print("Remainders: %d" % rems.size)
        rems_id = rems['id']
        PhantomTree.process_leafs(ptree, rems_id, lookup, rankup, purity_threshold)

        if(reduce):
            leafs, counts = np.unique(ptree['leaf'], return_counts=True)
            leafs = leafs[counts>lookup]

            reduce_mask = np.isin(ptree['leaf'], leafs)
            print('Reducing Trees... %d --> %d' % (np.sum(reduce_mask), reduce_mask.size))
            ptree = ptree[reduce_mask]

        return ptree

    @staticmethod
    def process_leafs(ptree, leafs_id, lookup, rankup, purity_threshold, desc_format=desc_format, pass_format=pass_format):
        for leaf_id in tqdm(leafs_id):
            halo = ptree[np.searchsorted(ptree['id'], leaf_id)]  # get a pointer
            halo['leaf'] = leaf_id
            while (True):
                if(rankup>1):
                    char='a'
                else:
                    char=''
                desc_ids = np.array([halo[desc_format % (i, char)] for i in np.arange(1, lookup )])
                mask = desc_ids > 0
                if (np.sum(mask) > 0):
                    desc_ids = desc_ids[mask]
                    if (rankup > 1):
                        npasses = np.array([halo[pass_format % (i, 'a')] for i in np.arange(1, lookup)])
                    else:
                        npasses = np.array([halo[pass_format % (i)] for i in np.arange(1, lookup)])
                    npasses = npasses[mask]

                    ilooks = np.searchsorted(ptree['id'], desc_ids)

                    descs = ptree[ilooks]
                    if('msf' in descs.dtype.names):
                        purity = npasses / (descs['nparts'] * (descs['m'] - descs['msf']) / descs['m'])
                    else:
                        purity = npasses / descs['nparts']

                    heiridx = np.argmax(purity)

                    if (purity[heiridx] - purity[0] < purity_threshold):
                        heiridx = 0
                    halo['desc'] = desc_ids[heiridx]

                    desc = ptree[np.searchsorted(ptree['id'], halo['desc'])]
                    if (desc['id'] != halo['desc']):
                        print('Something Wrong: descendent id does not exist')
                        break

                    desc['nprog'] += 1
                    if(desc['line']<=purity[heiridx]): # IMPORTANT: should include "equal"!
                        # write parent data, can be overwritten.
                        desc['leaf'] = leaf_id
                        desc['mainp'] = halo['id']
                        desc['line'] = purity[heiridx]
                    else:
                        break
                    halo = desc

                else:
                    break


    @staticmethod
    def load(repo, path_in_repo=path_in_repo, ptree_file=ptree_file, ptree_format=ptree_format, iout=None, msg=True):
        if(isinstance(repo, uri.RamsesSnapshot)):
            repo = repo.repo
        if(iout is None):
            filename = ptree_file
        else:
            filename = ptree_format % iout
        path = os.path.join(repo, path_in_repo, filename)
        return load(path, msg=msg)

    @staticmethod
    def save(ptree, repo, path_in_repo=path_in_repo, ptree_file=ptree_file, msg=True):
        path = os.path.join(repo, path_in_repo, ptree_file)
        return dump(ptree, path, msg=msg)

    @staticmethod
    def count_snapshots(repo, path_in_repo):
        ptree_filenames = [os.path.basename(filename) for filename in glob(os.path.join(repo, path_in_repo, 'ptree_*.pkl'))]
        parsed = [parse('ptree_{iout:d}.pkl', ptree_filename) for ptree_filename in ptree_filenames]
        iouts = [par['iout'] for par in parsed]
        iouts = np.sort(iouts)
        return iouts


    @staticmethod
    def calc_sfr(repo, path_in_repo=path_in_repo, halomaker_repo='GalaxyMaker/gal', ptree_format=ptree_format, mode='none', max_part_size=None, overwrite=True):
        # repo should be specified here since we use particle data.
        print("Starting SFR estimation for %s" % repo)

        iouts = PhantomTree.count_snapshots(repo, path_in_repo)
        max_iout = np.max(iouts)

        if (max_part_size is None):
            snap = uri.RamsesSnapshot(repo, max_iout, mode=mode)
            part_ids = HaloMaker.load(snap, halomaker_repo, galaxy=True, load_parts=True)[1]
            max_part_size = int(np.max(part_ids) * 1.1)

        part_pool = np.full(max_part_size, -1, dtype='i4')

        uri.timer.verbose = 0
        uri.verbose = 0

        psnap = uri.RamsesSnapshot(repo, iouts[0] - 1, mode=mode)
        for iout in tqdm(iouts): # never reorder this!
            nsnap = uri.RamsesSnapshot(repo, iout, mode=mode)
            ptree = PhantomTree.load(repo, path_in_repo=path_in_repo, ptree_format=ptree_format, iout=iout,
                                     msg=False)
            if (overwrite):
                ptree = drop_fields(ptree, 'msf', usemask=False)
            ptree = append_fields(ptree, 'msf', np.zeros(ptree.size), usemask=False)
            halo = ptree

            halo.sort(order='id')

            halomaker, part_ids = HaloMaker.load(nsnap, halomaker_repo, galaxy=True, load_parts=True)
            mask2 = np.isin(halomaker['id'], halo['id'])
            halomaker, part_ids = HaloMaker.cut_table(halomaker, part_ids, mask2)

            halo_idx = np.repeat(np.arange(halo.size), halo['nparts'])

            part_pool[:] = -1
            part_pool[part_ids] = halo_idx

            cpulist = nsnap.get_halos_cpulist(halo, buffer=3., n_divide=4)
            star = nsnap.get_part(target_fields=['id', 'm', 'epoch', 'cpu'], cpulist=cpulist)['star']
            newstar = star[(psnap['time'] < star['epoch']) & (nsnap['time'] >= star['epoch'])]

            ids = np.abs(newstar['id'])
            newstar_halos = part_pool[ids]

            mask3 = newstar_halos>=0
            newstar_halos = newstar_halos[mask3]
            newstar = newstar[mask3]

            newstar_mass = np.bincount(newstar_halos, weights=newstar['m', 'Msol'], minlength=halo.size)
            halo['msf'] = newstar_mass

            PhantomTree.save(halo, repo, path_in_repo=path_in_repo, ptree_file=ptree_format % iout, msg=False)
            psnap = nsnap
            gc.collect()

        uri.timer.verbose = 1
        uri.verbose = 1


    @staticmethod
    def measure_star_prop(repo, path_in_repo=path_in_repo, halomaker_repo='GalaxyMaker/gal', ptree_file=ptree_file, mode='none',
                          overwrite=True, backup_freq=30, sfr_measure_Myr=50., mass_cut_refine=2.4E-11,
                          output_file='ptree_SFR.pkl', backup_file='ptree_SFR.pkl.backup'):
        # repo should be specified here since we use particle data.
        print("Starting properties measure for %s" % repo)
        ptree_path = os.path.join(repo, path_in_repo)
        ptree = PhantomTree.load(repo, ptree_path, ptree_file=ptree_file)

        fields = ['sfr', 'sfr2', 'sfr4', 'msf', 'r90', 'r50', 'age', 'metal', 'contam', 'mbh', 'bh_offset']
        if(overwrite):
            ptree = drop_fields(ptree, ['idx', *fields], usemask=False)
        zero_double = np.zeros(ptree.size, dtype='f8')
        ptree = append_fields(ptree, ['idx', *fields],
                              [np.arange(ptree.size), *(zero_double,)*len(fields)], usemask=False)

        iouts = np.unique(ptree['timestep'])
        iouts = np.sort(iouts)
        min_iout = np.min(iouts)
        max_iout = np.max(iouts)

        snap = uri.RamsesSnapshot(repo, max_iout, mode=mode)
        part_ids = HaloMaker.load(snap, halomaker_repo, galaxy=True, load_parts=True)[1]
        max_part_size = int(np.max(part_ids) * 1.2)
        part_pool = np.full(max_part_size, -1, dtype='i4')

        uri.timer.verbose = 0
        uri.verbose = 0

        psnap = uri.RamsesSnapshot(repo, min_iout-1, mode=mode)
        for iout in tqdm(iouts):
            nsnap = uri.RamsesSnapshot(repo, iout, mode=mode)

            mask = ptree['timestep'] == iout
            gals = ptree[mask]

            gals.sort(order='hmid')

            halomaker, part_ids = HaloMaker.load(nsnap, halomaker_repo, galaxy=True, load_parts=True)

            # find missing galaxies (temporal SF clumps that was previously removed by PhantomTree)
            gal_missing = halomaker[np.isin(halomaker['id'], gals['hmid'], assume_unique=True, invert=True)]
            #halomaker, part_ids = HaloMaker.cut_table(halomaker, part_ids, mask2)

            idxs = np.arange(halomaker.size)
            halomaker_idx = np.repeat(idxs, halomaker['nparts'])

            cpulist = nsnap.get_halos_cpulist(gals, radius=1.05, radius_name='r', n_divide=5)
            nsnap.get_part(cpulist=cpulist)

            part_pool[:] = -1
            part_pool[part_ids] = halomaker_idx

            for gal in tqdm(gals):
                nsnap.set_box_halo(gal, radius=1., radius_name='r')
                nsnap.get_part(exact_box=False)
                dm = nsnap.part['dm']
                star = nsnap.part['star']
                smbh = nsnap.part['smbh']

                gal_mask = part_pool[np.abs(star['id'])] == idxs[halomaker['id'] == gal['hmid']]
                if(np.sum(gal_mask)==0):
                    continue
                gal_star = star[gal_mask]
                dists = get_distance(gal, gal_star)

                # first calculate r90 (may be updated if there's a subgalaxy
                r90 = weighted_quantile(dists, 0.9, sample_weight=gal_star['m'])

                subgals = gal_missing[get_distance(gal, gal_missing)<r90]

                if(subgals.size>0):
                    # get mask for both main galaxy and subgalaxies
                    subgal_mask = np.isin(halomaker['id'], np.concatenate([[gal['hmid']], subgals['id']]), assume_unique=True)
                    gal_mask = np.isin(part_pool[np.abs(star['id'])], idxs[subgal_mask], assume_unique=True)
                    gal_star = star[gal_mask]
                    dists = get_distance(gal, gal_star)
                    r90 = weighted_quantile(dists, 0.9, sample_weight=gal_star['m'])

                r50 = weighted_quantile(dists, 0.5, sample_weight=gal_star['m'])
                sfr = np.sum(gal_star[gal_star['age', 'Myr']<sfr_measure_Myr]['m', 'Msol']) / (sfr_measure_Myr*1E6)
                sfr2 = np.sum(gal_star[gal_star['age', 'Myr']<sfr_measure_Myr*2]['m', 'Msol']) / (sfr_measure_Myr*2E6)
                sfr4 = np.sum(gal_star[gal_star['age', 'Myr']<sfr_measure_Myr*4]['m', 'Msol']) / (sfr_measure_Myr*4E6)
                msf = np.sum(gal_star[(psnap['time'] < gal_star['epoch']) & (nsnap['time'] >= gal_star['epoch'])]['m', 'Msol'])

                age = np.average(gal_star['age', 'Gyr'], weights=gal_star['m'])
                metal = np.average(gal_star['metal'], weights=gal_star['m'])

                dm_r90 = uri.cut_halo(dm, gal, r90, use_halo_radius=False)
                contam = np.sum(dm_r90[dm_r90['m']>mass_cut_refine]['m'])/np.sum(dm_r90['m'])

                bh_r50 = uri.cut_halo(smbh, gal, r50, use_halo_radius=False)
                bh_max = bh_r50[np.argmax(bh_r50['m'])]
                mbh = bh_max['m']
                bh_offset = get_distance(gal, bh_max)

                ptree['sfr'][gal['idx']] = sfr
                ptree['sfr2'][gal['idx']] = sfr2
                ptree['sfr4'][gal['idx']] = sfr4
                ptree['msf'][gal['idx']] = msf
                ptree['r90'][gal['idx']] = r90
                ptree['r50'][gal['idx']] = r50
                ptree['m'][gal['idx']] = np.sum(gal_star['m', 'Msol'])

                ptree['vx'][gal['idx']] = np.average(gal_star['vx', 'km/s'], weights=gal_star['m'])
                ptree['vy'][gal['idx']] = np.average(gal_star['vy', 'km/s'], weights=gal_star['m'])
                ptree['vz'][gal['idx']] = np.average(gal_star['vz', 'km/s'], weights=gal_star['m'])

                ptree['nparts'][gal['idx']] = gal_star.size
                ptree['contam'][gal['idx']] = contam
                ptree['age'][gal['idx']] = age
                ptree['metal'][gal['idx']] = metal

                ptree['mbh'][gal['idx']] = mbh
                ptree['bh_offset'][gal['idx']] = bh_offset

            if(iout % backup_freq == 0):
                PhantomTree.save(ptree, repo, ptree_path, ptree_file=backup_file)

            psnap.clear()
            psnap = nsnap
            gc.collect()

        uri.timer.verbose = 1
        uri.verbose = 1
        ptree = drop_fields(ptree, 'idx', usemask=False)
        PhantomTree.save(ptree, repo, ptree_path, ptree_file=output_file)

    @staticmethod
    def measure_gas_prop(repo, path_in_repo=path_in_repo, ptree_file=ptree_file, mode='none', overwrite=True, backup_freq=30, min_radius=1., max_radius=4., radius_name='r90', iout_start=0):
        def wgas_mask(cell):
            # Torrey et al. 2012
            return np.log10(cell['T', 'K']) < 6 + 0.25 * np.log10(cell['rho', 'Msol/kpc3'] * snap['h'] ** 2 / 10 ** 10)

        def cgas_mask(cell):
            return np.log10(cell['T', 'K']) < 3.5 + 0.25 * np.log10(cell['rho', 'Msol/kpc3'] * snap['h'] ** 2 / 10 ** 10)

        def load_cell_snap(iout):
            gals = ptree[ptree['timestep'] == iout]
            snap = uri.RamsesSnapshot(repo, iout, mode=mode)
            cpulist = snap.get_halos_cpulist(gals, radius=max_radius*1.25, radius_name=radius_name)
            snap.get_cell(cpulist=cpulist)
            return gals, snap

        def measure_rgas(gas, rgal):
            mask = wgas_mask(gas)
            wgas = gas[mask]
            hgas = gas[~mask]

            wgas_dist = get_distance(gal, wgas)
            key = np.argsort(wgas_dist)
            wgas = wgas[key]
            wgas_dist = wgas_dist[key]
            wgas_cum = np.cumsum(wgas['vol'])
            min_idx = np.searchsorted(wgas_dist, min_radius * rgal)
            wgas_dist = wgas_dist[min_idx:]
            wgas_cum = wgas_cum[min_idx:]

            hgas_dist = get_distance(gal, hgas)
            key = np.argsort(hgas_dist)
            hgas = hgas[key]
            hgas_dist = hgas_dist[key]
            hgas_cum = np.cumsum(hgas['vol'])
            min_idx = np.searchsorted(hgas_dist, min_radius * rgal)
            hgas_dist = hgas_dist[min_idx:]
            hgas_cum = hgas_cum[min_idx:]

            if (hgas_dist.size == 0):  # no hot gas
                rgas = max_radius * rgal
            elif (wgas_dist.size == 0):  # no cold gas
                rgas = min_radius * rgal
            else:  # both exists
                res = np.interp(hgas_dist, wgas_dist, wgas_cum) - hgas_cum
                res_mask = res < 0

                if (np.all(res_mask)):  # if inverted at the first place
                    rgas = min_radius * rgal
                elif (np.any(res_mask)):  # if inversion happens
                    rgas = hgas_dist[res_mask][0]
                else:  # no inversion
                    rgas = max_radius * rgal

            wgas = uri.cut_halo(wgas, gal, rgas, code_unit=True)
            mask = cgas_mask(wgas)
            cgas = wgas[mask]
            wgas = wgas[~mask]
            hgas = uri.cut_halo(hgas, gal, rgas, code_unit=True)
            return cgas, wgas, hgas, rgas

        def measure_rgas_v2(gas, gal, rgal):
            # Adds outer gas
            mask = wgas_mask(gas)
            wgas = gas[mask] # includes cg also
            hgas = gas[~mask] # includes og also

            wgas_dist = get_distance(gal, wgas)
            key = np.argsort(wgas_dist)
            wgas = wgas[key]
            wgas_dist = wgas_dist[key]
            wgas_cum = np.cumsum(wgas['vol'])
            min_idx = np.searchsorted(wgas_dist, min_radius * rgal)
            wgas_dist = wgas_dist[min_idx:]
            wgas_cum = wgas_cum[min_idx:]

            hgas_dist = get_distance(gal, hgas)
            hgas = hgas[hgas_dist < min_radius * rgal]
            ogas = hgas[hgas_dist >= min_radius * rgal]

            ogas_dist = get_distance(gal, ogas)
            key = np.argsort(ogas_dist)
            ogas = ogas[key]
            ogas_dist = ogas_dist[key]
            ogas_cum = np.cumsum(ogas['vol'])
            min_idx = np.searchsorted(ogas_dist, min_radius * rgal)
            ogas_dist = ogas_dist[min_idx:]
            ogas_cum = ogas_cum[min_idx:]

            if (ogas_dist.size == 0):  # no hot gas
                rgas = max_radius * rgal
            elif (wgas_dist.size == 0):  # no cold gas
                rgas = min_radius * rgal
            else:  # both exists
                res = np.interp(ogas_dist, wgas_dist, wgas_cum) - ogas_cum
                res_mask = res < 0

                if (np.all(res_mask)):  # if inverted at the first place
                    rgas = min_radius * rgal
                elif (np.any(res_mask)):  # if inversion happens
                    rgas = ogas_dist[res_mask][0]
                else:  # no inversion
                    rgas = max_radius * rgal

            wgas = uri.cut_halo(wgas, gal, rgas, code_unit=True)
            mask = cgas_mask(wgas)
            cgas = wgas[mask]
            wgas = wgas[~mask]
            ogas = uri.cut_halo(ogas, gal, rgas*1.25, code_unit=True)

            return cgas, wgas, hgas, ogas, rgas

        def set_gas_properties(gal, gas, abbr):
            vgal = get_vector(gal, 'v') * snap.unit['km/s']
            pgal = get_vector(gal)

            gal['m%s' % abbr] = np.sum(gas['m'])
            gal['rho_%s' % abbr] = np.average(gas['rho'], weights=gas['m'])
            gal['metal_%s' % abbr] = np.average(gas['metal'], weights=gas['m'])

            vel = gas['vel'] - vgal
            pos = (gas['pos'] - pgal)
            vrad = np.sum(vel * pos, axis=-1) / rss(pos)
            gal['flux_%s' % abbr] = np.sum(vrad * gas['rho'] * gas['dx']**2)
            set_vector(gal, np.average(vel, axis=0, weights=gas['m']), abbr+'v')

        def fin_gas_propertiles(gals, abbr):
            gals['m%s' % abbr] /= snap.unit['Msol']
            gals['rho_%s' % abbr] /= snap.unit['H/cc']

            gals['%sx' % abbr] /= snap.unit['km/s']
            gals['%sy' % abbr] /= snap.unit['km/s']
            gals['%sz' % abbr] /= snap.unit['km/s']

            gals['flux_%s' % abbr] /= snap.unit['Msol/yr']

        ptree = PhantomTree.load(repo, path_in_repo=path_in_repo, ptree_file=ptree_file)

        gas_phases = ['cg', 'wg', 'hg', 'og']
        gas_names = ['m%s', '%svx', '%svy', '%svz', 'rho_%s', 'metal_%s']
        full_names = [name % abbr for abbr in gas_phases for name in gas_names]
        names = ['idx', *full_names, 'ramp', 'rgas']

        empty_double = np.zeros(ptree.size, dtype='f8')
        if(overwrite):
            ptree = drop_fields(ptree, names, usemask=True)
        if(iout_start==0):
            ptree = append_fields(ptree, names=names, data=[np.arange(ptree.size), *(empty_double.copy(),) * (len(names)-1)], usemask=False)

        leafs_id = np.unique(ptree['leaf'])
        ptree = ptree[np.isin(ptree['leaf'], leafs_id)]
        iouts = np.unique(ptree['timestep'])
        iouts = iouts[iouts>=iout_start]
        iouts.sort()

        print("Calculating RPS from iout = %d to %d..." % (iouts[0], iouts[-1]))

        uri.timer.verbose = 0
        uri.verbose = 0


        for iout in tqdm(iouts):
            gals, snap = load_cell_snap(iout)

            iterator = tqdm(gals)
            for gal in iterator:
                snap.set_box_halo(gal, max_radius, radius_name=radius_name)
                cell = snap.get_cell()
                cell = uri.cut_halo(cell, gal, max_radius, radius_name=radius_name)
                if(cell.size == 0):
                    continue

                cgas, wgas, hgas, ogas, radius = measure_rgas(cell, gal[radius_name])

                gal['rgas'] = radius

                if(cgas.size>0):
                    set_gas_properties(gal, cgas, 'cg')
                if(wgas.size>0):
                    set_gas_properties(gal, wgas, 'wg')
                if(hgas.size>0):
                    set_gas_properties(gal, hgas, 'hg')
                if(ogas.size > 0):
                    set_gas_properties(gal, ogas, 'og')

                    gal['ramp'] = ss(get_vector(gal, 'ogv')) * gal['rho_og']
            iterator.close()

            fin_gas_propertiles(gals, 'cg')
            fin_gas_propertiles(gals, 'wg')
            fin_gas_propertiles(gals, 'hg')
            fin_gas_propertiles(gals, 'og')

            gals['ramp'] /= snap.unit['Ba']

            ptree[gals['idx']] = gals
            if(iout % backup_freq == 0):
                PhantomTree.save(ptree, repo, ptree_file='ptree_RPS.pkl', msg=False)
            snap.clear()
            del cell, cgas, wgas, hgas, snap
            gc.collect()
        uri.timer.verbose = 1
        uri.verbose = 1

        ptree = drop_fields(ptree, ['idx'], usemask=False)

        PhantomTree.save(ptree, repo, ptree_file='ptree_RPS.pkl')

    #@staticmethod
    #def repair_tree(ptree):



class TreeMaker:
    dtype = [
        ('id', 'i4'), ('timestep', 'i4'), ('level', 'i4'),
        ('host', 'i4'), ('hostsub', 'i4'), ('nbsub', 'i4'), ('nextsub', 'i4'),
        ('nfat', 'i4'), ('fat1', 'i4'), ('fat2', 'i4'), ('fat3', 'i4'), ('fat4', 'i4'), ('fat5', 'i4'),
        ('nson', 'i4'), ('son1', 'i4'), ('son2', 'i4'), ('son3', 'i4'), ('son4', 'i4'), ('son5', 'i4'),
        ('aexp', 'f4'), ('age_univ', 'f4'), ('m', 'f4'), ('macc', 'f4'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'),
        ('Lx', 'f4'), ('Ly', 'f4'), ('Lz', 'f4'),
        ('r', 'f4'), ('a', 'f4'), ('b', 'f4'), ('c', 'f4'),
        ('ek', 'f4'), ('ep', 'f4'), ('et', 'f4'), ('spin', 'f4'),
        ('mfat1', 'f4'), ('mfat2', 'f4'), ('mfat3', 'f4'), ('mfat4', 'f4'), ('mfat5', 'f4'),
        ('rvir', 'f4'), ('mvir', 'f4'), ('tvir', 'f4'), ('cvel', 'f4'),
        ('rho0', 'f4'), ('rc', 'f4')]

    @staticmethod
    # boxsize: comoving length of the box in Mpc
    def load(snap, path_in_repo=None, galaxy=False, dp_ini=False):
        repo = snap.repo
        if(snap.mode=='yzics'):
            dp_ini=False
        elif(snap.mode=='nh'):
            dp_ini=True
        if (galaxy):
            if(path_in_repo is None):
                path_in_repo = 'GalaxyMaker/gal/tree.dat'
            path = os.path.join(repo, path_in_repo)
        else:
            if(path_in_repo is None):
                path_in_repo = 'halo/DM/tree.dat'
            path = os.path.join(repo, path_in_repo)
        dtype = TreeMaker.dtype
        if(not os.path.exists(path)):
            raise FileNotFoundError('Error: Tree file not found in path: %s' % path)

        print('Reading %s... ' % path)
        timer.start()
        readh.read_single_tree(path, galaxy, dp_ini=dp_ini)
        print('Took %.3fs' % timer.time())

        print('Building table for particles... ', end='')
        timer.start()
        array = fromarrays([*readh.integer_table.T, *readh.real_table.T], dtype=dtype)
        print('Took %.3fs' % timer.time())

        return HaloMaker.unit_conversion(array, snap)


class Rockstar:
    dtype_def = [
        ('id', 'i4'), ('num_p', 'i4'), ('mvir', 'f8'), ('mbound_vir', 'f8'), ('rvir', 'f8'),
        ('vmax', 'f8'), ('rvmax', 'f8'), ('vrms', 'f8'),
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'),
        ('Jx', 'f8'), ('Jy', 'f8'), ('Jz', 'f8'), ('E', 'f8'), ('Spin', 'f8'), ('PosUncertainty', 'f8'), ('VelUncertainty', 'f8'),
        ('bulk_vx', 'f8'), ('bulk_vy', 'f8'), ('bulk_vz', 'f8'), ('BulkVelUnc', 'f8'), ('n_core', 'i4'),
        ('m200b', 'f8'), ('m200c', 'f8'), ('m500c', 'f8'), ('m2500c', 'f8'), ('Xoff', 'f8'), ('Voff', 'f8'),
        ('spin_bullock', 'f8'), ('b_to_a', 'f8'), ('c_to_a', 'f8'), ('A[x]', 'f8'), ('A[y]', 'f8'), ('A[z]', 'f8'),
        ('b_to_a(500c)', 'f8'), ('c_to_a(500c)', 'f8'), ('A[x](500c)', 'f8'), ('A[y](500c)', 'f8'), ('A[z](500c)', 'f8'),
        ('Rs', 'f8'), ('Rs_Klypin', 'f8'), ('T/|U|', 'f8'), ('M_pe_Behroozi', 'f8'), ('M_pe_Diemer', 'f8'), ('Halfmass_Radius', 'f8'),
        ('idx', 'i4'), ('i_so', 'i4'), ('i_ph', 'i4'), ('num_cp', 'i4'), ('mmetric', 'f8'),
    ]

    dtype_amr = [
        ('id', 'i4'), ('num_p', 'i4'), ('mvir', 'f8'), ('mbound_vir', 'f8'), ('rvir', 'f8'),
        ('vmax', 'f8'), ('rvmax', 'f8'), ('vrms', 'f8'),
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'),
        ('Jx', 'f8'), ('Jy', 'f8'), ('Jz', 'f8'), ('E', 'f8'), ('Spin', 'f8'), ('PosUncertainty', 'f8'), ('VelUncertainty', 'f8'),
        ('bulk_vx', 'f8'), ('bulk_vy', 'f8'), ('bulk_vz', 'f8'), ('BulkVelUnc', 'f8'), ('n_core', 'i4'),
        ('m200b', 'f8'), ('m200c', 'f8'), ('m500c', 'f8'), ('m2500c', 'f8'), ('Xoff', 'f8'), ('Voff', 'f8'),
        ('spin_bullock', 'f8'), ('b_to_a', 'f8'), ('c_to_a', 'f8'), ('A[x]', 'f8'), ('A[y]', 'f8'), ('A[z]', 'f8'),
        ('Rs', 'f8'), ('Rs_Klypin', 'f8'), ('T/|U|', 'f8'), ('idx', 'f8'),
        ('i_ph', 'f8'), ('num_cp', 'f8'), ('mmetric', 'f8'),
    ]

    dtype_gal = [
        ('id', 'i4'), ('num_p', 'i4'), ('mvir', 'f8'), ('mbound_vir', 'f8'), ('rvir', 'f8'),
        ('vmax', 'f8'), ('rvmax', 'f8'), ('vrms', 'f8'),
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'),
        ('Jx', 'f8'), ('Jy', 'f8'), ('Jz', 'f8'), ('E', 'f8'), ('Spin', 'f8'), ('PosUncertainty', 'f8'), ('VelUncertainty', 'f8'),
        ('bulk_vx', 'f8'), ('bulk_vy', 'f8'),   ('bulk_vz', 'f8'), ('BulkVelUnc', 'f8'), ('n_core', 'i4'),
        ('m200b', 'f8'), ('m200c', 'f8'), ('m500c', 'f8'), ('m2500c', 'f8'), ('Xoff', 'f8'), ('Voff', 'f8'),
        ('spin_bullock', 'f8'), ('b_to_a', 'f8'), ('c_to_a', 'f8'), ('A[x]', 'f8'), ('A[y]', 'f8'), ('A[z]', 'f8'),
        ('b_to_a(500c)', 'f8'), ('c_to_a(500c)', 'f8'), ('A[x](500c)', 'f8'), ('A[y](500c)', 'f8'), ('A[z](500c)', 'f8'),
        ('Rs', 'f8'), ('Rs_Klypin', 'f8'), ('T/|U|', 'f8'), ('M_pe_Behroozi', 'f8'), ('M_pe_Diemer', 'f8'),
        ('Type', 'i4'), ('SM', 'f8'), ('Gas', 'f8'), ('BH', 'f8'),
        ('idx', 'f8'), ('i_ph', 'f8'), ('num_cp', 'f8'), ('mmetric', 'f8'),
    ]

    dtype_lin = [
        ('id', 'i4'), ('desc', 'i4'), ('mvir', 'f8'), ('vmax', 'f8'), ('vrms', 'f8'), ('rvir', 'f8'), ('Rs', 'f8'),
        ('num_p', 'i4'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'),
        ('Jx', 'f8'), ('Jy', 'f8'), ('Jz', 'f8'),
        ('mbound_vir', 'f8'), ('rvmax', 'f8'), ('E', 'f8'), ('Spin', 'f8'), ('PosUncertainty', 'f8'),
        ('VelUncertainty', 'f8'), ('bulk_vx', 'f8'), ('bulk_vy', 'f8'), ('bulk_vz', 'f8'), ('BulkVelUnc', 'f8'), ('n_core', 'i4'),
        ('m200b', 'f8'), ('m200c', 'f8'), ('m500c', 'f8'), ('m2500c', 'f8'), ('Xoff', 'f8'), ('Voff', 'f8'),
        ('spin_bullock', 'f8'), ('b_to_a', 'f8'), ('c_to_a', 'f8'), ('A[x]', 'f8'), ('A[y]', 'f8'), ('A[z]', 'f8'),
        ('Rs_Klypin', 'f8'), ('T/|U|', 'f8'), ('idx', 'f8'),
        ('i_ph', 'f8'), ('num_cp', 'f8'), ('mmetric', 'f8'),
    ]

    bin_nfields = 33
    bin_offset_npart = 25
    dtype = dtype_def

    @staticmethod
    def unit_conversion(array, snap):
        # all units to physical except distances; which are in code unit
        boxsize = snap.params['boxsize']
        h = snap.params['h']

        array['mvir'] /= h
        array['mbound_vir'] /= h
        array['rvir'] /= boxsize * 1000

        array['x'] /= boxsize
        array['y'] /= boxsize
        array['z'] /= boxsize

        array['Jx'] /= h**2
        array['Jy'] /= h**2
        array['Jy'] /= h**2

        array['m200b'] /= h
        array['m200c'] /= h
        array['m2500c'] /= h

        return array

    @staticmethod
    def load(snap, path_in_repo='rst', filefmt='halos_%d.list', unit_conversion=True):
        repo = snap.repo
        iout = snap.iout
        dtype = Rockstar.dtype
        table = np.genfromtxt(os.path.join(repo, path_in_repo, filefmt % iout), dtype=dtype)
        if(unit_conversion):
            return Rockstar.unit_conversion(table, snap)
        else:
            return table

    @staticmethod
    def load_header(snap, path_in_repo='rst', filefmt='halos_%d.list'):
        repo = snap.repo
        iout = snap.iout
        with open(os.path.join(repo, path_in_repo, filefmt % iout), 'r') as fi:
            lines = []
            for ln in fi:
                if ln.startswith("#"):
                    lines.append(ln)
        return "".join(lines[2:])


    @staticmethod
    def load_parts(snap, path_in_repo='rst', filefmt='halos_%d.bin'):
        repo = snap.repo
        iout = snap.iout
        with open(os.path.join(repo, path_in_repo, filefmt % iout), mode='rb') as fb:
            fb.seek(8*8) # skip some headers
            nhalo = np.fromfile(fb, '<i8', 1)[0]

            fb.seek(256 + Rockstar.bin_nfields*8*nhalo)
            part_ids = np.fromfile(fb, '<i8')
            part_ids = np.array(part_ids, dtype='i4')
            halo_ids = np.empty(part_ids.shape, dtype='i4')
            nparts_list = []

            now = 0
            for ihalo in np.arange(nhalo):
                fb.seek(256 + Rockstar.bin_nfields * 8 * ihalo)
                halo_id = np.fromfile(fb, '<i8', 1).astype('i4')[0] # should be same with ihalo, but just for insurance.
                fb.seek(256 + Rockstar.bin_nfields * 8 * ihalo + Rockstar.bin_offset_npart*8)  # skip all the way to nparts
                nparts = np.fromfile(fb, '<i8', 1).astype('i4')[0]
                halo_ids[now:now+nparts] = halo_id
                now+=nparts
                nparts_list.append(nparts)

            output = np.rec.fromarrays([part_ids, halo_ids], names=['pid', 'hid'])
            return output, np.array(nparts_list)


    @staticmethod
    def load_binary(file):
        with open(file, mode='rb') as fb:
            fb.seek(8*8) # skip some headers
            nhalo = np.fromfile(fb, '<i8', 1)[0]

            fb.seek(256 + Rockstar.bin_nfields*8*nhalo)
            part_ids = np.fromfile(fb, '<i8')
            part_ids = np.array(part_ids, dtype='i4')
            halo_ids = np.empty(part_ids.shape, dtype='i4')
            nparts_list = []

            now = 0
            for ihalo in np.arange(nhalo):
                fb.seek(256 + Rockstar.bin_nfields * 8 * ihalo)
                halo_id = np.fromfile(fb, '<i8', 1).astype('i4')[0] # should be same with ihalo, but just for insurance.
                fb.seek(256 + Rockstar.bin_nfields * 8 * ihalo + Rockstar.bin_offset_npart*8)  # skip all the way to nparts
                nparts = np.fromfile(fb, '<i8', 1).astype('i4')[0]
                halo_ids[now:now+nparts] = halo_id
                now+=nparts
                nparts_list.append(nparts)

            output = np.rec.fromarrays([part_ids, halo_ids], names=['pid', 'hid'])
            return output, np.array(nparts_list)


    @staticmethod
    def cut_part(nexts, next_part, next_np_list, npart_min=0):
        mask = nexts['num_p'] >= npart_min
        nexts = nexts[mask]
        next_np_list = next_np_list[mask]
        next_part = next_part[np.isin(next_part['hid'], nexts['id'])]
        return nexts, next_part, next_np_list

    @staticmethod
    def connect_desc_fast(snap, path_in_repo='rst', fileini='halos', end_snap=1, npart_min=0, max_part_size=None):
        print('Starting snap linking for %s' % snap.repo)

        next_part, next_np_list = Rockstar.load_parts(snap, path_in_repo, filefmt=fileini + '_%d.bin')
        #next_list = np.split(next_part['pid'], np.cumsum(next_np_list))
        nexts = Rockstar.load(snap, path_in_repo, filefmt=fileini+'_%d.list', unit_conversion=False)
        next_header = Rockstar.load_header(snap, path_in_repo, filefmt=fileini+'_%d.list')

        if (max_part_size is None):
            max_part_size = int(np.max(next_part['pid']) * 1.1)

        part_pool = np.full((2, max_part_size), -1, 'i4')

        next_part = np.sort(next_part, order='pid')
        part_pool[1, next_part['pid']] = next_part['hid']

        desc_hids = np.full(nexts.shape, -1)
        f_pass = np.zeros(nexts.shape)
        f_cont = np.zeros(nexts.shape)

        nexts = Rockstar.convert_to_mergertree(nexts, desc_hids, f_pass, f_cont)
        Rockstar.save(nexts, snap, path_in_repo, header=next_header)

        #nexts, next_part, next_np_list = Rockstar.cut_part(nexts, next_part, next_np_list, npart_min=npart_min)
        snap_iouts = np.arange(snap.iout, end_snap-1, -1)
        uri.verbose = 0

        iterator = tqdm(snap_iouts)
        for iout in iterator:
            #print('Linking snapshot %d - %d...' % (snap.iout, snap.iout-1))
            try:
                snap_former = uri.RamsesSnapshot(snap.snap_path, iout - 1, snap.mode, full_path=True)
                snap_former.repo = snap.repo
                prog_part, prog_np_list = Rockstar.load_parts(snap_former, path_in_repo, filefmt=fileini + '_%d.bin')

                progs = Rockstar.load(snap_former, path_in_repo, filefmt=fileini+'_%d.list', unit_conversion=False)
                prog_header = Rockstar.load_header(snap, path_in_repo, filefmt=fileini+'_%d.list')
            except(FileNotFoundError):
                iterator.close()
                break

            try:
                len(progs)
            except:
                iterator.close()
                break

            #prog_list = np.split(next_part['pid'], np.cumsum(prog_np_list))
            prog_part = np.sort(prog_part, order='pid')

            part_pool[0, prog_part['pid']] = prog_part['hid']
            #progs, prog_part, prog_np_list = Rockstar.cut_part(progs, prog_part, prog_np_list, npart_min=npart_min)

            bins = [np.arange(progs.size+1)-0.5, np.arange(nexts.size+1)-0.5]

            hist = np.histogram2d(part_pool[0], part_pool[1], bins=bins)[0]
            desc_hids = np.argmax(hist, axis=1)

            counts = np.max(hist, axis=1)
            desc_counts = next_np_list[desc_hids]
            prog_counts = prog_np_list

            f_pass = counts/prog_counts
            f_cont = counts/desc_counts

            progs = Rockstar.convert_to_mergertree(progs, desc_hids, f_pass, f_cont)
            Rockstar.save(progs, snap_former, path_in_repo, header=prog_header)
            next_part, next_np_list = prog_part, prog_np_list
            part_pool[1] = part_pool[0]
            part_pool[0] = -1

    @staticmethod
    def connect_desc(snap, path_in_repo='rst'):
        print('Starting snap linking for %s' % snap.repo)

        next_part, next_np_list = Rockstar.load_parts(snap, path_in_repo)
        # next_list = np.split(next_part['pid'], np.cumsum(next_np_list))
        nexts = Rockstar.load(snap, path_in_repo)
        next_header = Rockstar.load_header(snap, path_in_repo)
        next_part = np.sort(next_part, order='pid')

        desc_hids = np.full(nexts.shape, -1)
        f_pass = np.zeros(nexts.shape)
        f_cont = np.zeros(nexts.shape)

        nexts = Rockstar.convert_to_mergertree(nexts, desc_hids, f_pass, f_cont)
        Rockstar.save(nexts, snap, path_in_repo, header=next_header)

        # nexts, next_part, next_np_list = Rockstar.cut_part(nexts, next_part, next_np_list, npart_min=npart_min)

        while (snap.iout > 1):
            print('Linking snapshot %d - %d...' % (snap.iout, snap.iout - 1))
            try:
                snap_former = uri.RamsesSnapshot(snap.repo, snap.iout - 1, snap.mode)
                prog_part, prog_np_list = Rockstar.load_parts(snap_former, path_in_repo)
                prog_header = Rockstar.load_header(snap, path_in_repo)
                progs = Rockstar.load(snap_former, path_in_repo)
            except(FileNotFoundError):
                break

            prog_list = np.split(next_part['pid'], np.cumsum(prog_np_list))
            prog_part = np.sort(prog_part, order='pid')
            # progs, prog_part, prog_np_list = Rockstar.cut_part(progs, prog_part, prog_np_list, npart_min=npart_min)

            desc_hids = []
            f_pass = []
            f_cont = []

            for prog, prog_pid in zip(tqdm(progs), prog_list):
                mask = np.isin(next_part['pid'], prog_pid, True)
                mode_result = mode(next_part['hid'][mask])
                next_part = next_part[~mask]
                mode_id = mode_result.mode
                if (mode_id.size > 0):
                    desc_hid = mode_id[0]
                    count = mode_result.count[0]
                else:
                    desc_hid = -1
                    count = 0
                desc_hids.append(desc_hid)
                f_pass.append(count / prog_pid.size)
                f_cont.append(count / next_np_list[desc_hid])

            progs = Rockstar.convert_to_mergertree(progs, desc_hids, f_pass, f_cont)
            Rockstar.save(progs, snap_former, path_in_repo, header=prog_header)
            next_part, next_np_list, next_list = prog_part, prog_np_list, prog_list

    @staticmethod
    def convert_to_mergertree(progs, desc_hids, f_pass, f_cont):
        progs = append_fields(progs, ['desc', 'f_pass', 'f_cont', 'num_p0'], [desc_hids, f_pass, f_cont, progs['num_p']], usemask=False)
        names = progs.dtype.names

        # Change name order to match with Rockstar parallel mode output
        names_front = ['id', 'desc', 'mvir', 'vmax', 'vrms', 'rvir', 'Rs', 'num_p',
                       'x', 'y', 'z', 'vx', 'vy', 'vz', 'Jx', 'Jy', 'Jz', 'Spin']
        names_back = np.array(names)[np.isin(names, names_front, assume_unique=True, invert=True)]
        progs = progs[list(names_front)+list(names_back)]
        progs = fix_out_of_order_fields(progs)

        return progs


    @staticmethod
    def save(halos, snap, path_in_repo='rst', filefmt='out_%d.list', header=""):
        repo = snap.repo
        path = os.path.join(repo, path_in_repo, filefmt % snap.iout)
        first_line = "# %s\n" % " ".join(halos.dtype.names)
        second_line = "#a = %.6f\n" % snap.aexp
        header = first_line + second_line + header
        np.savetxt(path, halos, header=header, comments="", fmt=dtype_to_format(halos.dtype))


class ConsistentTrees:
    @staticmethod
    def unit_conversion(array, snap):
        # all units to physical except distances; which are in code unit
        boxsize = snap.params['boxsize']
        h = snap.params['h']

        array['mvir'] /= h
        array['rvir'] /= boxsize * 1000

        array['x'] /= boxsize
        array['y'] /= boxsize
        array['z'] /= boxsize

        array['Jx'] /= h**2
        array['Jy'] /= h**2
        array['Jy'] /= h**2

        names = array.dtype.names
        m_names = ['mbound_vir', 'm200b', 'm200c', 'm2500c']
        for m_name in m_names:
            if(m_name in names):
                array[m_name] /= h

        return array

    @staticmethod
    def load(snap, path_in_repo='rst/trees', filename='tree_0_0_0.dat', unit_conversion=True, dtype=None):
        ext = os.path.splitext(filename)[-1]
        path = os.path.join(snap.repo, path_in_repo, filename)
        if(ext == '.dat'):
            if(dtype is None):
                dtype = dtype_ct
            table = np.genfromtxt(path, skip_header=37, dtype=dtype)
        elif(ext == '.pkl'):
            table = load(path)
            unit_conversion = False

        if (unit_conversion):
            return ConsistentTrees.unit_conversion(table, snap)
        else:
            return table

    @staticmethod
    def process_branches(table):
        print("Processing branches for ConsistentTrees...")
        sort_key = np.argsort(table, order='id')
        table = table[sort_key]
        keys = np.arange(table.size)[sort_key]

        brch_id = np.full(table.shape, -1, dtype='i4')
        table = append_fields(table, ['brch_id'], [brch_id], usemask=False)

        leaf_idxs = np.where(table['num_prog'] == 0)[0]
        for leaf_idx in tqdm(leaf_idxs):
            leaf = table[leaf_idx]
            leaf['brch_id'] = leaf['id']
            halo = leaf
            while(halo['mmp'] == 1 and halo['desc_id'] != -1):
                desc_id = halo['desc_id']
                halo = table[np.searchsorted(table['id'], desc_id)]
                if(halo['id'] == desc_id):
                    halo['brch_id'] = leaf['id']
                else:
                    print("Warning: descendant id %d not found in the tree" % desc_id)
                    break
        return table[keys]

    def process_tree(table):
        print("Processing basic informations for ConsistentTrees...")
        sort_key = np.argsort(table, order='id')
        table = table[sort_key]
        keys = np.arange(table.size)[sort_key]

        brch_id = np.full(table.shape, -1, dtype='i4')
        table = append_fields(table, ['brch_id'], [brch_id], usemask=False)

        leaf_idxs = np.where(table['num_prog'] == 0)[0]
        for leaf_idx in tqdm(leaf_idxs):
            leaf = table[leaf_idx]
            leaf['brch_id'] = leaf['id']
            halo = leaf
            while(halo['mmp'] == 1 and halo['desc_id'] != -1):
                desc_id = halo['desc_id']
                halo = table[np.searchsorted(table['id'], desc_id)]
                if(halo['id'] == desc_id):
                    halo['brch_id'] = leaf['id']
                else:
                    print("Warning: descendant id %d not found in the tree" % desc_id)
                    break
        return table[keys]


class Rockstar_Galaxy:

    @staticmethod
    def load(snap, iout, path_in_repo=None):
        repo = snap.repo
        dtype = Rockstar_Galaxy.dtype
        table = np.genfromtxt(os.path.join(repo, path_in_repo, 'out_%d.list' % iout), dtype)

        return Rockstar.unit_conversion(table, snap)

