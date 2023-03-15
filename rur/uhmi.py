import os
from numpy.core.records import fromarrays
import numpy as np
from rur.utool import Timer, get_vector, dump, load, pairing, get_distance, rss, ss,\
    set_vector, discrete_hist2d, weighted_quantile, expand_shape
from rur.readhtm import readhtm as readh
from rur import uri
from rur.config import Table, tqdm, default_path_in_repo
from scipy.stats import mode
from numpy.lib.recfunctions import append_fields, drop_fields, merge_arrays
import gc
import string
from multiprocessing import Process, Queue
from time import sleep

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
        array['Lx'] *= mass_unit
        array['Ly'] *= mass_unit
        array['Lz'] *= mass_unit

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
    halo_dtype_dp = [
        ('nparts', 'i4'), ('id', 'i4'), ('timestep', 'i4'), ('level', 'i4'),
        ('host', 'i4'), ('hostsub', 'i4'), ('nbsub', 'i4'), ('nextsub', 'i4'),
        ('aexp', 'f4'), ('m', 'f4'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'),
        ('Lx', 'f4'), ('Ly', 'f4'), ('Lz', 'f4'),
        ('r', 'f4'), ('a', 'f4'), ('b', 'f4'), ('c', 'f4'),
        ('ek', 'f4'), ('ep', 'f4'), ('et', 'f4'), ('spin', 'f4'),('sigma', 'f4'),
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
    def load(snap, path_in_repo=None, galaxy=False, full_path=None, load_parts=False, double_precision=None, copy_part_id=True):
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
            if(snap.mode=='yzics'):
                double_precision=True
        if (galaxy):
            if(path_in_repo is None):
                path_in_repo = default_path_in_repo['GalaxyMaker']
            if(not double_precision):
                dtype = HaloMaker.galaxy_dtype
            else:
                dtype = HaloMaker.galaxy_dtype_dp
        else:
            if(path_in_repo is None):
                path_in_repo = default_path_in_repo['HaloMaker']
            if(not double_precision):
                dtype = HaloMaker.halo_dtype
            else:
                dtype = HaloMaker.halo_dtype_dp
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
            part_ids = readh.part_ids
            if(copy_part_id):
                part_ids = part_ids.copy()
            return array, part_ids
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

    @staticmethod
    def get_member_star(snap, table, part_ids, hmid):
        halo = table[table['id'] == hmid][0]
        snap.set_box_halo(halo, radius_name='r', use_halo_radius=True)
        snap.get_part()
        star = snap.part['star']
    
    @staticmethod
    def read_member_star(snap, hmid, nchem=0, galaxy=False, path_in_repo=None, full_path=None):
        if(full_path is None):
            if(path_in_repo is None):
                if (galaxy):        
                    path_in_repo = default_path_in_repo['GalaxyMaker']
                    temp = "GAL"
                else:
                    path_in_repo = default_path_in_repo['HaloMaker']
                    temp = "HAL"
            path = os.path.join(snap.repo, path_in_repo, f"{temp}_{snap.iout:05d}")
        else:
            path = full_path # ex: /storage8/.../galaxy/GAL_00001
            if not ("AL" in path):
                if (galaxy):        
                    temp = "GAL"
                else:
                    temp = "HAL"
                path = os.path.join(path, f"{temp}_{snap.iout:05d}")
        readh.read_one(path, galaxy, hmid, nchem)

        dtype = [('id', 'i4'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'), ('m', 'f8')]
        if(galaxy):
            dtype = dtype+[('epoch', 'f8'), ('metal', 'f8')]
            if nchem>0:
                raise ValueError("Currently non-zero `nchem` is not supported!")
        array = fromarrays([*readh.integer_table, *readh.real_table_dp], dtype=dtype)
        array['m'] *= 1e11
        boxsize_physical = snap['boxsize_physical']
        array['x'] = array['x'] / boxsize_physical + 0.5
        array['y'] = array['y'] / boxsize_physical + 0.5
        array['z'] = array['z'] / boxsize_physical + 0.5
        readh.close()

        return uri.Particle(array, snap)

    from rur.uri import RamsesSnapshot
    @staticmethod
    def findhalo_NH_pair(ihaloid:int, iout:int, jout:int=None, ipart_ids:np.ndarray=None, isnap:RamsesSnapshot=None, jsnap:RamsesSnapshot=None, jtable:np.recarray=None, jpart_ids:np.ndarray=None, mcut:float=0.01, base='nh', verbose=True, **snapkwargs:dict):
        """
        This function find NH2 pair halo of given NH halo (or, vice-versa).\n
        Scores are calculated based on 'maximum refined DM particles'  

        Parameters
        ----------
        ihaloid :   int
            ID of given halo
        iout :      int
            iout of given halo
        base =      str
            Simulation mode of given halo (one of `nh` or `nh2`). Default is `nh`
        mcut =      float
            Mass ratio cut. Default is 0.01
        
        Optional
        --------
        * You may want to iterate this function for more than 1 halo.
        * Then, I recommend you to fill below kwargs.
        
        jout :      int
            iout of pair candidates
        iparts_ids: np.ndarray or Iterable
            IDs of member particles of given halo
        isnap:      uri.RamsesSnapshot
            Snapshot object of given halo
        jsnap:      uri.RamsesSnapshot
            Snapshot object of pair candidates
        jtable:     np.recarray
            HaloMaker catalogue of pair candidates
        jpart_ids:  np.ndarray or Iterable
            IDs of member particles of pair candidates (should be aligned with `jtable`)
        verbose:    bool
            Print messages if True. Default is True.
        snapkwargs: dict
            Any other kwargs for `uri.RamsesSnapshot.__init__()`

        Returns
        -------
        jout:       int
            iout of candidate pair halos
        halo_ids:   np.ndarray with int
            ID of candidate pair halos (sorted by occurence)
        occurence:  np.ndarray with float
            Fraction of matched particles

        """
        if not verbose:
            iverbose = int(uri.timer.verbose)
            uri.timer.verbose=0
        nhrepo = "/storage6/NewHorizon"
        nh2repo = "/storage7/NH2"
        if base == 'nh':
            irepo = nhrepo; jrepo = nh2repo; iname="NH"; jname="NH2"
        elif base == 'nh2':
            irepo = nh2repo; jrepo = nhrepo; iname="NH2"; jname="NH"
        else:
            raise ValueError("You should choose one of [`nh`, `nh2`]!")
        avail1 = np.loadtxt(f"{irepo}/list_iout_avail.txt", usecols=[0,1], dtype=[("iout",int), ("aexp", float)])
        avail2 = np.loadtxt(f"{jrepo}/list_iout_avail.txt", usecols=[0,1], dtype=[("iout",int), ("aexp", float)])
        aexp1 = avail1[avail1['iout'] == iout]['aexp'][0]
        if jout is None:
            if verbose: print(f"(No specified iout of {jname} ... Auto finding ...)")
            arg = np.argmin(np.abs(avail2['aexp'] - aexp1))
            jout = avail2['iout'][arg]
        aexp2 = avail2[avail2['iout'] == jout]['aexp'][0]
        if verbose: print(f"Given: {iname} halo({ihaloid}) at {iout}(a={aexp1:.4f}) <-> Find: {jname} halo(?) at {jout}(a={aexp2:.4f})")

        if ipart_ids is None:
            if verbose: print("(No specified particle IDs ... Auto finding ...)")
            if isnap is None:
                isnap = uri.RamsesSnapshot(irepo, iout)
            itable, hmpids = HaloMaker.load(isnap, galaxy=False, load_parts=True, **snapkwargs)
            if len(itable) == 0:
                if verbose: print("(Find tree bricks in `halo_iap` ...)")
                itable, hmpids = HaloMaker.load(isnap, galaxy=False, load_parts=True, path_in_repo="halo_iap",**snapkwargs)
            nparts = itable['nparts']
            cparts = np.insert(np.cumsum(nparts), 0, 0)
            ipart_ids = hmpids[ cparts[ihaloid-1] : cparts[ihaloid] ]

        if jsnap is None:
            jsnap = uri.RamsesSnapshot(jrepo, jout)

        if (jtable is None) or (jpart_ids is None):
            jtable, jpart_ids = HaloMaker.load(jsnap, galaxy=False, load_parts=True, **snapkwargs)
            if len(jtable) == 0:
                if verbose: print("(Find tree bricks in `halo_iap` ...)")
                jtable, jpart_ids = HaloMaker.load(jsnap, galaxy=False, load_parts=True, path_in_repo="halo_iap",**snapkwargs)
        matchid = load(f"{irepo}/DMID_{iname}_to_{jname}.pickle", format='pkl', msg=False)[1][ipart_ids-1]
        matchid = matchid[(matchid > 0) & np.isin(matchid, jpart_ids)]
        if len(matchid) == 0:
            if verbose: print(f"Warning! This {iname} halo does not have matched high-res particles in {jname}!\n\t--> return 0, 0")
            return 0, 0
        halo_ids = np.repeat(jtable['id'], jtable['nparts'])
        arg = np.isin(jpart_ids, matchid)
        halo_ids, occurence = np.unique(halo_ids[arg], return_counts=True)
        occurence = occurence/len(ipart_ids)
        mask = occurence > mcut
        arg = np.argsort(occurence[mask])
        if verbose: print(f"{len(mask[mask])} halos are found!")
        else:
            uri.timer.verbose = iverbose
        return jout, halo_ids[mask][arg][::-1], occurence[mask][arg][::-1]
        

class PhantomTree:
    path_in_repo = 'ptree'
    ptree_file = 'ptree.pkl'
    ptree_file_format = 'ptree_%05d.pkl'
    desc_name = 'desc'
    pass_name = 'pass'
    full_path_ptree = None
    full_path_halomaker = None

    @staticmethod
    def from_halomaker(snap, lookup, rankup=1, path_in_repo=path_in_repo, max_part_size=None,
                       ptree_file_format=ptree_file_format, nparts_min=None,
                       part_array_buffer=1.1, skip_jumps=False, start_on_middle=False, 
                       path_in_repo_halomaker=default_path_in_repo['GalaxyMaker'], full_path_ptree=None,
                       full_path_halomaker=None,**kwargs):
        print('Building PhantomTree from HaloMaker data in %s' % snap.repo)
        max_iout = snap.iout
        uri.timer.verbose = 0
        snap_iouts = np.arange(snap.iout, 0, -1)

        if (max_part_size is None):
            halo, part_ids = HaloMaker.load(snap, path_in_repo=path_in_repo_halomaker,full_path=full_path_halomaker, load_parts=True, **kwargs)
            max_part_size = int(np.max(part_ids) * part_array_buffer)

        part_pool = np.full((lookup, max_part_size), -1, dtype='i4')
        sizes = np.zeros(lookup, dtype='i4')
        halo_ids = []
        buffer = 0

        iterator = tqdm(snap_iouts, unit='snapshot')
        for iout in iterator:
            try:
                snap = snap.switch_iout(iout)
            except FileNotFoundError:
                if(skip_jumps):
                    continue
                else:
                    iterator.close()
                    break

            halo, part_ids = HaloMaker.load(snap, load_parts=True, path_in_repo=path_in_repo_halomaker, full_path=full_path_halomaker,**kwargs)
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

            desc_ids = np.empty(shape=((lookup-1)*rankup, halo.size), dtype='i4')
            npass = np.empty(shape=((lookup-1)*rankup, halo.size), dtype='i4')

            # loop over lookup arrays (i.e. look foward snapshots)
            for ilook in np.arange(1, lookup):
                rank_range = slice((ilook - 1) * rankup, ilook * rankup)
                # record descendent halos and number of particles sent, sorted by rank
                if(ilook<=buffer):
                    desc_idx, npass_arr = PhantomTree.find_desc(
                        part_pool[np.array([0, ilook])], prog_n=sizes[0], next_n=sizes[ilook], rankup=rankup)
                    desc_ids[rank_range] = halo_ids[ilook-1][desc_idx]
                    desc_ids[rank_range][desc_idx==-1] = -1
                    npass[rank_range] = npass_arr
                else:
                    desc_ids[rank_range] = -1
                    npass[rank_range] = 0
            buffer += 1

            halo_ids = [halo['id']] + halo_ids
            if(len(halo_ids)>lookup-1):
                halo_ids = halo_ids[:lookup-1]

            if(start_on_middle and sizes[-1]==0):
                print("Skipping output of iout = %d..." % iout)
                continue

            tree_dtype = np.dtype([('desc', 'i4', (lookup-1, rankup)), ('npass', 'i4', (lookup-1, rankup))])
            tree_data = np.full(halo.size, fill_value=-2, dtype=tree_dtype)

            desc_ids = np.reshape(desc_ids, (lookup-1, rankup, halo.size))
            desc_ids = np.rollaxis(desc_ids, -1, 0)

            npass = np.reshape(npass, (lookup-1, rankup, halo.size))
            npass = np.rollaxis(npass, -1, 0)

            tree_data['desc'] = desc_ids
            tree_data['npass'] = npass

            # merge generated tree data
            halo = merge_arrays([halo, tree_data], fill_value=-2, flatten=True, usemask=False)

            if full_path_ptree is None:
                path = os.path.join(snap.repo, path_in_repo, ptree_file_format % iout)
            else:
                path = os.path.join(full_path_ptree, ptree_file_format % iout)
            dump(halo, path, msg=False)
        uri.timer.verbose = 1


    @staticmethod
    def find_desc(hid_arr, prog_n=None, next_n=None, rankup=1):
        """
        :param hid_arr: 2 * nparts array that specifies idx of halos for each particle.
        :param prog_n: number of progenitors.
        :param next_n: number of next halos.
        :return: list of descendent idx of each progenitor, number of particles passed
        """
        hid_arr = hid_arr[:, (hid_arr[0] != -1) & (hid_arr[1] != -1)]
        if (prog_n is None):
            prog_n = np.max(hid_arr[0]) + 1
        if (next_n is None):
            next_n = np.max(hid_arr[1]) + 1
        hist = discrete_hist2d((prog_n, next_n), hid_arr, use_long=True)

        desc_idx = np.argpartition(hist, -(np.arange(rankup) + 1), axis=1)[:, -1:-(rankup + 1):-1].T
        npass = np.zeros(desc_idx.shape, dtype='i4')
        for irank in np.arange(rankup):
            npass[irank] = hist[(np.arange(prog_n), desc_idx[irank])]
        desc_idx[npass == 0] = -1

        return desc_idx, npass

    @staticmethod
    def merge_ptree(repo, iout_max, full_path=None, path_in_repo=path_in_repo, ptree_file=ptree_file, ptree_file_format=ptree_file_format, skip_jumps=False, dtype_id='i8'):
        dirpath = os.path.join(repo, path_in_repo) if full_path is None else full_path
        iout = iout_max
        ptree = []

        while(True):
            path = os.path.join(dirpath, ptree_file_format % iout)
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
        ptree = PhantomTree.set_pairing_id(ptree, dtype_id=dtype_id)

        dump(ptree, os.path.join(dirpath, ptree_file))

    @staticmethod
    def set_pairing_id(ptree, save_hmid=True, dtype_id='i8'):
        #if(np.max(ptree['timestep'])*np.max(ptree['id'])+1 < 65535):
        #    raise ValueError("Max ID from halo is too big; You probably need to replace pairing id to 64-bit integer.")
        # desc_arr = np.full(ptree['desc'].shape, -1, dtype='i8')
        print("Setting halo id from pairing function...")
        halo_uid = pairing(ptree['timestep'].astype(dtype_id), ptree['id'].astype(dtype_id), ignore=-1)

        lookup, rankup = ptree['desc'].shape[-2:]
        lookup += 1

        iouts = np.unique(ptree['timestep'])
        np.sort(iouts)
        iouts_next = np.zeros((np.max(iouts+1), lookup-1), dtype='i4')
        for ilook in range(1, lookup):
            # set array indicates next iout
            iouts_next[iouts[:-ilook], ilook-1] = iouts[ilook:]

        iout_desc = iouts_next[ptree['timestep']][..., np.newaxis]
        desc_arr = pairing(iout_desc.astype(dtype_id), ptree['desc'].astype(dtype_id), ignore=-1)

        #for ilook in np.arange(1, lookup):
        #    idxs = np.searchsorted(iouts, ptree['timestep']) + ilook
        #    iout_desc = np.select([idxs<iouts.size, True], [iouts[np.minimum(idxs, iouts.size-1)], -1])
        #    desc_arr[:, ilook-1] = pairing(expand_shape(iout_desc, 0, 2), ptree['desc'][:, ilook-1], ignore=-1)
        print("Rewriting array...")
        ptree_add = np.zeros(ptree.shape, dtype=[('id', dtype_id), ('desc', 'i8', (lookup-1, rankup))])

        if(save_hmid):
            hmid = ptree['id']
            ptree = append_fields(ptree, 'hmid', hmid, usemask=False)

        ptree = drop_fields(ptree, ['id', 'desc'], usemask=False)
        ptree = merge_arrays([ptree, ptree_add], usemask=False, flatten=True)
        ptree['id'] = halo_uid
        ptree['desc'] = desc_arr

        return ptree

    @staticmethod
    def build_tree(ptree, overwrite=False, jump_ratio=0.5):
        print('Building tree from %d halo-nodes...' % ptree.size)
        names = ['fat', 'son', 'score_fat', 'score_son']
        ptree.sort(order='id')
        if(overwrite):
            ptree = drop_fields(ptree, names, usemask=False)
        id_ini = np.full(ptree.size, -1, dtype='i8')
        score_ini = np.zeros(ptree.size, dtype='f8')
        ptree = append_fields(
            ptree, names,
            [id_ini, id_ini, score_ini, score_ini], usemask=False)

        lookup, rankup = ptree['desc'].shape[-2:]
        lookup += 1

        frac_send = ptree['npass'] / expand_shape(ptree['nparts'], 0, 3)
        desc_idx = np.searchsorted(ptree['id'], ptree['desc'])
        frac_recv = ptree['npass'] / ptree['nparts'][desc_idx]
        score = frac_recv * frac_send

        # score is reduced by jump_ratio per each jump
        score *= expand_shape(jump_ratio**np.arange(1, lookup), 1, 3)
        max_score = np.zeros(ptree.size, dtype='f8')

        # flatten rank-lookup axis
        score = score.reshape(*score.shape[:-2], -1)
        desc = ptree['desc'].reshape(*ptree['desc'].shape[:-2], -1)
        score_max = np.max(score, axis=-1)
        mask = np.any(score>0, axis=-1)

        ptree['son'][mask] = desc[np.arange(0, ptree.size), np.argmax(score, axis=-1)][mask]
        ptree['score_son'] = score_max

        score = score.flatten()

        # sort descendent/progenitor index by increasing order
        key = np.argsort(score)
        score = score[key]
        mask = score > 0
        desc_idx = desc_idx.flatten()[key][mask]
        prog_idx = np.repeat(np.arange(ptree.size), rankup*(lookup-1))[key][mask]

        # We put progenitor id on the place of their respective descendent
        # progenitors with bigger score are always written later
        ptree['fat'][desc_idx] = ptree['id'][prog_idx]
        ptree['score_fat'][desc_idx] = score[mask]

        return ptree

    @staticmethod
    def add_tree_info(ptree, overwrite=False):
        # this function adds some useful tree information from tree data already built
        """
        This routine pre-computes father/son search with ptree.

        last : last descendent of "current" branch, if it ended up merging with other galaxy and it is not a main
        progenitor, the branch will end just before the merger. The search algorithm is done by choosing the 'best'
        progenitor among the nodes that are pointing me as a son in previous snapshot. The choice is done by following
        conditions.

            condition 1: if it is mutually linked to each other
            condition 2: higher score

        first : first progenitor of "current" main branch, including the main progenitor when there is a merger

        Parameters
        ----------
        ptree
        overwrite

        Returns
        -------
        An updated ptree table

        """
        print('Adding tree info from %d halo-nodes...' % ptree.size)
        names = ['nprog', 'ndesc', 'first', 'last', 'first_rev', 'last_rev']
        if(overwrite):
            ptree = drop_fields(ptree, names, usemask=False)
        id_ini = np.full(ptree.size, -1, dtype='i8')
        num_ini = np.zeros(ptree.size, dtype='i8')
        ptree = append_fields(
            ptree, names,
            [num_ini, num_ini, id_ini, id_ini, id_ini, id_ini], usemask=False)

        def search_tree_reverse(son_name, fat_name, last_name, nprog_name):
            ptree.sort(order=son_name)
            id_key = np.argsort(ptree['id'])

            # first fill end-nodes (leafs and roots)
            # nodes without son indicates they are either roots or branches
            last_idx = np.where(ptree[son_name]==-1)[0]
            ptree[last_name] = ptree['id']
            #ptree[last_name][last_idx] = ptree['id'][last_idx]

            t = tqdm(total=ptree.size)
            stat = [0, 0, 0]
            while last_idx.size > 0:
                t.update(last_idx.size)

                # leave only nodes referenced by other node as son
                # if none referenced, they are no longer alive (end of branch)
                last_idx = last_idx[np.isin(ptree[last_idx]['id'], ptree[son_name])]
                now = ptree[last_idx]
                my_id = now['id']

                # count the number of progenitors
                # find the range of idx of other nodes where pointing itself as a son
                lidxs = np.searchsorted(ptree[son_name], my_id, side='left')
                ridxs = np.searchsorted(ptree[son_name], my_id, side='right')
                ptree[nprog_name][last_idx] = ridxs - lidxs

                # find my father index (unique)
                fat_idx = id_key[np.searchsorted(ptree['id'], now[fat_name], sorter=id_key)]
                # check if there is any mutual link (fat & son directing each other)
                mask_mutual = (now[fat_name] != -1) & (my_id == ptree[son_name][fat_idx])

                # if there's no mutually linked node, select the progenitor with highest score
                fat_idx[~mask_mutual] = [np.argmax(ptree[l:r]['score_%s' % son_name])+l for l, r in zip(lidxs[~mask_mutual], ridxs[~mask_mutual])]

                # fill only mutually linked nodes (multiplicative)
                ptree[last_name][fat_idx] = ptree[last_name][last_idx]

                # I found no better solution then using for loop here...
                last_idx = [np.arange(l, r) for l, r in zip(lidxs, ridxs)]
                if(len(last_idx)>0):
                    last_idx = np.concatenate(last_idx)
                else:
                    last_idx = np.array(last_idx)
                stat[0] += np.sum(mask_mutual)
                stat[1] += np.sum(~mask_mutual)
            t.close()
            stat[2] = np.sum(~np.isin(ptree['id'], ptree[son_name]))

            print('%8d nodes have mutual link.' % stat[0])
            print('%8d nodes do not have mutual link.' % stat[1])
            print('%8d nodes are last points.' % stat[2])
            print('%8d nodes were processed in total' % np.sum(stat))

        search_tree_reverse('son', 'fat', 'last', 'ndesc')
        search_tree_reverse('fat', 'son', 'first', 'nprog')

        return ptree


    @staticmethod
    def load(repo, full_path=None, path_in_repo=path_in_repo, ptree_file=ptree_file, ptree_file_format=ptree_file_format, iout=None, msg=True):
        if(isinstance(repo, uri.RamsesSnapshot)):
            repo = repo.repo
        if(iout is None):
            filename = ptree_file
        else:
            filename = ptree_file_format % iout
        path = os.path.join(repo, path_in_repo, filename) if full_path is None else os.path.join(full_path, filename)
        return load(path, msg=msg)

    @staticmethod
    def save(ptree, repo, full_path=None, path_in_repo=path_in_repo, ptree_file=ptree_file, msg=True, format='pkl'):
        path = os.path.join(repo, path_in_repo, ptree_file) if full_path is None else os.path.join(full_path, ptree_file)
        dump(ptree, path, msg=msg, format=format)

    @staticmethod
    def measure_star_prop(snap, path_in_repo=path_in_repo, halomaker_repo=default_path_in_repo['GalaxyMaker'], ptree_file=ptree_file,
                          overwrite=True, backup_freq=30, sfr_measure_Myr=50., mass_cut_refine=2.4E-11,
                          output_file='ptree_SFR.pkl', backup_file='ptree_SFR.pkl.backup'):
        """
        repo should be specified here since we use particle data.
        following stellar properties are added
            sfr, sfr2, sfr4: star formation rate using 50, 100, 200Myr interval [Msol/yr]
            r90, r50: radius that encloses 90% and 50% of the total mass [code unit]
            age: mean age of stars [Gyr]
            tform: formation epoch (half-formation time) in age of the universe [Gyr]
            metal: mean metallicity of stars
            contam: mass fraction of low-resolution dark matter particles within r90
            mbh: mass of most massive bh within r90 [Msol]
            bh_offset: most massive bh's spatial offset from galactic center [code unit]
        """
        repo = snap.repo

        print("Starting stellar properties measure for %s" % repo)
        ptree = PhantomTree.load(repo, path_in_repo, ptree_file=ptree_file)

        fields = ['sfr', 'sfr2', 'sfr4', 'r90', 'r50', 'age', 'tform', 'metal', 'contam', 'mbh', 'bh_offset']
        if(overwrite):
            ptree = drop_fields(ptree, ['idx', *fields], usemask=False)
        zero_double = np.zeros(ptree.size, dtype='f8')
        ptree = append_fields(ptree, ['idx', *fields],
                              [np.arange(ptree.size), *(zero_double,)*len(fields)], usemask=False)

        iouts = np.sort(np.unique(ptree['timestep']))
        min_iout = np.min(iouts)

        part_ids = HaloMaker.load(snap, halomaker_repo, galaxy=True, load_parts=True)[1]
        max_part_size = int(np.max(part_ids) * 1.2)
        part_pool = np.full(max_part_size, -1, dtype='i4')

        uri.timer.verbose = 0
        uri.verbose = 0

        for iout in tqdm(iouts):
            nsnap = snap.switch_iout(iout)

            mask = ptree['timestep'] == iout
            gals = ptree[mask]

            gals.sort(order='hmid')

            halomaker, part_ids = HaloMaker.load(nsnap, halomaker_repo, galaxy=True, load_parts=True)

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

                r50 = weighted_quantile(dists, 0.5, sample_weight=gal_star['m'])
                sfr = np.sum(gal_star[gal_star['age', 'Myr']<sfr_measure_Myr]['m', 'Msol']) / (sfr_measure_Myr*1E6)
                sfr2 = np.sum(gal_star[gal_star['age', 'Myr']<sfr_measure_Myr*2]['m', 'Msol']) / (sfr_measure_Myr*2E6)
                sfr4 = np.sum(gal_star[gal_star['age', 'Myr']<sfr_measure_Myr*4]['m', 'Msol']) / (sfr_measure_Myr*4E6)

                age = np.average(gal_star['age', 'Gyr'], weights=gal_star['m'])
                tform = np.median(snap.age-gal_star['age', 'Gyr'])
                metal = np.average(gal_star['metal'], weights=gal_star['m'])

                dm_r90 = uri.cut_halo(dm, gal, r90, use_halo_radius=False)
                contam = np.sum(dm_r90[dm_r90['m']>mass_cut_refine]['m'])/np.sum(dm_r90['m'])

                bh_r90 = uri.cut_halo(smbh, gal, r90, use_halo_radius=False)

                ptree['sfr'][gal['idx']] = sfr
                ptree['sfr2'][gal['idx']] = sfr2
                ptree['sfr4'][gal['idx']] = sfr4
                ptree['r90'][gal['idx']] = r90
                ptree['r50'][gal['idx']] = r50

                ptree['nparts'][gal['idx']] = gal_star.size
                ptree['contam'][gal['idx']] = contam
                ptree['age'][gal['idx']] = age
                ptree['tform'][gal['idx']] = tform
                ptree['metal'][gal['idx']] = metal

                if(bh_r90.size>0):
                    bh_max = bh_r90[np.argmax(bh_r90['m'])]
                    mbh = bh_max['m']/snap.unit['Msol']
                    bh_offset = get_distance(gal, bh_max)

                    ptree['mbh'][gal['idx']] = mbh
                    ptree['bh_offset'][gal['idx']] = bh_offset

            if(iout % backup_freq == 0):
                PhantomTree.save(ptree, repo, path_in_repo, ptree_file=backup_file)

            nsnap.clear()
            gc.collect()

        uri.timer.verbose = 1
        uri.verbose = 1
        ptree = drop_fields(ptree, 'idx', usemask=False)
        os.remove(os.path.join(repo, path_in_repo, backup_file))
        PhantomTree.save(ptree, repo, path_in_repo, ptree_file=output_file)

    @staticmethod
    def measure_gas_prop(snap, path_in_repo=path_in_repo, ptree_file=ptree_file,
                         backup_freq=30, min_radius=1., max_radius=4., radius_name='r90', iout_start=0,
                         measure_contam=True, output_file='ptree_RPS.pkl', backup_file='ptree_RPS.pkl.backup',
                         subload_limit=10000, n_jobs=8, nchunk=10):
        def wgas_mask(cell):
            # Torrey et al. 2012
            return np.log10(cell['T', 'K']) < 6 + 0.25 * np.log10(cell['rho', 'Msol/kpc3'] * snap['h'] ** 2 / 10 ** 10)

        def cgas_mask(cell):
            return np.log10(cell['T', 'K']) < 3.5 + 0.25 * np.log10(cell['rho', 'Msol/kpc3'] * snap['h'] ** 2 / 10 ** 10)

        def get_cpulist_max(snap, halos, radius=3., use_halo_radius=True, radius_name='rvir', n_divide=4):
            cpulist_all = []
            for halo in halos:
                if(use_halo_radius):
                    extent = halo[radius_name]*radius*2
                else:
                    extent = radius*2
                box = uri.get_box(get_vector(halo), extent)
                cpulist = uri.get_cpulist(box, None, snap.levelmax, snap.bound_key, snap.ndim, n_divide)
                cpulist_all.append(np.max(cpulist))
            return np.array(cpulist_all)

        def load_cell_snap(snap, targets):
            cpulist = snap.get_halos_cpulist(targets, radius=max_radius*1.25, radius_name=radius_name)
            print("Number of domains to load = %d / %d" % (cpulist.size, snap.ncpu))
            snap.get_cell(cpulist=cpulist)

        def measure_rgas(gas, gal, rgal):
            """
            We classify gas as 5 different categories
                cgas: cold gas; represents molecular cloud, HI clouds
                wgas: warm gas; represents circumgalactic medium, HII regions
                hgas: (inner) hot gas; represents hot bubble from SN and AGN, should be within min_radius
                ogas: (outer) hot gas; represents ICM/IGM
                rgas: a radius where the volume of (outer) hot and cold gas becomes same.
            """
            mask = wgas_mask(gas)
            wgas = gas[mask] # includes cg also
            hgas = gas[~mask] # includes og also

            # sort wgas by the distance from center
            wgas_dist = get_distance(gal, wgas)
            key = np.argsort(wgas_dist)
            wgas = wgas[key]
            wgas_dist = wgas_dist[key]

            # calculate the cumsum of volume, find the minimum index where the distance is larger than min_radius
            wgas_cum = np.cumsum(wgas['vol'])
            min_idx = np.searchsorted(wgas_dist, min_radius * rgal)
            wgas_dist = wgas_dist[min_idx:]
            wgas_cum = wgas_cum[min_idx:]

            # divide hgas and ogas
            hgas_dist = get_distance(gal, hgas)
            ogas = hgas[hgas_dist >= min_radius * rgal]
            hgas = hgas[hgas_dist < min_radius * rgal]

            # sort ogas by the distance from center
            ogas_dist = get_distance(gal, ogas)
            key = np.argsort(ogas_dist)
            ogas = ogas[key]
            ogas_dist = ogas_dist[key]
            ogas_cum = np.cumsum(ogas['vol'])
            min_idx = np.searchsorted(ogas_dist, min_radius * rgal)
            ogas_dist = ogas_dist[min_idx:]
            ogas_cum = ogas_cum[min_idx:]

            # measure rgas based on the following criterion
            # 1. rgas is selected between min - max radius
            # 2. rgas is where cumulative sum of wgas and ogas crosses each other
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

            wgas = uri.cut_halo(wgas, gal, rgas, use_halo_radius=False)
            mask = cgas_mask(wgas)
            cgas = wgas[mask]
            wgas = wgas[~mask]
            ogas = uri.cut_halo(ogas, gal, rgas*1.25, use_halo_radius=False)

            return cgas, wgas, hgas, ogas, rgas

        def set_gas_properties(gal, gas, phase):
            # gal_gas is a pointer from ptree array that represents specific phase of the gas
            vgal = get_vector(gal, 'v') * snap.unit['km/s']
            pgal = get_vector(gal)
            gal_gas = gal[phase]

            gal_gas['m'] = np.sum(gas['m'])
            gal_gas['rho'] = np.average(gas['rho'], weights=gas['m'])
            gal_gas['metal'] = np.average(gas['metal'], weights=gas['m'])

            vel = gas['vel'] - vgal
            pos = (gas['pos'] - pgal)
            vrad = np.sum(vel * pos, axis=-1) / rss(pos)
            gal_gas['vr'] = np.average(vrad, weights=gas['m'])

            am = np.cross(pos, vel) * expand_shape(gas['m'], 0, 2)

            set_vector(gal_gas, np.average(vel, axis=0, weights=gas['m']), 'v')
            set_vector(gal_gas, np.sum(am, axis=0), 'L')

            if('contam' in gal_gas.dtype.names):
                gal_gas['contam'] = np.sum(gas[gas['refmask']<0.01]['m'])/gal_gas['m']

        def fin_gas_propertiles(gals, phase):
            gal_gas = gals[phase]
            gal_gas['m'] /= snap.unit['Msol']
            gal_gas['rho'] /= snap.unit['H/cc']

            gal_gas['vr'] /= snap.unit['km/s']
            gal_gas['vx'] /= snap.unit['km/s']
            gal_gas['vy'] /= snap.unit['km/s']
            gal_gas['vz'] /= snap.unit['km/s']

            am_unit = (snap.unit['km/s'] * snap.unit['Mpc'] * snap.unit['Msol'])
            gal_gas['Lx'] /= am_unit
            gal_gas['Ly'] /= am_unit
            gal_gas['Lz'] /= am_unit

        def measure_galaxy(st, ed, q):
            mygal = gals[st:ed]
            for gal in mygal:
                snap.set_box_halo(gal, max_radius, radius_name=radius_name)
                cell = snap.get_cell()
                cell = uri.cut_halo(cell, gal, max_radius, radius_name=radius_name)
                if(cell.size == 0):
                    return

                cgas, wgas, hgas, ogas, rgas = measure_rgas(cell, gal, gal[radius_name])

                gal['rgas'] = rgas
                if(cgas.size>0):
                    set_gas_properties(gal, cgas, 'cgas')
                if(wgas.size>0):
                    set_gas_properties(gal, wgas, 'wgas')
                if(hgas.size>0):
                    set_gas_properties(gal, hgas, 'hgas')
                if(ogas.size > 0):
                    set_gas_properties(gal, ogas, 'ogas')
                    gal['pram'] = ss(get_vector(gal['ogas'], 'v')) * gal['ogas']['rho']

            q.put((st, ed, mygal))

        repo = snap.repo
        ptree = PhantomTree.load(repo, path_in_repo=path_in_repo, ptree_file=ptree_file)
        #ort_key = np.argsort(ptree['x'])
        #sort_key_rev = np.arange(ptree.size)[sort_key]
        #ptree = ptree[sort_key]

        # Some workarounds for adding fields

        gas_phases = ['cgas', 'wgas', 'hgas', 'ogas']

        if(iout_start == 0):
            gas_names = ['m', 'vx', 'vy', 'vz', 'vr', 'Lx', 'Ly', 'Lz', 'rho', 'metal']
            if (measure_contam):
                gas_names.append('contam')

            extras = ['pram', 'rgas', 'idx']
            extra_formats = ['f8', 'f8', 'i4']

            gas_dtype = {
                'names': gas_names,
                'formats': ['f8'] * len(gas_names),
            }

            dtype = {
                'names': gas_phases + extras,
                'formats': [gas_dtype] * len(gas_phases) + extra_formats,
            }

            ptree = drop_fields(ptree, gas_phases, usemask=True)

            dtype = np.dtype(dtype)
            dtype = np.dtype(ptree.dtype.descr + dtype.descr)

            new = np.zeros(ptree.size, dtype=dtype)
            for name in ptree.dtype.names:
                new[name] = ptree[name]
            new['idx'] = np.arange(ptree.size)
            ptree = new
            iout_start = np.max(ptree['timestep'])

        iouts = np.unique(ptree['timestep'])
        iouts = iouts[iouts<=iout_start]
        iouts.sort()

        print("Measuring gas properties from iout = %d to %d..." % (iouts[-1], iouts[0]))

        uri.timer.verbose = 0
        uri.verbose = 0

        ts = uri.TimeSeries(snap)

        for iout in tqdm(iouts[::-1]):
            snap = ts[iout]
            gals_total = ptree[ptree['timestep'] == iout]

            # Sort halos according to max cpu id of their underlying domain
            cpulist_max = get_cpulist_max(snap, gals_total, radius=max_radius*1.25, radius_name=radius_name)
            gals_total = gals_total[np.argsort(cpulist_max)]

            if(gals_total.size > subload_limit):
                nsubload = int(np.ceil(gals_total.size/subload_limit))
            else:
                nsubload = 1
            for isubload in np.arange(nsubload):
                gals = gals_total[isubload*subload_limit:np.minimum(gals_total.size, (isubload+1)*subload_limit)]
                load_cell_snap(snap, gals)

                jobs = []
                iterator = tqdm(np.arange(int(np.ceil(gals.size/nchunk))), ncols=100)
                q = Queue()
                for i in iterator:
                    while (True):
                        for idx in np.arange(len(jobs))[::-1]:
                            if (not jobs[idx].is_alive()):
                                jobs.pop(idx)
                        if (len(jobs) >= n_jobs):
                            sleep(0.5)
                        else:
                            break
                    st, ed = i*nchunk, np.minimum((i+1)*nchunk, gals.size)
                    p = Process(target=measure_galaxy, args=(st, ed, q))
                    jobs.append(p)
                    p.start()
                    while not q.empty():
                        st, ed, procgal = q.get()
                        gals[st:ed] = procgal
                iterator.close()

                ok = False
                while not ok:
                    ok = True
                    for idx in np.arange(len(jobs)):
                        if (jobs[idx].is_alive()):
                            ok = False
                    if(not q.empty()):
                        st, ed, procgal = q.get()
                        gals[st:ed] = procgal
                    else:
                        sleep(0.5)


                for phase in gas_phases:
                    fin_gas_propertiles(gals, phase)

                gals['pram'] /= snap.unit['Ba']

                ptree[gals['idx']] = gals
                snap.clear()
                gc.collect()

            if(iout % backup_freq == 0):
                PhantomTree.save(ptree, repo, path_in_repo, ptree_file=backup_file, msg=True)

        uri.timer.verbose = 1
        uri.verbose = 1

        ptree = drop_fields(ptree, ['idx'], usemask=False)
        #ptree = ptree[sort_key_rev]
        os.remove(os.path.join(repo, path_in_repo, backup_file))
        PhantomTree.save(ptree, repo, path_in_repo, ptree_file=output_file)

    #@staticmethod
    #def repair_tree(ptree):

    @staticmethod
    def fix_mbh(ptree, snap):
        # check if mbh is alredy fixed
        if(any(ptree['mbh']>1)):
            return ptree
        else:
            ptree['mbh'] /= snap.unit['Msol']
        return ptree

    @staticmethod
    def fix_aexp(ptree, snap):
        # check if mbh is alredy fixed
        if(all(ptree['aexp']<=1)):
            return ptree
        else:
            ts = uri.TimeSeries(snap)
            iouts = np.unique(ptree['timestep'])
            iouts.sort()
            for iout in iouts:
                ptree['aexp'][ptree['timestep']==iout] = ts[iout].aexp

        return ptree


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
    def unit_conversion(array, snap):
        # distance in code unit, mass in Msun
        mass_unit = 1E11
        array['m'] *= mass_unit
        array['mvir'] *= mass_unit
        boxsize_physical = snap['boxsize_physical']

        pos = get_vector(array)
        append_fields(array, names=['xp', 'yp', 'zp'], data=pos.T, usemask=False)
        array['x'] = array['x'] / boxsize_physical / array['aexp'] + 0.5
        array['y'] = array['y'] / boxsize_physical / array['aexp'] + 0.5
        array['z'] = array['z'] / boxsize_physical / array['aexp'] + 0.5
        array['rvir'] /= boxsize_physical
        array['r'] /= boxsize_physical
        array['rc'] /= boxsize_physical

        return array

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

        print('Building table for %d nodes... ' % readh.integer_table.shape[-1] , end='')
        timer.start()
        array = fromarrays([*readh.integer_table.T, *readh.real_table.T], dtype=dtype)
        print('Took %.3fs' % timer.time())

        return TreeMaker.unit_conversion(array, snap)


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

