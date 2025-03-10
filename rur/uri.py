import os
from os.path import join, exists, getsize
from numpy.core.records import fromarrays as fromarrays

import scipy
if(scipy.__version__>='1.14.0'):
    from scipy.integrate import cumulative_trapezoid
    cumtrapz = cumulative_trapezoid
else:
    from scipy.integrate import cumtrapz
from collections.abc import Iterable
from scipy.spatial import cKDTree as KDTree
from numpy.lib.recfunctions import append_fields

from rur.fortranfile import FortranFile
from rur.hilbert3d import hilbert3d
from rur.readr import readr
from rur.io_ramses import io_ramses
from rur.config import *
from rur import utool
from rur.utool import *
import numpy as np
import warnings
import glob
import re
import datetime
from copy import deepcopy
from multiprocessing import Pool, shared_memory
import atexit, signal
from sys import exit
import configparser


class TimeSeries(object):
    """
    A class to manage multiple snapshots in the same repository
    """

    def __init__(self, snap: 'RamsesSnapshot'):
        self.snaps: dict[RamsesSnapshot] = {}
        self.basesnap = snap
        self.snaps[snap.iout] = snap
        self.iout_avail = None
        self.icoarse_avail = None

    def get_snap(self, iout=None, aexp=None, age=None) -> 'RamsesSnapshot':
        if (iout is None):
            if aexp is not None:
                self.read_iout_avail()
                iout = self.iout_avail[np.argmin(np.abs(self.iout_avail['aexp'] - aexp))]['iout']
            elif age is not None:
                self.read_iout_avail()
                iout = self.iout_avail[np.argmin(np.abs(self.iout_avail['age'] - age))]['iout']
            else:
                raise ValueError("One of the followings has to be specified: iout, aexp, age")

        if iout in self.snaps:
            return self.snaps[iout]
        else:
            self.snaps[iout] = self.basesnap.switch_iout(iout)
            return self.snaps[iout]

    def get_from_iout(self, iout):
        return self.get_snap(iout=iout)

    def get_from_aexp(self, aexp):
        return self.get_snap(aexp=aexp)

    def get_from_age(self, age):
        return self.get_snap(age=age)

    def __getitem__(self, item):
        return self.get_snap(item)

    def __getattr__(self, item):
        return self.basesnap.__getattribute__(item)

    def set_box_halo(self, halo, radius=1, use_halo_radius=True, radius_name='rvir', iout_name='timestep'):
        snap = self.get_snap(halo[iout_name])
        snap.set_box_halo(halo, radius=radius, use_halo_radius=use_halo_radius, radius_name=radius_name)
        return snap

    def interpolate_icoarse_table(self, value, name1, name2):
        if self.icoarse_avail is None:
            self.read_icoarse_avail()
        return np.interp(value, self.icoarse_avail[name1], self.icoarse_avail[name2])

    def icoarse_to_dt(self, icoarse):
        return self.interpolate_icoarse_table(icoarse + 0.5, 'icoarse', 'time') \
               - self.interpolate_icoarse_table(icoarse - 0.5, 'icoarse', 'time')

    def interpolate_iout_table(self, value, name1, name2):
        if self.iout_avail is None:
            self.read_iout_avail()
        return np.interp(value, self.iout_avail[name1], self.iout_avail[name2])

    def icoarse_to_aexp(self, icoarse):
        return self.interpolate_icoarse_table(icoarse, 'icoarse', 'aexp')

    def write_icoarse_avail(self, use_cache=True):
        path = join(self.repo, 'list_icoarse_avail.txt')
        self.write_iout_avail(use_cache=use_cache)
        # if(use_cache and exists(path)):
        #    self.read_icoarse_avail()
        timer.start("Writing available icoarses in %s..." % path, 1)
        icoarses = self.basesnap.search_sinkprops(path_in_repo='SINKPROPS')
        icoarse_table = np.zeros(icoarses.size, dtype=icoarse_avail_dtype)
        icoarse_table['icoarse'] = icoarses
        for i, icoarse in enumerate(icoarses):
            info = self.basesnap.read_sinkprop_info(path_in_repo='SINKPROPS', icoarse=icoarse, max_icoarse_offset=0)
            icoarse_table['aexp'][i] = info['aexp']
        names = icoarse_table.dtype.names
        to_add = self.iout_avail[~np.isin(self.iout_avail['icoarse'], icoarse_table['icoarse'])]
        icoarse_table_add = np.zeros(to_add.size, dtype=icoarse_avail_dtype)
        for name in icoarse_table_add.dtype.names:
            icoarse_table_add[name] = to_add[name]
        icoarse_table = np.concatenate([icoarse_table, icoarse_table_add])
        icoarse_table.sort(order='icoarse')
        icoarse_table['age'] = self.basesnap.aexp_to_age(icoarse_table['aexp'])
        icoarse_table['time'] = self.basesnap.interpolate_cosmo_table(icoarse_table['aexp'], 'aexp', 'u')
        self.icoarse_avail = icoarse_table
        np.savetxt(path, icoarse_table,
                   fmt='%18d %18.9e %18.9e %18.9e', header=('%16s' + ' %18s' * (len(names) - 1)) % names)
        timer.record()

    def read_icoarse_avail(self):
        path = join(self.repo, 'list_icoarse_avail.txt')
        if exists(path):
            self.icoarse_avail = np.loadtxt(path, dtype=icoarse_avail_dtype)
        else:
            return np.empty((0,), dtype=icoarse_avail_dtype)

    def write_iout_avail(self, use_cache=False):
        path = join(self.repo, 'list_iout_avail.txt')
        timer.start("Writing available timesteps in %s..." % path, 1)
        iouts = self.basesnap.get_iout_avail()
        if (use_cache and exists(path)):
            self.read_iout_avail()
        iout_table = np.zeros(len(iouts), dtype=iout_avail_dtype)
        i = 0
        for iout in iouts:
            if (use_cache and iout in self.iout_avail['iout']):
                iout_table[i] = self.iout_avail[np.searchsorted(self.iout_avail['iout'], iout)]
                i += 1
            else:
                try:
                    snap = self.get_snap(iout)
                except:
                    continue
                iout_table[i]['iout'] = iout
                iout_table[i]['aexp'] = snap.aexp
                iout_table[i]['age'] = snap.age
                iout_table[i]['icoarse'] = snap.nstep_coarse
                iout_table[i]['time'] = snap.time
                i += 1
        iout_table = iout_table[:i]
        names = iout_table.dtype.names
        self.iout_avail = iout_table
        np.savetxt(path, iout_table,
                   fmt='%18d %18.9e %18.9e %18d %18.9e', header=('%16s' + ' %18s' * (len(names) - 1)) % names)
        timer.record()

    def read_iout_avail(self):
        path = join(self.repo, 'list_iout_avail.txt')
        if exists(path):
            self.iout_avail = np.loadtxt(path, dtype=iout_avail_dtype)
        else:
            return np.empty((0,), dtype=iout_avail_dtype)

    def clear(self):
        # Later: need to load all **opened** snaps and clear them manually
        for iout in self.snaps.keys():
            self.snaps[iout].clear()
        self.snaps = {}
        self.basesnap.clear()


RamsesRepo = TimeSeries


class Particle(Table):
    def __init__(self, table, snap, units=None, ptype=None):
        super().__init__(table, snap, units=units)
        self.ptype = ptype
        self.extra_fields = custom_extra_fields(snap, 'particle')

    def __getitem__(self, item, return_code_unit=False):
        if isinstance(item, str):
            if item in part_family.keys():  # if an item exists among known particle family names
                if self.ptype is not None:
                    if item == self.ptype:
                        return self
                    else:
                        print(
                            f"\nYou loaded part only `{self.ptype}` but now you want `{item}`!\nIt forces to clear `{self.ptype}` data and retry get_part (so it's inefficient!)\n")
                        self.snap.part_data = None
                        self.snap.part = None
                        self.snap.box_part = None
                        cpulist = np.unique(self.snap.cpulist_part) if (
                                self.snap.box is None or np.array_equal(self.snap.box, self.default_box)) else None
                        self.snap.cpulist_part = np.array([], dtype='i4')
                        self.snap.bound_part = np.array([0], dtype='i4')
                        part = self.snap.get_part(box=self.snap.box, target_fields=self.table.dtype.names,
                                                  domain_slicing=True, exact_box=True, cpulist=cpulist, pname=item)
                        return part
                else:
                    return self.__copy__(classify_part(self.table, item, ptype=self.ptype), self.snap, ptype=item)

            elif item == 'smbh':
                # returns smbh position by summing up cloud particle positions
                return self.__copy__(find_smbh(self.table), self.snap, ptype='smbh')
        # none of the above, return to default
        return super().__getitem__(item, return_code_unit)

    def __copy__(self, table=None, snap=None, units=None, ptype=None):
        if table is None:
            table = self.table
        if snap is None:
            snap = self.snap
        if units is None:
            units = self.units
        if ptype is None:
            units = self.units
        return self.__class__(table, snap, units, ptype)


class Cell(Table):
    def __init__(self, table, snap, units=None):
        super().__init__(table, snap, units=units)
        self.extra_fields = custom_extra_fields(snap, 'cell')

    def __getitem__(self, item, return_code_unit=False):
        return super().__getitem__(item, return_code_unit)


# For multiprocessing, sub functions
def readorskip_real(f: FortranFile, dtype: type, key: str, search: Iterable, add=None):
    if key in search:
        if (add is not None):
            return f.read_reals(dtype) + add
        return f.read_reals(dtype)
    else:
        f.skip_records()


def readorskip_int(f: FortranFile, dtype: type, key: str, search: Iterable, add=None):
    if key in search:
        if (add is not None):
            return f.read_ints(dtype) + add
        return f.read_ints(dtype)
    else:
        f.skip_records()


def _classify(pname: str, npart:int, ids=None, epoch=None, m=None, family=None, sizeonly: bool = False, isstar=True):
    if (pname is None):
        mask = np.full(npart, True, dtype=bool)
        nsize = npart
    else:
        tracers = ["tracer", "cloud_tracer", "star_tracer", "gas_tracer"]
        if (family is not None):
            mask = np.isin(family, part_family[pname])
            nsize = np.count_nonzero(mask)
        elif (epoch is not None):
            if (pname == 'dm'):
                mask = (epoch == 0) & (ids > 0)
            elif (pname == 'star'):
                mask = ((epoch < 0) & (ids > 0)) | ((epoch != 0) & (ids < 0))
            elif (pname == 'sink' or pname == 'cloud'):
                mask = (ids < 0) & (m > 0) & (epoch == 0)
            nsize = np.count_nonzero(mask)
        elif (ids is not None):
            if (isstar):
                warnings.warn("Warning: either `family` or `epoch` should be given to classify particles.", UserWarning)
            if (pname == 'dm'):
                mask = ids > 0
                nsize = np.count_nonzero(mask)
            elif (pname in tracers):
                mask = (ids < 0) & (m == 0)
                nsize = np.count_nonzero(mask)
            else:
                mask = np.full(npart, False, dtype=bool)
                nsize = 0
        else:
            mask = np.full(npart, False, dtype=bool)
            nsize = 0
    if (sizeonly): return nsize
    return mask, nsize


def _calc_npart(fname: str, kwargs: dict, sizeonly=False):
    use_cache = kwargs.get('use_cache', True)
    repo = kwargs['repo']
    pname = kwargs.get('pname', None)
    if(pname is None): use_cache=False
    splits = fname.split('/')
    cache = f"{repo}/cache/{splits[-2]}/{pname}_{splits[-1]}.pkl"
    if (exists(cache))and(use_cache):
        result = utool.load(cache, msg=False)
        return result[0], result[1], int(fname[-5:])
    isfamily = kwargs.get('isfamily', False)
    isstar = kwargs.get('isstar', False)
    ids, epoch, m, family = None, None, None, None
    with FortranFile(f"{fname}", mode='r') as f:
        f.skip_records(2)
        npart = f.read_ints(np.int32)
        if(pname is None):
            result = _classify(pname, npart, ids=ids, epoch=epoch, m=m, family=family, sizeonly=sizeonly, isstar=isstar)
            return result[0], result[1], int(fname[-5:])
        f.skip_records(5)
        if (isfamily):
            f.skip_records(9)
            family = f.read_ints(np.int8)
        else:
            f.skip_records(6)
            m = f.read_reals(np.float64)
            ids = f.read_ints(np.int32)
            if (isstar):
                f.skip_records(1)
                epoch = f.read_reals(np.float64)
        result = _classify(pname, npart, ids=ids, epoch=epoch, m=m, family=family, sizeonly=sizeonly, isstar=isstar)
    if (not exists(cache))and(use_cache):
        if(not exists(f"{repo}/cache")):
            try: os.makedirs(f"{repo}/cache")
            except: pass
        try: utool.dump(result, cache, msg=False)
        except: pass
    return result[0], result[1], int(fname[-5:])


def _read_part(fname: str, kwargs: dict, part=None, mask=None, nsize=None, cursor=None, address=None,
               shape=None):
    pname, ids, epoch, m, family = None, None, None, None, None
    target_fields = kwargs["target_fields"] # input names
    dtype = kwargs["dtype"] # input dtypes
    ndeep = kwargs['ndeep']
    pname = kwargs["pname"]
    isfamily = kwargs["isfamily"]
    isstar = kwargs["isstar"]
    chem = kwargs["chem"]
    part_dtype = np.dtype(kwargs["part_dtype"]) # output dtype
    sequential = part is not None
    icpu = int(fname[-5:])
    with FortranFile(f"{fname}", mode='r') as f:
        # Read data
        f.skip_records(2)
        npart, = f.read_ints(np.int32)
        f.skip_records(5)
        if(ndeep>=1):
            x = readorskip_real(f, np.float64, 'x', target_fields)
            if(ndeep>=2):
                y = readorskip_real(f, np.float64, 'y', target_fields)
                if(ndeep>=3):
                    z = readorskip_real(f, np.float64, 'z', target_fields)
                    if(ndeep>=4):
                        vx = readorskip_real(f, np.float64, 'vx', target_fields)
                        if(ndeep>=5):
                            vy = readorskip_real(f, np.float64, 'vy', target_fields)
                            if(ndeep>=6):
                                vz = readorskip_real(f, np.float64, 'vz', target_fields)
                                if(ndeep>=7):
                                    if (pname is None)or(isfamily):
                                        m = readorskip_real(f, np.float64, 'm', target_fields)
                                        if(ndeep>=8):
                                            ids = readorskip_int(f, np.int32, 'id', target_fields)
                                    else:
                                        m = f.read_reals(np.float64)
                                        if(ndeep>=8):
                                            ids = f.read_ints(np.int32)
                                    if(ndeep>=9):
                                        level = readorskip_int(f, np.int32, 'level', target_fields)
                                        if(ndeep>=10):
                                            if (isfamily):
                                                family = f.read_ints(np.int8)  # family
                                                if(ndeep>=11):
                                                    tag = readorskip_int(f, np.int8, 'tag', target_fields)  # tag
                                            # if (isstar):
                                            if(ndeep>=12):
                                                if('epoch' in part_dtype.names):
                                                    epoch = readorskip_real(f, np.float64, 'epoch', target_fields) if pname is None else f.read_reals(np.float64) # epoch
                                            if(ndeep>=13):
                                                if('metal' in part_dtype.names):
                                                    metal = readorskip_real(f, np.float64, 'metal', target_fields)

        # Masking
        if (mask is None)or(nsize is None):
            mask, nsize = _classify(pname, npart, ids=ids, epoch=epoch, m=m, family=family, sizeonly=False, isstar=isstar)
            if (isinstance(mask, np.ndarray)):
                assert np.sum(mask) == nsize
        # Allocating
        if(address is None):
            if (part is None): part = np.empty(nsize, dtype=dtype)
            pointer = part[cursor:cursor + nsize].view() if (sequential) else part
        else:
            exist = shared_memory.SharedMemory(name=address)
            part = np.ndarray(shape=shape, dtype=dtype, buffer=exist.buf)
            pointer = part[cursor:cursor + nsize].view()
        if ('x' in target_fields): pointer['x'] = x[mask]
        if ('y' in target_fields): pointer['y'] = y[mask]
        if ('z' in target_fields): pointer['z'] = z[mask]
        if ('vx' in target_fields): pointer['vx'] = vx[mask]
        if ('vy' in target_fields): pointer['vy'] = vy[mask]
        if ('vz' in target_fields): pointer['vz'] = vz[mask]
        if ('m' in target_fields): pointer['m'] = m[mask]
        if ('epoch' in target_fields)and(isstar): pointer['epoch'] = epoch[mask]
        if ('metal' in target_fields)and(isstar): pointer['metal'] = metal[mask]
        if ('id' in target_fields): pointer['id'] = ids[mask]
        if ('level' in target_fields): pointer['level'] = level[mask]
        if ('family' in target_fields)and(isfamily): pointer['family'] = family[mask]
        if ('tag' in target_fields)and(isfamily): pointer['tag'] = tag[mask]
        if(ndeep>=14):
            newtypes = ["m0", "rho0", "partp"] + chem
            if True in np.isin(newtypes, target_fields):
                if('m0' in part_dtype.names):
                    if ('m0' in target_fields):
                        pointer['m0'] = f.read_reals(np.float64)[mask]
                    else:
                        f.read_reals(np.float64)
                if(ndeep>=15):
                    if len(chem) > 0:
                        for ichem in chem:
                            if (ichem in target_fields):
                                pointer[ichem] = f.read_reals(np.float64)[mask]
                            else:
                                f.read_reals(np.float64)
                    if('rho0' in part_dtype.names):
                        if ('rho0' in target_fields):
                            pointer['rho0'] = f.read_reals(np.float64)[mask]
                        else:
                            f.read_reals(np.float64)
                    if(ndeep>=16):
                        if('partp' in target_fields):
                            if ('partp' in target_fields):
                                pointer['partp'] = f.read_ints(np.int32)[mask]
                            else:
                                f.read_ints(np.int32)
        pointer['cpu'] = icpu
    if (sequential):
        cursor += nsize
        return cursor
    exist.close()


def _calc_ncell(fname: str, amr_kwargs: dict):
    ncpu = amr_kwargs['ncpu']
    nboundary = amr_kwargs['nboundary']
    nlevelmax = amr_kwargs['nlevelmax']
    ndim = amr_kwargs['ndim']
    twotondim = amr_kwargs['twotondim']
    skip_amr = amr_kwargs['skip_amr']

    icpu = int(fname[-5:])
    ncell = 0
    with FortranFile(fname, mode='r') as f:
        f.skip_records(21)
        numbl = f.read_ints()
        f.skip_records(3)
        if(nboundary>0):
            numbb = f.read_ints()
            f.skip_records(2)
        ngridfile = np.empty((ncpu + nboundary, nlevelmax), dtype='i4')
        for ilevel in range(nlevelmax):
            ngridfile[:ncpu, ilevel] = numbl[ncpu * ilevel: ncpu * (ilevel + 1)]
            if(nboundary>0):
                ngridfile[ncpu:ncpu+nboundary, ilevel]=numbb[nboundary*ilevel : nboundary*(ilevel+1)]
        f.skip_records(4)
        levels, cpus = np.where(ngridfile.T > 0)
        for ilevel, jcpu in zip(levels, cpus + 1):
            f.skip_records(3)
            if jcpu == icpu:
                f.skip_records(3 * ndim + 1)
                for _ in range(twotondim):
                    son = f.read_ints()
                    if 0 in son:
                        ncell += len(son.flatten()) - np.count_nonzero(son)
                f.skip_records(2 * twotondim)
            else:
                f.skip_records(skip_amr)
    return ncell


def _read_cell(icpu: int, snap_kwargs: dict, amr_kwargs: dict, cell=None, nsize=None, cursor=None,
               address=None, shape=None):
    # 0) From snapshot
    nhvar = snap_kwargs['nhvar']
    hydro_names = snap_kwargs['hydro_names']
    repo = snap_kwargs['repo']
    iout = snap_kwargs['iout']
    ncpu = amr_kwargs['ncpu']
    skip_hydro = snap_kwargs['skip_hydro']
    read_grav = snap_kwargs['read_grav']
    dtype = snap_kwargs['dtype']
    target_fields = snap_kwargs['names']

    nboundary = amr_kwargs['nboundary']
    nlevelmax = amr_kwargs['nlevelmax']
    ndim = amr_kwargs['ndim']
    twotondim = amr_kwargs['twotondim']
    skip_amr = amr_kwargs['skip_amr']
    boxlen = amr_kwargs['boxlen']
    icoarse_min = amr_kwargs['icoarse_min']
    jcoarse_min = amr_kwargs['jcoarse_min']
    kcoarse_min = amr_kwargs['kcoarse_min']

    # 1) Read headers
    hydro_fname = f"{repo}/output_{iout:05d}/hydro_{iout:05d}.out{icpu:05d}"
    f_hydro = FortranFile(hydro_fname, mode='r')
    f_hydro.skip_records(6)

    if (read_grav):
        grav_fname = f"{repo}/output_{iout:05d}/grav_{iout:05d}.out{icpu:05d}"
        if(not os.path.exists(grav_fname)):
            read_grav = False
        else:
            f_grav = FortranFile(grav_fname, mode='r')
            f_grav.skip_records(1)
            ndim1, = f_grav.read_ints()
            output_particle_density = ndim1 == ndim + 2
            f_grav.skip_records(2)
            skip_grav = twotondim * (2 + ndim) if output_particle_density else twotondim * (1 + ndim)

    amr_fname = f"{repo}/output_{iout:05d}/amr_{iout:05d}.out{icpu:05d}"
    sequential = True
    if (cell is None): sequential = False
    if (nsize is None): nsize = _calc_ncell(amr_fname, amr_kwargs)
    f_amr = FortranFile(amr_fname, mode='r')
    f_amr.skip_records(21)
    numbl = f_amr.read_ints()
    f_amr.skip_records(3)
    if nboundary>0:
        numbb=f_amr.read_ints()
        f_amr.skip_records(2)
        ngridfile = np.vstack((numbl.reshape(nlevelmax, ncpu).T, numbb.reshape(nlevelmax, nboundary).T))
    else:
        ngridfile = numbl.reshape(nlevelmax, ncpu).T
    f_amr.skip_records(4)
    if (cursor is None): cursor = 0
    if (address is None):
        if (cell is None): cell = np.empty(nsize, dtype=dtype)
        pointer = cell[cursor:cursor + nsize].view() if (sequential) else cell
        icursor = 0 if (sequential) else cursor
    else:
        exist = shared_memory.SharedMemory(name=address)
        cell = np.ndarray(shape=shape, dtype=dtype, buffer=exist.buf)
        pointer = cell[cursor:cursor + nsize].view()
        icursor = 0

    # 2) Level by Level
    # Loop over levels
    for ilevel in range(nlevelmax):
        ncpu_befo = icpu - 1
        ncpu_afte = ncpu + nboundary - icpu
        ncache_befo = np.count_nonzero(ngridfile[:ncpu_befo, ilevel])
        ncache = ngridfile[icpu - 1, ilevel]
        ncache_afte = np.count_nonzero(ngridfile[icpu:, ilevel])

        # Skip jcpu<icpu
        f_hydro.skip_records(2 * ncpu_befo + skip_hydro * ncache_befo)
        f_amr.skip_records((3 + skip_amr) * ncache_befo)
        if (read_grav): f_grav.skip_records(2 * ncpu_befo + skip_grav * ncache_befo)
        # Now jcpu==icpu
        f_hydro.skip_records(2)
        if (read_grav): f_grav.skip_records(2)
        if (ncache > 0):
            f_amr.skip_records(3)
            x = readorskip_real(f_amr, np.float64, 'x', target_fields,
                                add=oct_x / 2 ** (ilevel + 1))
            y = readorskip_real(f_amr, np.float64, 'y', target_fields,
                                add=oct_y / 2 ** (ilevel + 1))
            z = readorskip_real(f_amr, np.float64, 'z', target_fields,
                                add=oct_z / 2 ** (ilevel + 1))
            
            # Convert xyz coordinate to boxlen unit
            x = (x - icoarse_min) * boxlen
            y = (y - jcoarse_min) * boxlen
            z = (z - kcoarse_min) * boxlen

            f_amr.skip_records(2 * ndim + 1)  # Skip father index & nbor index
            # Read son index to check refinement
            ileaf = f_amr.read_arrays(twotondim) == 0
            f_amr.skip_records(2 * twotondim)  # Skip cpu, refinement map
            icell = np.count_nonzero(ileaf)
            # Allocate hydro variables
            hydro_vars = [None] * nhvar
            for ivar in range(nhvar):
                if (hydro_names[ivar] in target_fields):
                    hydro_vars[ivar] = np.empty((twotondim, ncache), dtype='f8')
            if (read_grav): grav_vars = np.empty((twotondim, ncache), dtype='f8')

            # Read hydro variables
            for j in range(twotondim):
                for ivar in range(nhvar):
                    if (hydro_names[ivar] in target_fields):
                        hydro_vars[ivar][j] = f_hydro.read_reals()
                    else:
                        f_hydro.skip_records(1)
                if (read_grav):
                    if output_particle_density: f_grav.skip_records(1)
                    grav_vars[j] = f_grav.read_reals()
                    f_grav.skip_records(ndim)

            # Merge amr & hydro data
            if True in ileaf:
                if ('x' in target_fields): pointer[icursor: icursor + icell]['x'] = x[ileaf]
                if ('y' in target_fields): pointer[icursor: icursor + icell]['y'] = y[ileaf]
                if ('z' in target_fields): pointer[icursor: icursor + icell]['z'] = z[ileaf]
                for ivar in range(nhvar):
                    key = hydro_names[ivar]
                    if (key in target_fields): pointer[icursor: icursor + icell][key] = hydro_vars[ivar][ileaf]
                if (read_grav): pointer[icursor: icursor + icell]['pot'] = grav_vars[ileaf]
                pointer[icursor: icursor + icell]['level'] = ilevel + 1
                pointer[icursor: icursor + icell]['cpu'] = icpu

                icursor += icell
                cursor += icell
        # Skip jcpu>icpu
        f_hydro.skip_records(2 * ncpu_afte + skip_hydro * ncache_afte)
        f_amr.skip_records((3 + skip_amr) * ncache_afte)
        if (read_grav): f_grav.skip_records(2 * ncpu_afte + skip_grav * ncache_afte)
    f_amr.close()
    f_hydro.close()
    if (read_grav): f_grav.close()
    if (sequential):
        return cursor
    exist.close()


class RamsesSnapshot(object):
    """A handy object to store RAMSES AMR/Particle snapshot data.

    One RamsesSnapshot represents single snapshot of the simulation. Efficient reads from RAMSES binary are guaranteed
    by only touching necessary files.

    Parameters
    ----------
    repo : string
        Repository path, path to directory where RAMSES raw data is stored.
    iout : integer
        RAMSES output number
    mode : 'none', 'yzics', 'nh', 'gem', etc.
        RAMSES version. Define it in config.py if necessary
    box : numpy.array with shape (3, 2)
        Bounding box of region to open in the simulation, in code unit, should be within (0, 1)
    path_in_repo : string
        Path to actual snapshot directories in the repository path, path where 'output_XXXXX' is stored.
    full_path : string
        Overrides repo and path_in_repo, full path to directory where 'output_XXXXX' is stored.
    snap : RamsesSnapshot object
        Previously created snapshot object from same simulation (but different output number) saves time to compute
        cosmology.

    Attributes
    ----------
    part : Particle object
        Loaded particle table in the box
    cell : Cell object
        Loaded particle table in the box
    box : numpy.array with shape (3, 2)
        Current bounding box. Change this value if bounding box need to be changed.
    part_data : numpy.recarray
        Currently cached particle data, if data requested by get_part() have already read, retrieve from this instead.
    cell_data : numpy.recarray
        Currently cached cell data, if data requested by get_cell() have already read, retrieve from this instead.

    Examples
    ----------
    To read data from ramses raw binary data:

    >>> from rur import uri
    >>> repo = "path/to/your/ramses/outputs"
    >>> snap = uri.RamsesSnapshot(repo, iout=1, path_in_repo='')
    >>> snap.box = np.array([[0, 1], [0, 1], [0, 1]])

    To read particle and cell data:

    >>> snap.get_part()
    >>> snap.get_cell()
    >>> print(snap.part.dtype)
    dtype((numpy.record, [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
    ('m', '<f8'), ('epoch', '<f8'), ('metal', '<f8'), ('id', '<i4'), ('level', 'u1'), ('cpu', '<i4')]))

    >>> print(snap.cell.dtype)
    dtype((numpy.record, [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('rho', '<f8'), ('vx', '<f8'), ('vy', '<f8'),
    ('vz', '<f8'), ('P', '<f8'), ('metal', '<f8'), ('zoom', '<f8'), ('level', '<i4'), ('cpu', '<i4')]))

    >>> snap.clear()

    """

    def __init__(self, repo, iout, mode='none', box=None, path_in_repo=default_path_in_repo['snapshots'], snap=None,
                 longint=False, verbose=None, z=None):
        self.repo = repo
        self.path_in_repo = path_in_repo
        self.snap_path = join(repo, path_in_repo)
        self.verbose = timer.verbose if verbose is None else verbose

        if z is not None:
            path = join(self.repo, 'list_iout_avail.txt')
            if exists(path):
                iout_avail = np.loadtxt(path, dtype=iout_avail_dtype)
            zs = 1/iout_avail['aexp'] - 1
            if (z > zs.max())or(z < zs.min()):
                raise ValueError(f"z={z} is out of range ({zs.min():.6f} ~ {zs.max():.6f})")
            argmin = np.argmin(np.abs(zs - z))
            iout = iout_avail[argmin]['iout']
            print(f"Find nearest iout={iout} for z={z}")

        if (iout < 0):
            iouts = self.get_iout_avail()
            iout = iouts[iout]
        self.iout = iout

        self.path = join(self.snap_path, output_format.format(snap=self))
        self.params = {}
        self.unitmode = 'code' # 'code', 'physical'
        self.unitfactor = 1

        self.mode = mode
        self.info_path = join(self.path, f'info_{self.iout:05d}.txt')
        if(self.mode=='ng'): self.info_path = join(self.path, f'info.txt')

        self.memory = []
        self.shmprefix = 'rur'
        self.tracer_mem = None
        self.part_mem = None
        self.cell_mem = None
        self.part_data = None
        self.cell_data = None
        self.sink_data = None

        self.part = None
        self.cell = None
        self.sink = None

        self.pcmap = None

        self.longint = longint

        if (self.longint is None):
            if (mode == 'fornax'):
                self.longint = True
            else:
                self.longint = False

        if (mode == 'ng'):
            self.classic_format = False
        else:
            self.classic_format = True
        self.read_params(snap, verbose=self.verbose)

        # set initial box and default box from boxlen
        self.default_box = default_box * self.params['boxlen']
        if box is not None:
            # if box exists as input
            self.box = np.array(box)
        else:
            # if not, set as default box
            self.box = self.default_box

        self.region = BoxRegion(self.box)

        # individual box by type for reducing redundant calculation
        self.box_cell = None
        self.box_part = None
        self.box_sink = None
        self.alert = False

    def terminate(self, signum, frame):
        self.flush(msg=True, parent=f'[Signal{signum}]')
        atexit.unregister(self.flush)
        exit(0)

    def flush(self, msg=False, parent='', verbose=timer.verbose):
        try:
            if (len(self.memory) > 0):
                if (msg or verbose >= 1): print(f"{parent} Clearing memory")
                if (msg or verbose > 1): print(f"  {[i.name for i in self.memory]}")
            self.tracer_mem = None
            self.part_mem = None
            self.cell_mem = None
            while (len(self.memory) > 0):
                try:
                    mem = self.memory.pop()
                    if (msg or verbose >= 1): print(f"\tUnlink `{mem.name}`")
                    mem.close()
                    mem.unlink()
                    del mem
                except:
                    pass
            if (self.alert):
                # atexit.unregister(self.flush)
                self.alert = False
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGPIPE, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
        except:
            pass

    def switch_unitmode(self):
        if (self.unitmode == 'code'):
            self.unitmode = 'physical'
            print("Switched to physical unit mode (kpc and km/s)")
            self.box = self.box/self.unit['kpc']
            self.default_box = self.default_box/self.unit['kpc']
            self.unitfactor = self.unit['kpc']
            if(self.part is not None):
                self.part.snap = self
                for key in dim_keys: self.part_data[key] /= self.unit['kpc']
                for key in vel_keys: self.part_data[key] /= self.unit['km/s']
                self.part_data['m'] /= self.unit['Msol']
                if('m0' in self.part_data.dtype.names): self.part_data['m0'] /= self.unit['Msol']
                if(id(self.part.table) != id(self.part_data)):
                    for key in dim_keys: self.part[key] /= self.unit['kpc']
                    for key in vel_keys: self.part[key] /= self.unit['km/s']
                    self.part['m'] /= self.unit['Msol']
                    if('m0' in self.part.dtype.names): self.part['m0'] /= self.unit['Msol']
            if(self.cell is not None):
                self.cell.snap = self
                for key in dim_keys: self.cell_data[key] /= self.unit['kpc']
                for key in vel_keys: self.cell_data[key] /= self.unit['km/s']
                if(id(self.cell.table) != id(self.cell_data)):
                    for key in dim_keys: self.cell[key] /= self.unit['kpc']
                    for key in vel_keys: self.cell[key] /= self.unit['km/s']
            if(self.sink is not None):
                self.sink.snap = self
                for key in dim_keys: self.sink_data[key] /= self.unit['kpc']
                for key in vel_keys: self.sink_data[key] /= self.unit['km/s']
                self.sink_data['m'] /= self.unit['Msol']
                self.sink_data['dM'] /= self.unit['Msol']
                self.sink_data['dMBH'] /= self.unit['Msol']
                self.sink_data['dMEd'] /= self.unit['Msol']
                if(id(self.sink.table) != id(self.sink_data)):
                    for key in dim_keys: self.sink[key] /= self.unit['kpc']
                    for key in vel_keys: self.sink[key] /= self.unit['km/s']
                    self.sink['m'] /= self.unit['Msol']
                    self.sink['dM'] /= self.unit['Msol']
                    self.sink['dMBH'] /= self.unit['Msol']
                    self.sink['dMEd'] /= self.unit['Msol']
        else:
            self.unitmode = 'code'
            print("Switched to code unit mode")
            self.box = self.box*self.unit['kpc']
            self.default_box = self.default_box*self.unit['kpc']
            self.unitfactor = 1
            if(self.part is not None):
                self.part.snap = self
                for key in dim_keys: self.part_data[key] *= self.unit['kpc']
                for key in vel_keys: self.part_data[key] *= self.unit['km/s']
                self.part_data['m'] *= self.unit['Msol']
                if('m0' in self.part_data.dtype.names): self.part_data['m0'] *= self.unit['Msol']
                if(id(self.part.table) != id(self.part_data)):
                    for key in dim_keys: self.part[key] *= self.unit['kpc']
                    for key in vel_keys: self.part[key] *= self.unit['km/s']
                    self.part['m'] *= self.unit['Msol']
                    if('m0' in self.part.dtype.names): self.part['m0'] *= self.unit['Msol']
            if(self.cell is not None):
                self.cell.snap = self
                for key in dim_keys: self.cell_data[key] *= self.unit['kpc']
                for key in vel_keys: self.cell_data[key] *= self.unit['km/s']
                if(id(self.cell.table) != id(self.cell_data)):
                    for key in dim_keys: self.cell[key] *= self.unit['kpc']
                    for key in vel_keys: self.cell[key] *= self.unit['km/s']
            if(self.sink is not None):
                self.sink.snap = self
                for key in dim_keys: self.sink_data[key] *= self.unit['kpc']
                for key in vel_keys: self.sink_data[key] *= self.unit['km/s']
                self.sink_data['m'] *= self.unit['Msol']
                self.sink_data['dM'] *= self.unit['Msol']
                self.sink_data['dMBH'] *= self.unit['Msol']
                self.sink_data['dMEd'] *= self.unit['Msol']
                if(id(self.sink.table) != id(self.sink_data)):
                    for key in dim_keys: self.sink[key] *= self.unit['kpc']
                    for key in vel_keys: self.sink[key] *= self.unit['km/s']
                    self.sink['m'] *= self.unit['Msol']
                    self.sink['dM'] *= self.unit['Msol']
                    self.sink['dMBH'] *= self.unit['Msol']
                    self.sink['dMEd'] *= self.unit['Msol']


    def __del__(self):
        atexit.unregister(self.flush)
        self.flush(parent='[__del__]')

    def get_iout_avail(self):
        output_names = glob.glob(join(self.snap_path, output_glob))
        iouts = [int(arr[-5:]) for arr in output_names]
        return np.sort(iouts)

    def switch_iout(self, iout):
        # returns to other snapshot while maintaining repository, box, etc.
        return RamsesSnapshot(self.repo, iout, self.mode, self.box, self.path_in_repo, snap=self, longint=self.longint, verbose=self.verbose)

    def __getitem__(self, item):
        try:
            return self.params[item]
        except:
            raise AttributeError(f"Attribute `{item}` not found in snapshot `{repr(self)}`")

    def __getattr__(self, item):
        if item.startswith('__') and item.endswith('__'):
            return super(RamsesSnapshot, self).__getattr__(item)
        return self.__getitem__(item)

    def get_path(self, type='amr', icpu=1):
        if(type == 'namelist'):
            return join(self.path, "namelist.txt")
        if(self.mode=='ng'): return join(self.path, f"{type}.out{icpu:05d}")
        return join(self.path, f"{type}_{self.iout:05d}.out{icpu:05d}")

    def H_over_H0(self, aexp, params=None):
        if (params is None):
            params = self.params
        return np.sqrt(params['omega_m'] * aexp ** -3 + params['omega_l'])

    def set_cosmology(self, params=None, n=5000, snap=None, verbose=timer.verbose):
        # calculates cosmology table with given cosmology paramters
        # unit of conformal time (u) is in function of hubble time at z=0 and aexp
        # unit_t = dt/du = (aexp)**2/H0
        if (params is None):
            params = self.params

        if (snap is None):
            # Integrate manually because astropy cosmology calculation is too slow...
            aarr = np.linspace(0, 1, n)[1:] ** 2
            aarr_st = (aarr[:-1] + aarr[1:]) / 2
            duda = 1. / (aarr_st ** 3 * self.H_over_H0(aarr_st))
            dtda = 1. / (params['H0'] * km * Gyr / Mpc * aarr_st * self.H_over_H0(aarr_st))
            aarr = aarr[1:]

            uarr = cumtrapz(duda[::-1], aarr[::-1], initial=0)[::-1]
            tarr = cumtrapz(dtda, aarr, initial=0)
            self.cosmo_table = fromarrays([aarr, tarr, uarr], dtype=[('aexp', 'f8'), ('t', 'f8'), ('u', 'f8')])
        else:
            self.cosmo_table = snap.cosmo_table

        self.params['age'] = np.interp(params['time'], self.cosmo_table['u'], self.cosmo_table['t'])
        self.params['lookback_time'] = self.cosmo_table['t'][-1] - self.params['age']

        if (verbose >= 1):
            print('[Output %05d] Age (Gyr) : %.3f / %.3f, z = %.5f (a = %.4f)' % (
            self.iout, self.params['age'], self.cosmo_table['t'][-1], params['z'], params['aexp']))

    def interpolate_cosmo_table(self, value, name1, name2):
        return np.interp(value, self.cosmo_table[name1], self.cosmo_table[name2])

    def aexp_to_age(self, aexp):
        return self.interpolate_cosmo_table(aexp, 'aexp', 't')

    def age_to_aexp(self, age):
        return self.interpolate_cosmo_table(age, 't', 'aexp')

    def epoch_to_age(self, epoch):
        return self.interpolate_cosmo_table(epoch, 'u', 't')

    def epoch_to_aexp(self, epoch):
        return self.interpolate_cosmo_table(epoch, 'u', 'aexp')

    def aexp_to_H(self, aexp=None):
        if aexp is None:
            aexp = self.aexp
        return self['H0'] * km * Gyr / Mpc * self.H_over_H0(aexp)

    def aexp_to_dtdu(self, aexp=None):
        # returns dt over du (derivative of proper time t in function of ramses time unit u)
        if aexp is None:
            aexp = self.aexp
        return aexp ** 2 / (self['H0'] * km * Gyr / Mpc)

    def aexp_to_dadt(self, aexp=None):
        # returns da over dt (derivative of aexp in function of proper time t)
        if aexp is None:
            aexp = self.aexp
        return aexp * self.aexp_to_H(aexp)

    def aexp_to_dadu(self, aexp=None):
        # returns da over du (derivative of aexp in function of ramses time unit u)
        if aexp is None:
            aexp = self.aexp
        return aexp ** 3 * self.H_over_H0(aexp)

    def dcrit(self):
        return self.aexp_to_dcrit(self.aexp)

    def aexp_to_dcrit(self, aexp=None):
        # returns critical density in cgs unit using current cosmological parameters
        # assumes flat LCDM only!
        if aexp is None:
            aexp = self.aexp
        return 3 * (self.H0 * km / Mpc) ** 2 / (8 * np.pi * G_const) / aexp ** 3

    def set_unit(self):
        set_custom_units(self)

    def set_box(self, center, extent, unit=None):
        """set center and extent of the current target bounding box of the simulation.
        if unit is None, it is recognized as code unit
        """
        if (unit is not None):
            extent = extent / self.unit[unit]
            center = center / self.unit[unit]
        self.box = get_box(center, extent)
        if (self.box.shape != (3, 2)):
            raise ValueError("Incorrect box shape: ", self.box.shape)
        self.region = BoxRegion(self.box)

    def set_box_halo(self, halo, radius=1, use_halo_radius=True, radius_name='rvir'):
        if (isinstance(halo, np.ndarray)):
            warnings.warn(
                "numpy.ndarray is passed instead of np.void in halo parameter. Assuming first row as input halo...",
                UserWarning)
            halo = halo[0]
        center = get_vector(halo)
        if (use_halo_radius):
            extent = halo[radius_name] * radius * 2
            if (self.unitmode == 'physical'):
                extent /= self.unit['kpc']
        else:
            extent = radius * 2
        self.set_box(center, extent)

    def part_desc(self):
        fname = f"part_file_descriptor.txt"
        chem = []
        if(os.path.exists(f"{self.snap_path}/output_{self.iout:05d}/{fname}")):
            with open(f"{self.snap_path}/output_{self.iout:05d}/{fname}", "r") as f:
                texts = [it.split(',') for it in f.readlines() if(not it.startswith("#"))]
                convert = desc2dtype
                part_dtype = [0]*len(texts)
                for i, it in enumerate(texts):
                    ikey = it[1].strip()
                    dname = convert[ikey] if(ikey in convert.keys()) else ikey
                    part_dtype[i] = (dname, format_f2py[it[2].strip()])
                    if('chem' in ikey): chem.append(part_dtype[i][0])
            part_dtype = part_dtype+[('cpu', 'i4')]
        else:
            if (timer.verbose>0):
                warnings.warn(f"Warning! No `part_file_descriptor.txt` found, using default dtype", UserWarning)
            part_dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'), ('m', 'f8')]
            if(self.star): part_dtype = part_dtype+[('epoch', 'f8'), ('metal', 'f8')]
            if(self.mode == 'hagn'):
                chem = ['H', 'O', 'Fe', 'C', 'N', 'Mg', 'Si']
                part_dtype = part_dtype + [('H', 'f8'), ('O', 'f8'), ('Fe', 'f8'), ('C', 'f8'),
                        ('N', 'f8'), ('Mg', 'f8'), ('Si', 'f8')]
            _i = 'i8' if(self.longint) else 'i4'
            part_dtype = part_dtype+[('id', _i), ('level', 'i4'), ('cpu', 'i4')]
        return part_dtype, chem

    def hydro_desc(self):
        fname = f"hydro_file_descriptor.txt"
        chem = []
        if(os.path.exists(f"{self.snap_path}/output_{self.iout:05d}/{fname}")):
            with open(f"{self.snap_path}/output_{self.iout:05d}/{fname}", "r") as f:
                texts = [it.split(',') for it in f.readlines() if(not it.startswith("#"))]
                convert = desc2dtype
                # Old format of hydro_file_descriptor.txt
                if(texts[0][0][:4] == 'nvar'):
                    nvar = int(texts[0][0].split()[-1])
                    hydro_names = [0]*nvar
                    for i, it in enumerate(texts[1:]):
                        ikey = it[0].split(':')[-1].strip()
                        if(ikey == 'passive_scalar_1'): ikey = 'metal'
                        if(ikey == 'passive_scalar_2'): ikey = 'refmask'
                        dname = convert[ikey] if(ikey in convert.keys()) else ikey
                        hydro_names[i] = dname
                else:
                    hydro_names = [0]*len(texts)
                    for i, it in enumerate(texts):
                        ikey = it[1].strip()
                        dname = convert[ikey] if(ikey in convert.keys()) else ikey
                        hydro_names[i] = dname
                        if('chem' in ikey): chem.append(hydro_names[i])
        else:
            if (timer.verbose>0):
                warnings.warn(f"Warning! No `hydro_file_descriptor.txt` found, using default dtype", UserWarning)
            hydro_names = ['rho', 'vx', 'vy', 'vz', 'P', 'metal', 'refmask']
            veryolds = ['dm_only', 'ng']
            olds = ['hagn','yzics','yzics_dm_only']
            if(self.mode in veryolds): hydro_names = hydro_names[:5]
            if(self.mode in olds): hydro_names = hydro_names[:6]
            if(self.mode == 'hagn'):
                chem = ['H', 'O', 'Fe', 'C', 'N', 'Mg', 'Si']
                hydro_names = hydro_names + chem
        return hydro_names, chem

    def make_shm_name(self, kind):
        now = datetime.datetime.now()
        try: shmprefix = self.shmprefix
        except: shmprefix = 'rur'
        fname = f"{shmprefix}_{kind}_{self.mode.replace('_','')}_u{os.getuid()}_{now.strftime('%Y%m%d_%H%M%S_%f')}"
        count = 0
        while(exists(f"/dev/shm/{fname}")):
            fname = f"{shmprefix}_{kind}_{self.mode.replace('_','')}_u{os.getuid()}_{now.strftime('%Y%m%d_%H%M%S_%f')}r{count}"
            count += 1
        return fname

    def read_params(self, snap, verbose=timer.verbose):

        opened = open(self.info_path, mode='r')

        int_regex = re.compile(r'(?P<name>\w+)\s*=\s*(?P<data>\d+)')
        float_regex = re.compile(r'(?P<name>\w+)\s*=\s*(?P<data>.+)')
        str_regex = re.compile(r'(?P<name>\w+)\s*=\s*(?P<data>.+)')
        domain_regex = re.compile(r'(?P<domain>.+)\s+(?P<ind_min>.+)\s+(?P<ind_max>.+)')

        params = {}
        # read integer data
        line = opened.readline().strip()
        while len(line) > 0:
            matched = int_regex.search(line)
            if (not matched):
                raise ValueError("A line in the info file is not recognized: %s" % line)
            params[matched.group('name')] = int(matched.group('data'))
            line = opened.readline().strip()

        # read float data
        line = opened.readline().strip()
        while len(line) > 0:
            matched = float_regex.search(line)
            if (not matched):
                raise ValueError("A line in the info file is not recognized: %s" % line)
            params[matched.group('name')] = float(matched.group('data'))
            line = opened.readline().strip()

        # some cosmological calculations
        params['unit_m'] = params['unit_d'] * params['unit_l'] ** 3
        params['h'] = params['H0'] / 100
        params['z'] = 1 / params['aexp'] - 1
        params['boxsize'] = params['unit_l'] * params['h'] / Mpc / params['aexp']
        params['boxsize_physical'] = params['boxsize'] / (params['h']) * params['aexp']
        params['boxsize_comoving'] = params['boxsize'] / (params['h'])

        params['icoarse'] = params['nstep_coarse']

        if (self.classic_format):
            # read hilbert key boundaries
            line = opened.readline().strip()
            params['ordering'] = str_regex.search(line).group('data')
            opened.readline()
            if (params['ordering'] == 'hilbert'):
                # reads more precise boundary key by reading amr 1
                amr_filename = self.get_path('amr', 1)
                with FortranFile(amr_filename) as file:
                    for _ in range(25):
                        file.read_record('b')
                    bounds = file.read_record(dtype='b')
                if (bounds.size == 16 * (params['ncpu'] + 1)):
                    # quad case
                    self.bound_key = quad_to_f16(bounds)[1:-1]
                else:
                    # double case
                    self.bound_key = bounds.view('f8')[1:-1]

            if (exists(self.get_path('part', 1))):
                self.params['nstar'] = self._read_nstar()
                self.params['star'] = self.params['nstar'] > 0
            else:
                self.params['nstar'] = 0
                self.params['star'] = False
        else:
            self.params['star'] = True
        
        part_dtype, chem = self.part_desc()
        hydro_names, hchem = self.hydro_desc()
        if(not np.array_equal(chem, hchem)):
            scalars = [it for it in hydro_names if(it[:6]=='scalar')]
            warnings.warn(f"\nChemical elements are not in agreement!\n\t`{chem}` from part\n\t`{hchem}` from hydro", UserWarning, stacklevel=2)
            if(len(scalars) == len(chem)):
                hydro_names = [it if(it not in scalars) else chem[int(it[-2:])-1] for it in hydro_names]
                warnings.warn(f"\n`{scalars}`\nis found in hydro descriptor\nThese will be considered to\n`{chem}`", UserWarning, stacklevel=2)
            else:
                warnings.warn(f"\n`{scalars}`\nis found in hydro descriptor\nThese will be ignored", UserWarning, stacklevel=2)
        self.part_dtype = part_dtype
        self.hydro_names = hydro_names
        self.chem = chem

        # initialize cpu list and boundaries
        self.cpulist_cell = np.array([], dtype='i4')
        self.cpulist_part = np.array([], dtype='i4')
        self.bound_cell = np.array([0], dtype='i4')
        self.bound_part = np.array([0], dtype='i4')
        self.params.update(params)

        self.cell_extra = custom_extra_fields(self, 'cell')
        self.part_extra = custom_extra_fields(self, 'particle')

        self.set_cosmology(snap=snap, verbose=verbose)
        self.set_unit()

    def get_involved_cpu(self, box=None, binlvl=None, n_divide=5):
        """Get the list of involved cpu domain for specific region.

        """
        if (box is None):
            box = self.box
        if (self.classic_format and not box is None):
            box = np.array(box)
            if (self.unitmode != 'code'):
                box = box * self.unit['kpc'] # kpc box -> code box
            maxlvl = self.params['levelmax']

            box_unit = box / self.params['boxlen']
            involved_cpu = get_cpulist(box_unit, binlvl, maxlvl, self.bound_key, self.ndim, n_divide,
                                       ncpu=self.params['ncpu'])
        else:
            involved_cpu = np.arange(self.params['ncpu']) + 1
        return involved_cpu
    
    def extract_header(self):
        fname = f"header_{self.iout:05d}.txt"
        header = {'total':0,
                'dm':0,
                'star':0,
                'sink':0,
                'cloud':0,
                'tracer':0,
                'other_tracer':0,
                'debris_tracer':0,
                'cloud_tracer':0,
                'star_tracer':0,
                'gas_tracer':0,
                'debris':0,
                'other':0,
                'undefined':0
                }
        if(os.path.exists(f"{self.snap_path}/output_{self.iout:05d}/{fname}")):
            with open(f"{self.snap_path}/output_{self.iout:05d}/{fname}", "r") as f:
                texts = [it.strip().split() for it in f.readlines() if(not it.startswith("#"))and(len(it)==24)]
            if(len(texts)==0): # NewHorizon Case
                with open(f"{self.snap_path}/output_{self.iout:05d}/{fname}", "r") as f:
                    texts = [it.strip() for it in f.readlines()]
                header['total'] = int(texts[1])
                header['dm'] = int(texts[3])
                header['star'] = int(texts[5])
                header['sink'] = int(texts[7])
                header['cloud'] = int(texts[7])*2109
            else: # Normal Case
                for it in texts:
                    if(len(it)==1):
                        s=it[0]; family = s.rstrip('0123456789'); count = int(s[len(family):])
                    else:
                        family, count = it[0], int(it[1])
                    if(family in header.keys()): header[family] += count
                    if('tracer' in family): header['tracer'] += count
                    header['total'] += count
                    header[family.lower()] = count
                header['sink'] = int(header['cloud']/2109)
        else:
            warnings.warn(f"Warning! No `{fname}` found.", UserWarning)
        return header

    def read_sink_table(self):
        if (self.mode == 'nh'):
            table = np.genfromtxt(self.path + '/sink_%05d.csv' % self.iout, dtype=sink_table_dtype, delimiter=',')
        else:
            raise ValueError('This function works only for NH-version RAMSES')
        return table

    def read_part_py(self, pname:str, cpulist:Iterable, target_fields:Iterable=None, nthread=1, use_cache=True):
        part_dtype, chem = self.part_dtype, self.chem
        allfiles = glob.glob(f"{self.snap_path}/output_{self.iout:05d}/part*out*")
        files = [fname for fname in allfiles if int(fname[-5:]) in cpulist]
        files.sort()

        sequential = nthread == 1
        isstar = self.star
        isfamily = 'family' in [p[0] for p in part_dtype]

        header = self.extract_header()
        # If the total number of particles is zero, we need to calculate it.
        if(header['total']==0)and(sequential):
            with FortranFile(f"{allfiles[0]}", mode='r') as f:
                f.skip_records(4); header['star'] = f.read_ints(np.int32)[0]
                f.skip_records(2); header['sink'] = f.read_ints(np.int32)[0]
                header['cloud'] = 2109 * header['sink']
            header['dm'] = 0
            header['tracer'] = 0
            if (pname == 'dm') or (pname is None):
                header['total'] = 0
                for fname in allfiles:
                    if (pname is None) and (not int(fname[-5:]) in cpulist): continue
                    with FortranFile(f"{fname}", mode='r') as f:
                        f.skip_records(2)
                        header['total'] += f.read_ints(np.int32)[0]
                header['DM'] = header['total'] - header['star'] - header['cloud']

        dtype = part_dtype
        if target_fields is not None:
            if ('cpu' not in target_fields): target_fields = np.append(target_fields, 'cpu')
            dtype = [idtype for idtype in dtype if idtype[0] in target_fields]
        else:
            target_fields = [idtype[0] for idtype in dtype]
        field_indicies = ['x','y','z','vx','vy','vz','m','id','level','family','tag','epoch','metal','m0','rho0','partp']
        where = np.where(np.isin(field_indicies, target_fields))[0]
        ndeep = np.max(where)+1 if(len(where)>0) else 0
        kwargs = {
            "pname": pname, "isfamily": isfamily, "isstar": isstar, "chem": chem, "part_dtype": part_dtype,
            "target_fields": target_fields, "dtype": dtype, "ndeep":ndeep, "repo":self.repo, "use_cache":use_cache}

        if (timer.verbose > 0):
            print("\tAllocating Memory...", end=' ')
            ref = time.time()
        if (sequential):
            tracers = ["tracer", "cloud_tracer", "star_tracer", "gas_tracer"]
            if(pname is None):
                size = header['total']
            elif(pname in tracers):
                size = header['tracer']
            elif(pname=='sink'):
                size = header['cloud']
            else:
                size = header[pname]
            part = np.empty(size, dtype=dtype)
            if (timer.verbose > 0): print(f"\tDone ({time.time() - ref:.3f} sec) -> {size} particles")
            if (size == 0): return part, None
        else:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            with Pool(processes=nthread) as pool:
                results = pool.starmap(_calc_npart, [(fname, kwargs) for fname in files])
            results = np.asarray(results, dtype=[("mask", object), ("size", int), ("icpu", int)])
            signal.signal(signal.SIGTERM, self.terminate)
            argsort = np.argsort(results['icpu'])
            results = results[argsort]
            sizes = results['size']
            masks = results['mask']
            size = np.sum(sizes)
            cursors = np.cumsum(sizes) - sizes
            part = np.empty(size, dtype=dtype)
            if (size == 0): return part, None
            if (not self.alert):
                atexit.register(self.flush, msg=True, parent='[Auto]')
                signal.signal(signal.SIGINT, self.terminate)
                signal.signal(signal.SIGPIPE, self.terminate)
                self.alert = True
            self.part_mem = shared_memory.SharedMemory(name=self.make_shm_name('part'),create=True, size=part.nbytes)
            self.memory.append(self.part_mem)
            part = np.ndarray(part.shape, dtype=np.dtype(dtype), buffer=self.part_mem.buf)
        if (timer.verbose > 0): print(f"\tDone ({time.time() - ref:.3f} sec) -> {part.shape[0]} particles")

        # 5) Read output part files
        if (sequential):
            cursor = 0
            iterobj = tqdm(files, desc=f"Reading parts") if (timer.verbose >= 1) else files
            for fname in iterobj:
                cursor = _read_part(fname, kwargs, part=part, mask=None, nsize=None, cursor=cursor,
                                    address=None, shape=None)
            part = part[:cursor]
        else:
            if(timer.verbose>=1):
                pbar = tqdm(total=len(files), desc=f"Reading parts")
                def update(*a):
                    pbar.update()
            else:
                update = None   
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            with Pool(processes=nthread) as pool:
                async_result = [pool.apply_async(_read_part, (
                fname, kwargs, None, mask, size, cursor, self.part_mem.name, part.shape), callback=update) for
                                fname, mask, size, cursor in zip(files, masks, sizes, cursors)]
                # iterobj = tqdm(async_result, total=len(async_result), desc=f"Reading parts") if (
                #             timer.verbose >= 1) else async_result
                iterobj = async_result
                for r in iterobj:
                    r.get()
            signal.signal(signal.SIGTERM, self.terminate)
        if(sequential): return part, None
        return part, np.append(cursors, size)

    def read_part(self, target_fields=None, cpulist=None, pname=None, nthread=8, python=True, use_cache=True):
        """Reads particle data from current box.

        Parameters
        ----------
        target_fields: list of string
            target field name to read. If None is passed, read all fields.
        cpulist: list of integer
            target list of cpus, if specified, read selected cpus regardless of current box.
        Returns
        -------
        part : Particle object
            particle data, can be accessed as attribute also.

        """
        if (cpulist is None):
            cpulist = self.get_involved_cpu()
        if (self.part is not None):
            if not isinstance(self.part, tuple):
                if pname == self.part.ptype:
                    if (timer.verbose >= 1): print('Searching for extra files...')
                    cpulist = np.array(cpulist)[np.isin(cpulist, self.cpulist_part, assume_unique=True, invert=True)]

        if (cpulist.size > 0):
            filesize = 0
            for icpu in cpulist:
                filesize += getsize(self.get_path('part', icpu))
            timer.start(
                'Reading %d part files (%s) in %s... ' % (cpulist.size, utool.format_bytes(filesize), self.path), 1)
            nthread = min(nthread, cpulist.size)
            if (python):
                self.clear_shm(clean=False)
                part, bound = self.read_part_py(pname, cpulist, target_fields=target_fields, nthread=nthread, use_cache=use_cache)
            else:
                bound = None
                progress_bar = cpulist.size > progress_bar_limit and timer.verbose >= 1
                mode = self.mode
                if mode == 'nc':
                    mode = 'y4'
                readr.read_part(self.snap_path, self.iout, cpulist, mode, progress_bar, self.longint, nthread)
                timer.record()

                timer.start('Building table for %d particles... ' % readr.integer_table.shape[1], 1)
                dtype = self.part_dtype
                if (target_fields is not None):
                    if ('cpu' not in target_fields):
                        target_fields = np.append(target_fields, 'cpu')
                    if (self.longint):
                        arr = [*readr.real_table, *readr.long_table, *readr.integer_table, *readr.byte_table]
                    else:
                        arr = [*readr.real_table, *readr.integer_table, *readr.byte_table]

                    target_idx = np.where(np.isin(np.dtype(dtype).names, target_fields))[0]
                    arr = [arr[idx] for idx in target_idx]
                    dtype = [dtype[idx] for idx in target_idx]
                    ids, epoch, m, family = None, None, None, None
                    family = arr[np.where(np.array(target_fields) == 'family')[0][0]] if (
                                'family' in target_fields) else None
                    if (family is None):
                        ids = arr[np.where(np.array(target_fields) == 'id')[0][0]] if ('id' in target_fields) else None
                        if (self.star):
                            epoch = arr[np.where(np.array(target_fields) == 'epoch')[0][0]] if (
                                        'epoch' in target_fields) else None
                        if (pname != 'dm') and (pname != 'star'):
                            m = arr[np.where(np.array(target_fields) == 'm')[0][0]] if ('m' in target_fields) else None
                    mask, _ = _classify(pname, len(arr[0]), ids=ids, epoch=epoch, m=m, family=family, isstar=self.star)
                    if (pname is not None): arr = [iarr[mask] for iarr in arr]
                    part = fromarrays(arr, dtype=dtype)
                else:
                    if (self.longint):
                        arrs = [readr.real_table.T, readr.long_table.T, readr.integer_table.T, readr.byte_table.T]
                    else:
                        arrs = [readr.real_table.T, readr.integer_table.T, readr.byte_table.T]

                    ids, epoch, m, family = None, None, None, None
                    family = arrs[-1][:, 0] if ('family' in np.dtype(dtype).names) else None
                    if (family is None):
                        names = {'epoch': None, 'id': None, 'm': None}
                        if (not self.star): del names['epoch']
                        if (pname == 'dm') or (pname == 'star'): del names['m']
                        for key in list(names.keys()):
                            idx = np.where(np.array(np.dtype(dtype).names) == key)[0][0]
                            # Real table
                            i = 0
                            # Long/Int table
                            if (idx + 1 > arrs[0].shape[1]):
                                idx -= arrs[0].shape[1]
                                i += 1
                                # Int/byte
                                if (idx + 1 > arrs[1].shape[1]):
                                    idx -= arrs[1].shape[1]
                                    i += 1
                                    # byte
                                    if (idx + 1 > arrs[2].shape[1]):
                                        idx -= arrs[2].shape[1]
                                        i += 1
                            names[key] = arrs[i][:, idx]
                        ids = names.pop('id', None)
                        epoch = names.pop('epoch', None)
                        m = names.pop('m', None)
                    mask, _ = _classify(pname, arrs[0].shape[0], ids=ids, epoch=epoch, m=m, family=family, isstar=self.star)
                    if (pname is not None): arrs = [arr[mask] for arr in arrs]

                    # re-order dtype to match with readr output
                    fdtype = [idt for idt in dtype if 'f' in idt[1]]
                    ldtype = [idt for idt in dtype if 'i8' in idt[1]]
                    idtype = [idt for idt in dtype if 'i4' in idt[1]]
                    bdtype = [idt for idt in dtype if 'i1' in idt[1]]
                    dtype = fdtype + ldtype + idtype + bdtype

                    part = fromndarrays(arrs, dtype)
                readr.close()
            timer.record()
            if(self.unitmode != 'code'):
                for key in dim_keys: part[key] /= self.unit['kpc']
                for key in vel_keys: part[key] /= self.unit['km/s']
                part['m'] /= self.unit['Msol']
                if('m0' in part.dtype.names): part['m0'] /= self.unit['Msol']
            if(bound is None):
                timer.start('Compute boundary on cpumap... ', 1)
                bound = compute_boundary(part['cpu'], cpulist)
                timer.record()
            if (self.part_data is None):
                self.part_data = part
            else:
                self.part_data = np.concatenate([self.part_data, part])

            self.bound_part = np.concatenate([self.bound_part[:-1], self.bound_part[-1] + bound])
            self.cpulist_part = np.concatenate([self.cpulist_part, cpulist])

        else:
            if (timer.verbose >= 1):
                print('CPU list already satisfied.')

    def read_cell_py(self, cpulist: Iterable, target_fields: Iterable = None, nthread: int = 8, read_grav: bool = False, use_cache=True):
        # 1) Read AMR params
        sequential = nthread == 1   
        fname = self.get_path('amr', 1)
        with FortranFile(fname, mode='r') as f:
            ncpu, = f.read_ints()
            ndim, = f.read_ints()
            nx, ny, nz, = f.read_ints()
            nlevelmax, = f.read_ints()
            f.skip_records(1)
            nboundary, = f.read_ints()
            f.skip_records(1)
            boxlen, = f.read_reals()

        # measures x, y, z offset based on the boundary condition
        # does not work if boundaries are asymmetric, which can only be determined in namelist
        coarse_min = [0, 0, 0]
        key = ['i', 'j', 'k']
        nxyz = [nx, ny, nz]
        if nboundary > 0:
            for i in range(ndim):
                if nxyz[i] == 3:
                    coarse_min[i] += 1
                if nxyz[i] == 2:
                    nml = self.read_namelist()
                    if len(nml) == 0:
                        warnings.warn("Assymetric boundaries detected, which cannot be determined without namelist file. \
                                      Move namelist.txt file to the output directory or manually apply offset to the cell position.")
                    else:
                        bound_min = np.array(str_to_tuple(nml['BOUNDARY_PARAMS']['%sbound_min' % key[i]]))
                        bound_max = np.array(str_to_tuple(nml['BOUNDARY_PARAMS']['%sbound_max' % key[i]]))
                        if np.any(((bound_min * bound_max) == 1) & bound_min == -1):
                            coarse_min += 1
        icoarse_min, jcoarse_min, kcoarse_min = tuple(coarse_min)

        amr_kwargs = {
            'nboundary': nboundary, 'nlevelmax': nlevelmax, 'ndim': ndim,
            'ncpu': ncpu, 'twotondim': 2 ** ndim, 'skip_amr': 3 * (2 ** ndim + ndim) + 1,
            'nx': nx, 'ny': ny, 'nz': nz, 'boxlen': boxlen,
            'icoarse_min': icoarse_min, 'jcoarse_min': jcoarse_min, 'kcoarse_min': kcoarse_min,}

        # 2) Read Hydro params
        fname = self.get_path('hydro', 1)
        with FortranFile(fname, mode='r') as f:
            f.skip_records(1)
            nhvar, = f.read_ints()

        # 3) Set dtype
        self.params['nhvar'] = nhvar
        formats = ['f8'] * self.params['ndim'] + ['f8'] * self.params['nhvar'] + ['i4'] * 2
        names = list(dim_keys[:self.params['ndim']]) + self.hydro_names + ['level', 'cpu']
        if (read_grav):
            formats.insert(-2, "f8")
            names.insert(-2, "pot")
        if target_fields is not None:
            if 'cpu' not in target_fields:
                target_fields = np.append(target_fields, 'cpu')
            if 'level' not in target_fields:
                target_fields = np.append(target_fields, 'level')
            target_idx = np.where(np.isin(names, target_fields))[0]
            formats = [formats[idx] for idx in target_idx]
            names = [names[idx] for idx in target_idx]
            dtype = [(nm, fmt) for nm, fmt in zip(names, formats)]
        else:
            if(np.__version__ >= '2.0.0'):
                dtype = np.rec.format_parser(formats=formats, names=names, titles=None).dtype
            else:
                dtype = np.format_parser(formats=formats, names=names, titles=None).dtype

        # 4) Calculate total number of cells
        if (timer.verbose > 0):
            print("\tAllocating Memory...", end=' ')
            ref = time.time()
        sizes = None
        if (sequential):
            ncell_tot = 0
            if(use_cache):
                if(not exists(f"{self.repo}/cache/output_{self.iout:05d}/ncells.pkl")):
                    given_cpulist = cpulist
                    cpulist = np.arange(self.ncpu)+1
                else:
                    sizes = load(f"{self.repo}/cache/output_{self.iout:05d}/ncells.pkl", msg=False)[cpulist-1]
            if(sizes is None):
                sizes = np.zeros(len(cpulist), dtype=np.int32)
                for i, icpu in enumerate(cpulist):
                    fname = f"{self.snap_path}/output_{self.iout:05d}/amr_{self.iout:05d}.out{icpu:05d}"
                    sizes[i] = _calc_ncell(fname, amr_kwargs)
                if(use_cache):
                    if(not exists(f"{self.repo}/cache/output_{self.iout:05d}/ncells.pkl")):
                        if(not exists(f"{self.repo}/cache")):
                            try: os.makedirs(f"{self.repo}/cache")
                            except: pass
                        dump(sizes, f"{self.repo}/cache/output_{self.iout:05d}/ncells.pkl", msg=False)
                        cpulist = given_cpulist
                        sizes = sizes[cpulist-1]
            ncell_tot = np.sum(sizes)
            cell = np.empty(ncell_tot, dtype=dtype)
            if (timer.verbose > 0): print(f"\tDone ({time.time() - ref:.3f} sec) -> {ncell_tot} cells")
        else:
            if(use_cache):
                if(not exists(f"{self.repo}/cache/output_{self.iout:05d}/ncells.pkl")):
                    given_cpulist = cpulist
                    cpulist = np.arange(self.ncpu)+1
                else:
                    sizes = load(f"{self.repo}/cache/output_{self.iout:05d}/ncells.pkl", msg=False)[cpulist-1]
            if(sizes is None):
                files = [f"{self.snap_path}/output_{self.iout:05d}/amr_{self.iout:05d}.out{icpu:05d}" for icpu in cpulist]
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                with Pool(processes=nthread) as pool:
                    sizes = pool.starmap(_calc_ncell, [(fname, amr_kwargs) for fname in files])
                signal.signal(signal.SIGTERM, self.terminate)
                sizes = np.asarray(sizes, dtype=np.int32)
                if(use_cache):
                    if(not exists(f"{self.repo}/cache/output_{self.iout:05d}/ncells.pkl")):
                        if(not exists(f"{self.repo}/cache")):
                            try: os.makedirs(f"{self.repo}/cache")
                            except: pass
                        dump(sizes, f"{self.repo}/cache/output_{self.iout:05d}/ncells.pkl", msg=False)
                        cpulist = given_cpulist
                        sizes = sizes[cpulist-1]
            cursors = np.cumsum(sizes) - sizes
            cell = np.empty(np.sum(sizes), dtype=dtype)
            if (not self.alert):
                atexit.register(self.flush, msg=True, parent='[Auto]')
                signal.signal(signal.SIGINT, self.terminate)
                signal.signal(signal.SIGPIPE, self.terminate)
                self.alert = True
            self.cell_mem = shared_memory.SharedMemory(name=self.make_shm_name('cell'), create=True, size=cell.nbytes)
            self.memory.append(self.cell_mem)
            cell = np.ndarray(cell.shape, dtype=np.dtype(dtype), buffer=self.cell_mem.buf)
        if (timer.verbose > 0): print(f"\tDone ({time.time() - ref:.3f} sec) -> {cell.shape[0]} cells")

        snap_kwargs = {
            'nhvar': nhvar, 'hydro_names': self.hydro_names, 'repo': self.snap_path, 'iout': self.iout,
            'skip_hydro': nhvar * 2 ** ndim, 'read_grav': read_grav, 'dtype': dtype, 'names': names}
        # 5) Read data
        if (sequential):
            cursor = 0
            iterobj = tqdm(enumerate(cpulist), total=len(cpulist), desc=f"Reading cells") if (
                        timer.verbose >= 1) else enumerate(cpulist)
            for i, icpu in iterobj:
                cursor = _read_cell(icpu, snap_kwargs, amr_kwargs, cell=cell, nsize=sizes[i], cursor=cursor,
                                    address=None, shape=None)
            cell = cell[:cursor]
        else:
            if(timer.verbose>=1):
                pbar = tqdm(total=len(cpulist), desc=f"Reading cells")
                def update(*a):
                    pbar.update()
            else:
                update = None
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            with Pool(processes=nthread) as pool:
                async_result = [pool.apply_async(_read_cell, (
                icpu, snap_kwargs, amr_kwargs, None, size, cursor, self.cell_mem.name, cell.shape), callback=update) for
                                icpu, size, cursor in zip(cpulist, sizes, cursors)]
                # iterobj = tqdm(async_result, total=len(async_result), desc=f"Reading cells") if (
                #             timer.verbose >= 1) else async_result
                iterobj = async_result
                for r in iterobj:
                    r.get()
            signal.signal(signal.SIGTERM, self.terminate)
        if(sequential): return cell, None
        return cell, np.append(cursors, np.sum(sizes))

    def read_cell(self, target_fields=None, read_grav=False, cpulist=None, python=True, nthread=8, use_cache=True):
        """Reads amr data from current box.

        Parameters
        ----------
        target_fields: list of string
            target field name to read. If None is passed, read all fields.
        cpulist: list of integer
            target list of cpus, if specified, read selected cpus regardless of current box.
        Returns
        -------
        cell : Cell object
            amr data, can be accessed as attribute also.

        """
        if (cpulist is None):
            cpulist = self.get_involved_cpu()
        else:
            cpulist = np.array(cpulist)
        if (self.cell_data is not None):
            if (timer.verbose >= 1):
                print('Searching for extra files...')
            cpulist = cpulist[np.isin(cpulist, self.cpulist_cell, assume_unique=True, invert=True)]

        if (cpulist.size > 0):
            filesize = 0
            for icpu in cpulist:
                filesize += getsize(self.get_path('amr', icpu))
                filesize += getsize(self.get_path('hydro', icpu))
            timer.start(
                'Reading %d AMR & hydro files (%s) in %s... ' % (cpulist.size, utool.format_bytes(filesize), self.path),
                1)
            nthread = min(nthread, cpulist.size)
            if (python):
                self.clear_shm(clean=False)
                cell, bound = self.read_cell_py(cpulist, read_grav=read_grav, nthread=nthread, target_fields=target_fields, use_cache=use_cache)
            else:
                bound=None
                if (nthread > 1):
                    warnings.warn(
                        f"\n[read_cell] In Fortran mode, \nmulti-threading is usually slower than single-threading\nunless there are lots of hydro variables!",
                        UserWarning)
                progress_bar = cpulist.size > progress_bar_limit and timer.verbose >= 1
                readr.read_cell(self.snap_path, self.iout, cpulist, self.mode, read_grav, progress_bar, nthread)
                self.params['nhvar'] = int(readr.nhvar)
                timer.record()

                formats = ['f8'] * self.params['ndim'] + ['f8'] * self.params['nhvar'] + ['i4'] * 2
                names = list(dim_keys[:self.params['ndim']]) + self.hydro_names + ['level', 'cpu']
                if (read_grav):
                    formats.insert(-2, "f8")
                    names.insert(-2, "pot")

                arr = [*readr.real_table, *readr.integer_table]

                if (len(arr) != len(names)):
                    raise ValueError(
                        "Number of fields and size of the hydro array does not match\n"
                        "Consider changing the content of RamsesSnapshot.hydro_names\n"
                        "Received: %d, Desired: %d" % (len(names), len(arr)))

                timer.start('Building table for %d cells... ' % (arr[0].size), 1)
                if target_fields is not None:
                    target_idx = np.where(np.isin(names, target_fields))[0]
                    arr = [arr[idx] for idx in target_idx]
                    formats = [formats[idx] for idx in target_idx]
                    names = [names[idx] for idx in target_idx]
                    dtype = [(nm, fmt) for nm, fmt in zip(names, formats)]
                    cell = fromarrays(arr, formats=formats, names=names)
                else:
                    dtype = np.format_parser(formats=formats, names=names, titles=None).dtype
                    arrs = [readr.real_table.T, readr.integer_table.T]
                    cell = fromndarrays(arrs, dtype)
                readr.close()
            timer.record()
            if(self.unitmode != 'code'):
                for key in dim_keys: cell[key] /= self.unit['kpc']
                for key in vel_keys: cell[key] /= self.unit['km/s']
            if(bound is None):
                timer.start('Compute boundary on cpumap... ', 1)
                bound = compute_boundary(cell['cpu'], cpulist)
                timer.record()
            if (self.cell_data is None):
                self.cell_data = cell
            else:
                self.cell_data = np.concatenate([self.cell_data, cell])

            self.bound_cell = np.concatenate([self.bound_cell[:-1], self.bound_cell[-1] + bound])
            self.cpulist_cell = np.concatenate([self.cpulist_cell, cpulist])

        else:
            if (timer.verbose >= 1):
                print('CPU list already satisfied.')

    def read_ripses(self, target_fields=None, cpulist=None):
        """Reads ripses output data from current box.

        Parameters
        ----------
        target_fields: list of string
            target field name to read. If None is passed, read all fields.
        cpulist: list of integer
            target list of cpus, if specified, read selected cpus regardless of current box.
        Returns
        -------
        cell : Cell object
            amr data, can be accessed as attributes also.

        """
        if (cpulist is None):
            cpulist = self.get_involved_cpu()
        else:
            cpulist = np.array(cpulist)
        if (self.cell_data is not None):
            if (timer.verbose >= 1):
                print('Searching for extra files...')
            cpulist = cpulist[np.isin(cpulist, self.cpulist_cell, assume_unique=True, invert=True)]

        if (cpulist.size > 0):
            timer.start('Reading %d grid files in %s... ' % (cpulist.size, self.path), 1)

            io_ramses.read_ripses_cell(self.snap_path, self.iout, cpulist)
            self.params['nhvar'] = int(io_ramses.nvar)
            timer.record()

            formats = ['f8'] * self.params['ndim'] + ['f8'] * self.params['nhvar'] + ['i4'] * 2

            names = list(dim_keys[:self.params['ndim']]) + self.hydro_names + ['level', 'cpu']

            arr = [*io_ramses.xc.T, *io_ramses.uc.T, io_ramses.lvlc, io_ramses.cpuc]

            if (len(arr) != len(names)):
                io_ramses.close()
                raise ValueError(
                    "Number of fields and size of the hydro array does not match\n"
                    "Consider changing the content of RamsesSnapshot.hydro_names\n"
                    "nvar = %d, recieved: %d" % (len(names), len(arr)))

            if target_fields is not None:
                target_idx = np.where(np.isin(names, target_fields))[0]
                arr = [arr[idx] for idx in target_idx]
                formats = [formats[idx] for idx in target_idx]
                names = [names[idx] for idx in target_idx]

            timer.start('Building table for %d cells... ' % (arr[0].size), 1)
            cell = fromarrays(arr, formats=formats, names=names)
            io_ramses.close()

            bound = compute_boundary(cell['cpu'], cpulist)
            if (self.cell_data is None):
                self.cell_data = cell
            else:
                self.cell_data = np.concatenate([self.cell_data, cell])

            self.bound_cell = np.concatenate([self.bound_cell[:-1], self.bound_cell[-1] + bound])
            self.cpulist_cell = np.concatenate([self.cpulist_cell, cpulist])
            timer.record()

        else:
            if (timer.verbose >= 1):
                print('CPU list already satisfied.')

    def read_sink(self):
        if (self.sink_data is not None and timer.verbose >= 1):
            print('Sink data already loaded.')
        # since sink files are composed of identical data, we read number 1 only.
        filesize = 0
        cpulist = [1]
        for icpu in cpulist:
            if(os.path.exists(self.get_path('sink', icpu))):
                filesize += getsize(self.get_path('sink', icpu))
        if(filesize==0):
            self.sink_data = np.empty(0, dtype=sink_dtype)
        else:
            timer.start('Reading a sink file (%s) in %s... ' % (utool.format_bytes(filesize), self.path), 1)
            readr.read_sink(self.snap_path, self.iout, cpulist, self.levelmin, self.levelmax)
            arr = [*readr.integer_table, *readr.real_table[:19]]
            sink = fromarrays(arr, sink_dtype)
            if(self.unitmode != 'code'):
                for key in dim_keys: sink[key] /= self.unit['kpc']
                for key in vel_keys: sink[key] /= self.unit['km/s']
                for key in ['m','dM','dMBH','dMEd']: sink[key] /= self.unit['Msol']
            self.sink_data = sink
            timer.record()

    def read_sink_raw(self, icpu, path_in_repo='snapshots'):
        # reads sink file as raw array including sink_stat
        path = join(self.repo, path_in_repo)
        readr.read_sink(path, self.iout, icpu, self.levelmin, self.levelmax)
        arr = [readr.integer_table.T, readr.real_table.T]
        readr.close()
        return arr

    def search_sinkprops(self, path_in_repo='SINKPROPS'):
        sinkprop_regex = re.compile(r'sink_\s*(?P<icoarse>\d+).dat')
        path = join(self.repo, path_in_repo)
        sinkprop_names = glob.glob(join(path, sinkprop_glob))
        if (timer.verbose >= 1):
            print('Found %d sinkprop files' % len(sinkprop_names))

        icoarses = []
        for name in sinkprop_names:
            matched = sinkprop_regex.search(name)
            icoarses.append(int(matched.group('icoarse')))
        return np.array(icoarses)

    def check_sinkprop(self, path_in_repo='SINKPROPS', icoarse=None, max_icoarse_offset=1):
        if (icoarse is None):
            icoarses = self.search_sinkprops(path_in_repo)
            icoarse = icoarses[np.argmin(np.abs((self.nstep_coarse) - icoarses))]
            if (icoarse != self.nstep_coarse):
                if not np.abs(icoarse - self.nstep_coarse) > max_icoarse_offset:
                    warnings.warn(
                        'Targeted SINKPROP file not found with icoarse = %d\nFile with icoarse = %d is loaded instead.' % (
                        self.nstep_coarse, icoarse))
        path = join(self.repo, path_in_repo)
        check = join(path, sinkprop_format.format(icoarse=icoarse))
        if (not exists(check)):
            raise FileNotFoundError('Sinkprop file not found: %s' % check)
        return path, icoarse

    def read_sinkprop_info(self, path_in_repo='SINKPROPS', icoarse=None, max_icoarse_offset=1):
        # reads header information from sinkprops file returns as dict
        info = dict()
        path, icoarse = self.check_sinkprop(path_in_repo=path_in_repo, icoarse=icoarse,
                                            max_icoarse_offset=max_icoarse_offset)
        filename = join(path, sinkprop_format.format(icoarse=icoarse))
        with FortranFile(filename) as file:
            info['nsink'] = file.read_ints()
            info['ndim'] = file.read_ints()
            info['aexp'] = file.read_reals()
            info['unit_l'] = file.read_reals()
            info['unit_d'] = file.read_reals()
            info['unit_t'] = file.read_reals()
        return info

    def read_sinkprop(self, path_in_repo='SINKPROPS', icoarse=None, drag_part=True, max_icoarse_offset=1,
                      raw_data=False, return_aexp=False):
        """Reads single sinkprop file from given coarse step number,
        if icoarse not specified, reads the step number of current snapshot
        if file is not found, tries to search for sinkprop file with nearest step number (up to max_icoarse_offset)
        """
        if (drag_part):
            dtype = sink_prop_dtype_drag
        else:
            dtype = sink_prop_dtype
        if (self.mode == 'fornax'):
            dtype = sink_prop_dtype_drag_fornax
        if (self.mode == 'y2' or self.mode == 'y3' or self.mode == 'y4' or self.mode == 'nc' or self.mode == 'y5'):
            dtype = sink_prop_dtype_drag_y2

        path, icoarse = self.check_sinkprop(path_in_repo=path_in_repo, icoarse=icoarse,
                                            max_icoarse_offset=max_icoarse_offset)
        if path is None:
            warnings.warn(
                'Targeted SINKPROP file not found with icoarse = %d\nEmpty array will be loaded.' % (
                    self.nstep_coarse))
            return np.empty((0,), dtype=dtype)
        readr.read_sinkprop(path, icoarse, drag_part, self.mode)
        arrs = [readr.integer_table.T, readr.real_table.T]

        timer.start('Building table for %d smbhs... ' % arrs[0].shape[0], 1)
        if (raw_data):
            return arrs
        if (arrs[0].shape[1] + arrs[1].shape[1] != len(dtype)):
            readr.close()
            raise ValueError('Number of fields mismatch\n'
                             'Received: %d, Allocated: %d' % (arrs[0].shape[1] + arrs[1].shape[1], len(dtype)))
        sink = fromndarrays(arrs, dtype=dtype)
        if (self.unitmode != 'code'):
            for key in dim_keys: sink[key] /= self.unit['kpc']
            for key in vel_keys: sink[key] /= self.unit['km/s']
        timer.record()
        aexp = np.copy(readr.aexp)
        readr.close()

        if (return_aexp):
            return sink, aexp
        else:
            return sink

    def read_sinkprops(self, path_in_repo='SINKPROPS', drag_part=True, use_cache=False, cache_name='sinkprops.pkl',
                       reset_cache=False, cache_format='pkl', progress=False):
        """Searches and reads all files in the sinkprops directory and returns as single table.
        if use_cache=True, it saves pickle file to save reading time
        """
        icoarses = self.search_sinkprops(path_in_repo)
        path = join(self.repo, path_in_repo)

        cache = None
        if (use_cache and not reset_cache):
            cache_file = join(path, cache_name)
            if (exists(cache_file)):
                cache = utool.load(cache_file, format=cache_format)
                icoarses = icoarses[~np.isin(icoarses, np.unique(cache['icoarse']))]
                if (icoarses.size == 0):
                    print('Found cached file: %s' % cache_file)
                    return cache

        int_table = []
        real_table = []
        nsinks = []
        aexps = []

        timer.start('Reading files...')
        if (progress):
            iterator = tqdm(icoarses)
        else:
            iterator = icoarses
        for icoarse in iterator:
            readr.read_sinkprop(path, icoarse, drag_part, self.mode)
            nsink = readr.integer_table.shape[1]
            nsinks.append(nsink)

            int_table.append(np.copy(readr.integer_table.T))
            real_table.append(np.copy(readr.real_table.T))
            aexps.append(np.copy(readr.aexp))

        int_table = np.concatenate(int_table)
        real_table = np.concatenate(real_table)

        aexp_table = np.repeat(aexps, nsinks)
        icoarse_table = np.repeat(icoarses, nsinks)

        arrs = [int_table, real_table]

        if (drag_part):
            dtype = sink_prop_dtype_drag
        else:
            dtype = sink_prop_dtype
        timer.record()

        if (self.mode == 'nh'):
            dtype = sink_prop_dtype
        if (self.mode == 'fornax'):
            dtype = sink_prop_dtype_drag_fornax
        if (self.mode == 'y2' or self.mode == 'y3' or self.mode == 'y4' or self.mode == 'nc' or self.mode == 'y5'):
            dtype = sink_prop_dtype_drag_y2
        if (arrs[0].shape[1] + arrs[1].shape[1] != len(dtype)):
            readr.close()
            raise ValueError('Number of fields mismatch\n'
                             'Received: %d, Allocated: %d' % (arrs[0].shape[1] + arrs[1].shape[1], len(dtype)))

        timer.start('Building table for %d smbhs...' % arrs[0].shape[0])
        sink = fromndarrays(arrs, dtype=dtype)
        if (self.unitmode != 'code'):
            for key in dim_keys: sink[key] /= self.unit['kpc']
            for key in vel_keys: sink[key] /= self.unit['km/s']
            sink['m']
        sink = append_fields(sink, ['aexp', 'icoarse'], [aexp_table, icoarse_table], usemask=False)

        timer.record()
        readr.close()
        if cache is not None:
            sink = np.concatenate([cache, sink])

        if (reset_cache or use_cache):
            cache_file = join(path, cache_name)
            utool.dump(sink, cache_file, format=cache_format)

        return sink
    
    def clear_shm(self, clean=True):
        shms = glob.glob(f'/dev/shm/rur*')
        kmps = glob.glob(f'/dev/shm/__KMP*_{os.getuid()}')
        if(len(shms)>0):
            shms = [shm.split('/')[-1] for shm in shms]
            olds = []
            for shm in shms:
                nubar = shm.count('_')
                if(nubar==6): # Default (hhmmss_xxxxxx)
                    _, _, _, fuid, fdate, _, _ = shm.split('_')
                elif(nubar==5): # Old (hhmmss)
                    _, _, _, fuid, fdate, _ = shm.split('_')
                else: # Weird mode (ex: yohan_dust)
                    splits = shm.split('_')
                    fuid = splits[-4]
                    fdate = splits[-3]
                if(f'u{os.getuid()}' == fuid):
                    date_diff = datetime.datetime.now() - datetime.datetime.strptime(fdate, '%Y%m%d')
                    if(date_diff.days>=7):
                        olds.append(shm)
            if(len(olds)>0):
                total_size = np.sum([os.path.getsize(f"/dev/shm/{shm}") for shm in olds])
                if(not clean): warnings.warn(f"Warning! Found {len(olds)} old shared memory ({total_size/(1024**3):.2f} GB)", UserWarning)
                for old in olds:
                    if(not clean): print(f" > `/dev/shm/{old}`")
                    else:
                        size = os.path.getsize(f"/dev/shm/{old}")/(1024**3)
                        os.remove(f"/dev/shm/{old}")
                        print(f"Removed: `/dev/shm/{old}` ({size:.2f} GB)")
                if(not clean): print("\nIf you want to remove them, run `snap.clear_shm()`")
        if(len(kmps)>0):
            olds = []
            for kmp in kmps:
                date_diff = datetime.datetime.now()-datetime.datetime.fromtimestamp(os.path.getmtime(kmp))
                if(date_diff.days>=7):
                    olds.append(kmp)
            if(len(olds)>0):
                print(f" > Found {len(olds)} old KMP files (`/dev/shm/__KMP*_{os.getuid()}`)")
                if(clean):
                    size = 0
                    for old in olds:
                        size = os.path.getsize(kmp)/(1024**2)
                        os.remove(old); size += size
                    print(f"{len(olds)} files removed ({size:.2f} MB)")
                else:
                    print("\nIf you want to remove them, run `snap.clear_shm()`")


    def clear(self, part=True, cell=True):
        """Clear exsisting cache from snapshot data.

        Parameters
        ----------
        part : bool
            if True, clear particle cache
        cell : bool
            if True, clear amr cache

        Returns
        -------

        """
        if(self.unitmode != 'code'):
            self.switch_unitmode()
        self.box = self.default_box
        self.pcmap = None
        if (part):
            self.part_data = None
            self.part = None
            self.box_part = None
            self.cpulist_part = np.array([], dtype='i4')
            self.bound_part = np.array([0], dtype='i4')
        if (cell):
            self.cell_data = None
            self.cell = None
            self.box_cell = None
            self.cpulist_cell = np.array([], dtype='i4')
            self.bound_cell = np.array([0], dtype='i4')
        self.flush(msg=False, parent='[clear]')
        readr.close()

    def _read_nstar(self):
        part_file = FortranFile(self.get_path('part', 1))
        part_file.skip_records(4)
        if (not self.longint):
            val = part_file.read_ints()
        else:
            val = part_file.read_longs()
        try: val = val[0]
        except: pass
        return val

    def read_hydro_ng(self):
        # not used

        hydro_uolds = []
        for icpu in np.arange(1, self.params['ncpu'] + 1):
            path = self.get_path(type='hydro', icpu=icpu)
            opened = open(path, mode='rb')

            header = np.fromfile(opened, dtype=np.int32, count=4)
            ndim, nvar, levelmin, nlevelmax = header  # nothing to do

            nocts = np.fromfile(opened, dtype=np.int32, count=nlevelmax - levelmin + 1)
            hydro_uold = np.fromfile(opened, dtype=np.float64)

            hydro_uolds.append(hydro_uold)

            opened.close()

        hydro_uolds = np.concatenate(hydro_uolds)
        hydro_uolds = np.reshape(hydro_uolds, [-1, nvar, 2 ** ndim])
        hydro_uolds = np.swapaxes(hydro_uolds, 1, 2)

        self.params['nvar'] = nvar
        self.var = hydro_uolds

    def read_amr_ng(self):
        amr_poss = []
        amr_lvls = []
        amr_refs = []
        amr_cpus = []

        for icpu in np.arange(1, self.params['ncpu'] + 1):
            amr_path = self.get_path(type='amr', icpu=icpu)
            opened = open(amr_path, mode='rb')

            header = np.fromfile(opened, dtype=np.int32, count=3)
            ndim, levelmin, nlevelmax = header

            nocts = np.fromfile(opened, dtype=np.int32, count=nlevelmax - levelmin + 1)
            amr_size = np.sum(nocts)

            block_size = ndim + 2 ** ndim
            amr = np.fromfile(opened, dtype=np.int32)
            amr = np.reshape(amr, [-1, block_size])

            poss, lvls = ckey2idx(amr[:, :ndim], nocts, levelmin)
            amr_poss.append(poss)
            amr_lvls.append(lvls)
            amr_refs.append(amr[:, ndim:].astype(np.bool))
            amr_cpus.append(np.full(amr_size, icpu))

            opened.close()

        self.x = np.swapaxes(np.concatenate(amr_poss), 1, 2)
        self.lvl = np.concatenate(amr_lvls)
        self.ref = np.concatenate(amr_refs)
        self.cpu = np.concatenate(amr_cpus)

    def get_cell(self, box=None, target_fields=None, domain_slicing=True, exact_box=True, cpulist=None, read_grav=False,
                 ripses=False, python=True, nthread=8, use_cache=False):
        if (box is not None):
            # if box is not specified, use self.box by default
            self.box = box
        if (cpulist is None):
            if (self.box is None or np.array_equal(self.box, self.default_box)):
                # box is default box or None: load the whole volume
                domain_slicing = False
                exact_box = False
        else:
            # if cpulist is set,
            if (not domain_slicing):
                warnings.warn("cpulist cannot be set without domain_slicing!", UserWarning)
                domain_slicing = True
            exact_box = False

        if(target_fields=='basic'):
            target_fields = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'rho','P','level', 'cpu']

        if (self.box is None or not np.array_equal(self.box, self.box_cell) or cpulist is not None):
            if (cpulist is None):
                cpulist = self.get_involved_cpu()
            else:
                domain_slicing = False
                exact_box = False
            if (not ripses):
                self.read_cell(target_fields=target_fields, read_grav=read_grav, cpulist=cpulist, python=python,
                               nthread=nthread, use_cache=use_cache)
            else:
                self.read_ripses(target_fields=target_fields, cpulist=cpulist)
            if (domain_slicing):
                if (np.isin(cpulist, self.cpulist_cell).all() & np.isin(self.cpulist_cell, cpulist).all()):
                    cell = self.cell_data
                else:
                    timer.start('Domain Slicing...')
                    cell = domain_slice(self.cell_data, cpulist, self.cpulist_cell, self.bound_cell)
                    timer.record()
            else:
                cell = self.cell_data

            if (exact_box):
                timer.start("Masking...",tab=1)
                mask = box_mask_table(cell, self.box, snap=self, size=self.cell_extra['dx'](cell), nthread=nthread)
                timer.record(tab=1)
                if(timer.verbose>1):
                    msg = 'Masking cells... %d / %d (%.4f)' % (np.sum(mask), mask.size, np.sum(mask) / mask.size)
                else:
                    msg = 'Masking cells...'
                timer.start(msg, 1)
                cell = cell[mask]
                timer.record()

            cell = Cell(cell, self)
            self.box_cell = self.box
            self.cell = cell
        return self.cell

    def correct_unit(self, l1=1, l2=2):
        '''
        Corrects the unit to follow `boxlen` unit
        '''
        if(self.unitmode == 'physical'):
            warnings.warn("You must do this when `unitmode`='code'!", UserWarning)
            return
        boxlen = self.params['boxlen']
        print(f"Current: {l1}~{l2}")
        print(f"Desired: 0~{boxlen}")
        self.cell['x'] -= l1
        self.cell['y'] -= l1
        self.cell['z'] -= l1
        self.cell['x'] *= boxlen/(l2-l1)
        self.cell['y'] *= boxlen/(l2-l1)
        self.cell['z'] *= boxlen/(l2-l1)

    def get_part(self, box=None, target_fields=None, domain_slicing=True, exact_box=True, cpulist=None, pname=None,
                 python=True, nthread=8, use_cache=False):
        if (box is not None):
            # if box is not specified, use self.box by default
            self.box = box
        if (cpulist is None):
            if (self.box is None or np.array_equal(self.box, self.default_box)):
                # box is default box or None: load the whole volume
                domain_slicing = False
                exact_box = False
        else:
            # if cpulist is set,
            if (not domain_slicing):
                warnings.warn("cpulist cannot be set without domain_slicing!", UserWarning)
                domain_slicing = True
            exact_box = False
        do = False
        if self.part is not None:
            if not isinstance(self.part, tuple):
                if pname != self.part.ptype:
                    print(
                        f"\nYou loaded part only `{self.part.ptype}` but now you want `{pname}`!\nIt forces to remove `{self.part.ptype}` data and retry get_part (so it's inefficient!)\n")
                    self.part_data = None
                    self.part = None
                    self.box_part = None
                    self.cpulist_part = np.array([], dtype='i4')
                    self.bound_part = np.array([0], dtype='i4')
                    do = True
        if (self.box is None or not np.array_equal(self.box, self.box_part) or cpulist is not None or do):
            if (cpulist is None):
                cpulist = self.get_involved_cpu()
            else:
                domain_slicing = True
                exact_box = False
            self.read_part(target_fields=target_fields, cpulist=cpulist, pname=pname, nthread=nthread, python=python, use_cache=use_cache)
            if (domain_slicing):
                if (np.isin(cpulist, self.cpulist_part).all() & np.isin(self.cpulist_part, cpulist).all()):
                    part = self.part_data
                else:
                    timer.start('Domain Slicing...')
                    part = domain_slice(self.part_data, cpulist, self.cpulist_part, self.bound_part)
                    timer.record()
            else:
                part = self.part_data
            if (self.box is not None):
                if (exact_box):
                    timer.start("Masking...", tab=1)
                    mask = box_mask_table(part, self.box, snap=self, nthread=nthread)
                    timer.record(tab=1)
                    if(timer.verbose>1):
                        msg = 'Masking particles... %d / %d (%.4f)' % (np.sum(mask), mask.size, np.sum(mask) / mask.size)
                    else:
                        msg = 'Masking particles...'
                    timer.start(msg, 1)
                    part = part[mask]
                    timer.record()
            part = Particle(part, self, ptype=pname)
            self.box_part = self.box
            self.part = part
        return self.part

    def get_sink(self, box=None, all=False):
        if (all):
            self.box_sink = self.default_box
            self.read_sink()
            self.sink = Particle(self.sink_data, self)
            return self.sink
        if (box is not None):
            # if box is not specified, use self.box by default
            self.box = box
        if (self.box is None or not np.array_equal(self.box, self.box_sink)):
            self.read_sink()
            sink = self.sink_data
            if (self.box is not None):
                mask = box_mask(get_vector(sink), self.box)
                timer.start('Masking sinks... %d / %d (%.4f)' % (np.sum(mask), mask.size, np.sum(mask) / mask.size), 1)
                sink = sink[mask]
                timer.record()
            sink = Particle(sink, self)
            self.box_sink = self.box
            self.sink = sink
        return self.sink

    def get_halos_cpulist(self, halos, radius=1., use_halo_radius=True, radius_name='r', n_divide=4, nthread=1, full=False, manual=True):
        # returns cpulist that encloses given list of halos
        cpulist = []

        def _ibox(halo, radius=1., use_halo_radius=True, radius_name='r'):
            if (use_halo_radius):
                extent = halo[radius_name] * radius * 2
            else:
                extent = radius * 2
            return get_box(get_vector(halo), extent)

        galaxy = True if('sigma_bulge' in halos.dtype.names) else False
        path_in_repo = 'galaxy' if galaxy else 'halo'
        prefix = 'GAL' if galaxy else 'HAL'
        path = f"{self.repo}/{path_in_repo}/{prefix}_{self.iout:05d}/domain_{self.iout:05d}.dat"
        if (radius != 1.)or(not use_halo_radius)or(not radius_name == 'r'):
            manual=True
        if (exists(path))and(not manual):
            domain = domload(path)
            cpulist = [domain[i-1] for i in halos['id']]
        else:
            if (nthread == 1):
                for halo in halos:
                    box = _ibox(halo, radius=radius, use_halo_radius=use_halo_radius, radius_name=radius_name)
                    cpulist.append(get_cpulist(box, None, self.levelmax, self.bound_key, self.ndim, n_divide,
                                            ncpu=self.params['ncpu']))
            else:
                with Pool(processes=nthread) as pool:
                    async_result = [
                        pool.apply_async(
                            get_cpulist,
                            (_ibox(halo, radius, use_halo_radius, radius_name), None, self.levelmax, self.bound_key,
                            self.ndim, n_divide, self.params['ncpu'])
                        ) for halo in halos
                    ]
                    for r in async_result:
                        cpulist.append(r.get())
        if (full):
            return np.unique(np.concatenate(cpulist)), cpulist
        return np.unique(np.concatenate(cpulist))

    def get_cpulist_from_part(self, ids, path_in_repo='part_cpumap', mode='init', filename='%s_cpumap_%05d.pkl'):
        """
        reads particle-cpumap file (if there's any) and returns appropriate cpulist of domains
        that encompass selected id list of paritcles
        mode can either be 'init'(dm + tracer) or 'star'
        """
        if (self.pcmap is None):
            path = join(self.repo, path_in_repo, filename % (mode, self.iout))
            self.pcmap = utool.load(path)
        return np.unique(self.pcmap[ids]).astype('i8')

    def diag(self):
        # prints a brief description of the current status of snapshot.
        dm_tot = 0
        star_tot = 0
        gas_tot = 0
        smbh_tot = 0
        if (self.box is not None):
            volume = np.prod(self.box[:, 1] - self.box[:, 0]) / (self.unit['Mpc'] / self.params['h']) ** 3 / \
                     self.params['aexp'] ** 3
            print('=============================================')
            print('Description of the snapshot %05d (%s)' % (self.iout, self.repo))
            print('---------------------------------------------')
            print('Redshift (z) = %.5f (a = %.5f), Age of the Universe = %.4f Gyr' % (
            self.z, self.aexp, self.params['age']))
            print('Comoving volume of the box: %.3e (Mpc/h)^3' % (volume))
        if (self.part is not None):
            part = self.part
            part = part[box_mask(get_vector(part), self.box)]
            print('---------------------------------------------')
            print('Total  number of particles: %d' % part.size)
            dm = part['dm']
            if(dm.size > 0):
                dm_tot = np.sum(dm['m', 'Msol'])
                dm_min = np.min(dm['m', 'Msol'])

                print('Number of     DM particles: %d with total mass of %.3e Msol, Min. particle mass: %.3e Msol' % (
                dm.size, dm_tot, dm_min))

                contam = np.sum(dm[dm['m'] > np.min(dm['m'])]['m'] * 1.01) / np.sum(dm['m'])
                if (contam > 0.0):
                    print('DM Contamination fraction within the box: %.3f %%' % (contam * 100))

            tracer = part['tracer']

            if (tracer.size > 0):
                tracer_tot = np.sum(tracer['m', 'Msol'])
                tracer_min = np.min(tracer['m', 'Msol'])

                print('Number of tracer particles: %d with total mass of %.3e Msol, Min. particle mass: %.3e Msol' % (
                tracer.size, tracer_tot, tracer_min))

            if (self.params['star']):
                star = part['star']
                smbh = part['smbh']

                star_tot = np.sum(star['m', 'Msol'])
                star_min = np.min(star['m', 'Msol'])

                print('---------------------------------------------')

                print(
                    'Number of       star particles: %d with total mass of %.3e Msol, Min. particle mass: %.3e Msol' % (
                    star.size, star_tot, star_min))
                if (star.size > 0):
                    sfr100 = np.sum(star[star['age', 'Myr'] < 100]['m', 'Msol']) / 1E8
                    sfr10 = np.sum(star[star['age', 'Myr'] < 10]['m', 'Msol']) / 1E7
                    sfr1 = np.sum(star[star['age', 'Myr'] < 1]['m', 'Msol']) / 1E6
                    print('SFR within the box (last 100, 10, 1Myr): %.3e, %.3e %.3e Msol/yr' % (sfr100, sfr10, sfr1))
                    print('Max. stellar velocity: %.3e km/s', np.max(utool.rss(star['vel', 'km/s'])))

                if (smbh.size > 0):
                    smbh_tot = np.sum(smbh['m', 'Msol'])
                    smbh_max = np.max(smbh['m', 'Msol'])

                    print(
                        'Number of       SMBH particles: %d with total mass of %.3e Msol, Max. SMBH mass: %.3e Msol' % (
                        smbh.size, smbh_tot, smbh_max))
                print('DM/Stellar mass ratio is %.3f' % (dm_tot / star_tot))

                star_den = star_tot / volume
                print('Stellar Mass density is %.3e Msol / (Mpc/h)^3' % (star_den))

        if (self.cell is not None):
            cell = self.cell
            cell = cell[box_mask(get_vector(cell), self.box, size=cell['dx'])]
            print('---------------------------------------------')
            print('Min. spatial resolution = %.4f pc (%.4f pc/h in comoving)' % (
            np.min(self.cell['dx', 'pc']), self.boxsize * 1E6 * 0.5 ** np.max(self.cell['level'])))
            print('Total number of cells: %d' % cell.size)
            gas_tot = np.sum(cell['rho'] * (cell['dx']) ** 3) / self.unit['Msol']
            print('Total gas mass: %.3e Msol' % gas_tot)
            print('Max. gas density    : %.3e H/cc' % np.max(self.cell['rho', 'H/cc']))
            print('Max. gas temperature: %.3e K' % np.max(self.cell['T', 'K']))
            print('Max. gas sound speed: %.3e km/s' % np.max(cell['cs', 'km/s']))
            print('Max. gas velocity   : %.3e km/s' % np.max(utool.rss(cell['vel', 'km/s'])))

            if('refmask' in cell.dtype.names):
                contam = 1.-np.sum(cell[cell['refmask']>0.01]['m'])/np.sum(cell['m'])
                if(contam>0.):
                    print('Cell Contamination fraction within the box: %.3f %%' % (contam*100))

        if (self.cell is not None and self.part is not None):
            print('Baryonic fraction: %.3f' % (
                        (gas_tot + star_tot + smbh_tot) / (dm_tot + gas_tot + star_tot + smbh_tot)))

    def write_contam_part(self, mdm_cut):
        self.clear()
        self.get_part(box=self.default_box, pname='dm', target_fields=['x', 'y', 'z', 'm', 'cpu'])
        part = self.part
        contam_part = part[part['m'] > mdm_cut]
        dirpath = join(self.repo, 'contam')
        os.makedirs(dirpath, exist_ok=True)
        utool.dump(contam_part.table, join(dirpath, 'contam_part_%05d.pkl' % self.iout))

    def get_ncell(self, cpulist=None):
        if cpulist is None:
            cpulist = np.arange(1, self.ncpu + 1)
        readr.count_cell(self.snap_path, self.iout, cpulist, self.mode)
        return readr.ncell_table

    def read_namelist(self):
        # reads namelist file (if there is) and returns dictionary of strings
        if self.namelist_data is None:
            path = self.get_path('namelist')
            try:
                self.namelist_data = parse_namelist(path)
            except FileNotFoundError:
                self.namelist_data = {}
        return self.namelist_data

Snapshot = RamsesSnapshot


def trace_parts(part_ini, cropped):
    return part_ini[np.isin(part_ini['id'], cropped['id'], True)]


def write_zoomparts_music(part_ini: Particle, cropped: Particle,
                          filepath: str, reduce: int = None, offset=0.):
    """
    writes position table of particles in MUSIC format.
    offset can be found in music output, and should be divided by 2^level before the input
    """
    cropped_ini = part_ini[np.isin(part_ini['id'], cropped['id'], True)]
    if reduce is not None:
        cropped_ini = np.random.choice(cropped_ini, cropped_ini.size // reduce, replace=False)
    pos = get_vector(cropped_ini) - np.array(offset)
    np.savetxt(filepath, pos)
    return cropped_ini


def write_parts_rockstar(part: Particle, snap: RamsesSnapshot, filepath: str):
    """
    writes particle data in ASCII format that can be read by Rockstar
    Need to be updated
    """
    timer.start('Writing %d particles in %s... ' % (part.size, filepath), 1)

    pos = get_vector(part) * snap.params['boxsize']
    vel = get_vector(part, 'v') * snap.get_unit('v', 'km/s')
    table = fromarrays([*pos.T, *vel.T, part['id']], formats=['f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'i4'])
    np.savetxt(filepath, table, fmt=('%.16e',) * 6 + ('%d',))

    timer.record()


def write_snaps_rockstar(repo: str, start: int, end: int, mode='none',
                         path_in_repo='snapshots', ncpu=48, min_halo_particles=100):
    # write particles in format that can be read by Rockstar
    path = join(repo, 'rst')
    dm_flist = []
    star_flist = []
    for iout in np.arange(start, end):
        snap = RamsesSnapshot(repo, iout, mode, path_in_repo=path_in_repo)
        part = snap.get_part()

        filepath = join(path, 'dm_%05d.dat' % iout)
        write_parts_rockstar(part['dm'], snap, filepath)
        dm_flist.append(filepath)

        if (snap.params['star']):
            filepath = join(path, 'star_%05d.dat' % iout)
            write_parts_rockstar(part['star'], snap, filepath)
            star_flist.append(filepath)

    with open(join(path, 'dmlist.dat'), 'w') as opened:
        for fname in dm_flist:
            opened.write(fname + '\n')

    with open(join(path, 'dm.cfg')) as opened:
        opened.write('SNAPSHOT_NAMES = %s\n' % join(path, 'dmlist.dat'))
        opened.write('NUM_WRITERS = %d\n' % ncpu)
        opened.write('FILE_FORMAT = ASCII\n')

        opened.write('BOX_SIZE = %.3f\n' % snap.params['boxsize'])
        opened.write('PARTICLE_MASS = %.3f\n' % np.min(snap.part['m']) * snap.get_unit('m', 'Msun') * snap.params['h'])
        opened.write('h0 = %.4f\n' % snap.params['h'])
        opened.write('Ol = %.4f\n' % snap.params['omega_l'])
        opened.write('Om = %.4f\n' % snap.params['omega_m'])

        opened.write('MIN_HALO_PARTICLES = %d\n' % min_halo_particles)

    if (snap.params['star']):

        with open(join(path, 'starlist.dat'), 'w') as opened:
            for fname in star_flist:
                opened.write(fname + '\n')

        with open(join(path, 'star.cfg')) as opened:
            opened.write('SNAPSHOT_NAMES = %s\n' % join(path, 'starlist.dat'))
            opened.write('NUM_WRITERS = %d\n' % ncpu)
            opened.write('FILE_FORMAT = ASCII\n')

            opened.write('BOX_SIZE = %.3f\n' % snap.params['boxsize'])
            opened.write(
                'PARTICLE_MASS = %.3f\n' % np.min(snap.part['m']) * snap.get_unit('m', 'Msun') * snap.params['h'])
            opened.write('h0 = %.4f\n' % snap.params['h'])
            opened.write('Ol = %.4f\n' % snap.params['omega_l'])
            opened.write('Om = %.4f\n' % snap.params['omega_m'])

            opened.write('MIN_HALO_PARTICLES = %d\n' % min_halo_particles)


def save_part_cpumap(snap, full_box=False, icpu_dtype='u2', path_in_repo='part_cpumap', mode='init',
                     filename='%s_cpumap_%05d.pkl'):
    # writes cpumap that tells what particle belongs to which cpu
    if (full_box):
        snap.box = None
    snap.get_part()
    if (mode == 'init'):  # dm and tracer
        part = snap.part['init']
    elif (mode == 'star'):
        part = snap.part['star']
    else:
        raise ValueError("Unknown mode: %s" % mode)
    if (part.size > 0):
        size = np.max(part['id']) + 1
        pcmap = np.zeros(size, dtype=icpu_dtype)
        pcmap[part['id']] = part['cpu']
        path = join(snap.repo, path_in_repo, filename % (mode, snap.iout))
        utool.dump(pcmap, path)
    else:
        print("No particle detected, skipping..")


def cut_spherical(table, center, radius, prefix='', ndim=3, inverse=False):
    distances = rss(center - get_vector(table, prefix, ndim))
    if (inverse):
        mask = distances > radius
    else:
        mask = distances <= radius
    return table[mask]


def cut_halo(table, halo, radius=1, use_halo_radius=True, inverse=False, radius_name='rvir'):
    center = get_vector(halo)
    if (use_halo_radius):
        radius = halo[radius_name] * radius
    else:
        radius = radius
    return cut_spherical(table, center, radius, inverse=inverse)


def classify_part(part, pname, ptype=None):
    # classify particles, if familty exists in the data, use it.
    # if not, use id, mass and epoch instead.
    timer.start('Classifying %d particles... ' % part.size, 2)
    if (ptype is not None):
        if isinstance(ptype, str):
            if (pname == ptype):
                return part
            else:
                return np.array([], dtype=part.dtype)
        else:
            raise TypeError(f"Invalid type of given `ptype`({type(ptype)}) instead of `list` or `str`!")

    names = part.dtype.names
    if ('family' in names):
        # Do a family-based classification
        mask = np.isin(part['family'], part_family[pname])

    elif ('epoch' in names):
        # Do a parameter-based classification
        if (pname == 'dm'):
            mask = (part['epoch'] == 0) & (part['id'] > 0)
        elif (pname == 'star'):
            mask = ((part['epoch'] < 0) & (part['id'] > 0)) \
                   | ((part['epoch'] != 0) & (part['id'] < 0))
        elif (pname == 'sink' or pname == 'cloud'):
            mask = (part['id'] < 0) & (part['m'] > 0) & (part['epoch'] == 0)
        elif (pname == 'tracer'):
            mask = (part['id'] < 0) & (part['m'] == 0)
        else:
            mask = False
    elif ('id' in names):
        warnings.warn(
                        f"No `family` or `epoch` field found, using `id` and `mass` instead.",
                        UserWarning)
        # DM-only simulation
        if (pname == 'dm'):
            mask = part['id'] > 0
        elif (pname == 'tracer'):
            mask = (part['id'] < 0) & (part['m'] == 0)
        else:
            mask = False
    else:
        # No particle classification is possible
        raise ValueError('Particle data structure not classifiable.')
    output = part[mask]

    timer.record()
    return output


def find_smbh(part, verbose=None):
    # Find SMBHs by merging sink (cloud) particles
    verbose_tmp = timer.verbose
    if (verbose is not None):
        timer.verbose = verbose
    timer.start('Searching for SMBHs in %d particles...' % part.size, 2)
    sink = classify_part(part, 'cloud')
    names = part.dtype.names
    ids = np.unique(sink['id'])

    smbh = []
    for id in ids:
        smbh_cloud = sink[sink['id'] == id]
        row = ()
        for name in names:
            if (name == 'm'):
                val = np.sum(smbh_cloud[name])
            elif (name == 'family'):
                val = 4
            elif (name == 'tag'):
                val = 0
            else:
                val = np.average(smbh_cloud[name])
            row += (val,)
        smbh.append(row)
    smbh = np.array(smbh, dtype=part.dtype)
    timer.record()
    if (timer.verbose >= 2):
        print('Found %d SMBHs.' % smbh.size)
    timer.verbose = verbose_tmp
    return smbh

def _mask_table(shape, address, tmp, box, trad, ith, jth):
    exist = shared_memory.SharedMemory(name=address)
    data = np.ndarray(shape, dtype=bool, buffer=exist.buf)
    imask = (tmp['x'] >= box[0,0]-trad) & (tmp['x'] <= box[0,1]+trad) & (tmp['y'] >= box[1,0]-trad) & (tmp['y'] <= box[1,1]+trad) & (tmp['z'] >= box[2,0]-trad) & (tmp['z'] <= box[2,1]+trad)
    data[ith:jth] = imask

def box_mask_table(table, box, snap=None, size=0, exclusive=False, chunksize=50000, nthread=8):
    if (exclusive):
        size *= -1
    rad = size/2
    box = np.array(box)
    if(len(table) < (20*chunksize*nthread)):
        box_mask = (table['x'] >= box[0,0]-rad) & (table['x'] <= box[0,1]+rad) & (table['y'] >= box[1,0]-rad) & (table['y'] <= box[1,1]+rad) & (table['z'] >= box[2,0]-rad) & (table['z'] <= box[2,1]+rad)
    else:
        box_mask = np.empty(len(table), dtype=bool)
        shmname = 'boxmask'
        name = snap.make_shm_name(shmname) if(snap is not None) else "boxmask"
        memory = shared_memory.SharedMemory(name=name, create=True, size=box_mask.nbytes)
        data = np.ndarray(box_mask.shape, dtype=bool, buffer=memory.buf)
        Nchunk = int(np.ceil(len(table)/chunksize))
        indicies = np.append(np.arange(0, len(table), chunksize), len(table))
        nthread = min(nthread, Nchunk)

        if(timer.verbose>=2):
            pbar = tqdm(total=len(indicies-1), desc=f"Mask with Chunk")
            def update(*a):
                pbar.update()
        else:
            update = None

        if(snap is not None): signal.signal(signal.SIGTERM, signal.SIG_DFL)
        with Pool(processes=nthread) as pool:
            async_result = []
            if(np.isscalar(rad)): # For part
                async_result = [pool.apply_async(_mask_table, args=(box_mask.shape, memory.name, table[ith:jth], box, rad, ith, jth), callback=update) for ith,jth in zip(indicies[:-1], indicies[1:])]
            else: # For cell
                async_result = [pool.apply_async(_mask_table, args=(box_mask.shape, memory.name, table[ith:jth], box, rad[ith:jth], ith, jth), callback=update) for ith,jth in zip(indicies[:-1], indicies[1:])]
            # iterobj = tqdm(async_result, desc='Mask with Chunk') if(timer.verbose >= 2) else async_result
            iterobj = async_result
            for r in iterobj:
                r.get()

        if(snap is not None): signal.signal(signal.SIGTERM, snap.terminate)
        box_mask[:] = data[:]
        memory.close()
        memory.unlink()
    return box_mask

def box_mask(coo, box, size=None, exclusive=False, nchunk=10000000):
    # masking coordinates based on the box
    if size is not None:
        size = expand_shape(size, [0], 2)
    else:
        size = 0
    if (exclusive):
        size *= -1
    box = np.array(box)
    mask_out = []
    for i0 in range(0, coo.shape[0], nchunk):
        i1 = np.minimum(i0 + nchunk, coo.shape[0])
        if not np.isscalar(size) and size.shape[0] == coo.shape[0]:
            size_now = size[i0:i1]
        else:
            size_now = size
        mask = np.all((box[:, 0] <= coo[i0:i1] + size_now / 2) & (coo[i0:i1] - size_now / 2 <= box[:, 1]), axis=-1)
        mask_out.append(mask)
    mask_out = np.concatenate(mask_out)
    return mask_out


def interpolate_part(part1, part2, name, fraction=0.5, periodic=False):
    assert (part1.snap.unitmode == 'code') and (part2.snap.unitmode == 'code'), "Interpolation is only available in code unit."
    # Interpolates two particle snapshots based on their position and fraction
    timer.start('Interpolating %d, %d particles...' % (part1.size, part2.size), 2)

    id1 = part1['id']
    id2 = part2['id']

    id2 = np.abs(id2)

    part_size = np.maximum(np.max(id1), np.max(id2)) + 1

    val1 = part1[name]
    val2 = part2[name]

    if (name == 'pos' or name == 'vel'):
        pool = np.zeros((part_size, 3), dtype='f8')
    else:
        pool = np.zeros(part_size, dtype=val1.dtype)

    mask1 = np.zeros(part_size, dtype='?')
    mask2 = np.zeros(part_size, dtype='?')

    mask1[id1] = True
    mask2[id2] = True

    active_mask = mask1 & mask2
    if (periodic):
        diff = np.zeros((part_size, 3), dtype='f8')
        diff[id1] += val1
        diff[id2] -= val2

        # if val1-val2 >> 0, the particle moved outside the boundary 1
        # if val1-val2 << 0, the particle moved outside the boundary 0
        val2 = val2.copy()

        val2[diff[id2] > 0.5] += 1.
        val2[diff[id2] < -0.5] -= 1.

    pool[id1] += val1 * (1. - fraction)
    pool[id2] += val2 * fraction
    val = pool[active_mask]

    if (periodic):
        val = np.mod(val, 1.)

    if (timer.verbose >= 2):
        print("Particle interpolation - part1[%d], part2[%d], result[%d]" % (id1.size, id2.size, np.sum(active_mask)))
    timer.record()

    return val


def interpolate_part_pos(part1, part2, Gyr_interp, fraction=0.5):
    assert (part1.snap.unitmode == 'code') and (part2.snap.unitmode == 'code'), "Interpolation is only available in code unit."
    # Interpolates two particle snapshots based on their position and fraction
    timer.start('Interpolating %d, %d particles...' % (part1.size, part2.size), 2)

    id1 = part1['id']
    id2 = part2['id']

    id1 = np.abs(id1)
    id2 = np.abs(id2)

    part_size = np.maximum(np.max(id1), np.max(id2)) + 1

    pos1 = part1['pos']
    pos2 = part2['pos']

    vel1 = part1['vel']
    vel2 = part2['vel']

    pool = np.zeros((part_size, 3), dtype='f8')

    mask1 = np.zeros(part_size, dtype='?')
    mask2 = np.zeros(part_size, dtype='?')

    mask1[id1] = True
    mask2[id2] = True

    active_mask = mask1 & mask2

    time_interval = (part2.snap.age - part1.snap.age) * Gyr_interp

    pool[id1] += interp_term(pos1, vel1, fraction, time_interval, 1)
    pool[id2] += interp_term(pos2, vel2, 1 - fraction, time_interval, -1)
    val = pool[active_mask]

    if (timer.verbose >= 2):
        print("Particle interpolation - part1[%d], part2[%d], result[%d]" % (id1.size, id2.size, np.sum(active_mask)))
    timer.record()

    return val


def interp_term(pos, vel, fraction, time_interval, vel_sign=1):
    fun = lambda x: -np.cos(x * np.pi) / 2 + 0.5  # arbitrary blending function I just invented...
    return (pos + time_interval * fraction * vel * vel_sign) * fun(1 - fraction)


def sync_tracer(tracer, cell, copy=False, **kwargs):
    assert (tracer.snap.unitmode == 'code') and (cell.snap.unitmode == 'code'), "sync_tracer is only available in code unit."
    tid, cid = match_tracer(tracer, cell, **kwargs)
    tracer[tid] = utool.set_vector(tracer, cell[cid]['vel'], prefix='v', copy=copy)
    if (copy):
        return tracer


def match_part_to_cell(part, cell, n_search=16):
    assert (part.snap.unitmode == 'code') and (cell.snap.unitmode == 'code'), "match_part_to_cell is only available in code unit."
    tree = KDTree(cell['pos'])
    dists, idx_cell = tree.query(part['pos'], k=n_search, p=np.inf)

    star_pos = utool.expand_shape(part['pos'], [0, 2], 3)
    dists_cand = np.max(np.abs(cell[idx_cell]['pos'] - star_pos), axis=-1) / cell[idx_cell]['dx']

    min_idxs = np.argmin(dists_cand, axis=-1)
    min_dists = np.min(dists_cand, axis=-1)
    if (np.any(min_dists > 0.5)):
        print(min_dists)
        raise RuntimeError(
            "%d particles are not matched corretly. Try increasing n_search. If it doesn't work, it could mean your cell data is incomplete." % np.sum(
                min_dists > 0.5))

    idx_cell = idx_cell[(np.arange(part.size), min_idxs)]

    return idx_cell


def match_tracer(tracer, cell, min_dist_pc=1, use_cell_size=False):
    assert (tracer.snap.unitmode == 'code') and (cell.snap.unitmode == 'code'), "match_tracer is only available in code unit."
    # match MC gas tracer particles to cell
    timer.start("Matching %d tracers and %d cells..." % (tracer.size, cell.size), 1)
    tree = KDTree(tracer['pos'])
    dists, idx_tracer = tree.query(cell['pos'], p=1)

    if (use_cell_size):
        mask = dists < min_dist_pc * cell.snap.unit['pc']
    else:
        mask = dists < cell['dx']

    idx_cell = np.arange(cell.size)
    idx_cell = idx_cell[mask]
    idx_tracer = idx_tracer[mask]

    print("%d / %d tracers are matched to %d / %d cells"
          % (np.unique(idx_tracer).size, tracer.size, np.unique(idx_cell).size, cell.size))
    timer.record()
    return idx_tracer, idx_cell

def load_tracer(tracers, input_iouts=None, target_fields=None, verbose=True, validate=False):
    header = load(f"/storage5/TRACER/header.pkl")
    minid = header['minid']
    iout_table = header['nout']

    npart = len(tracers)
    if input_iouts is None:
        input_iouts = iout_table
    else:
        isin = np.isin(input_iouts, iout_table)
        if verbose:
            if len(isin) != len(input_iouts):
                print(f" > {input_iouts[~isin]} are not found in the header.")
        input_iouts = input_iouts[isin]
        
    lenout = len(input_iouts)
    if target_fields is None:
        target_fields = ['x','y','z','cpu','family']
    names = [('x','f8'), ('y','f8'), ('z','f8'), ('cpu','i2'), ('family','i1')]
    names = [it for it in names if it[0] in target_fields]
    dtype = names + [('id','i4'), ('iout','i4')]
    itemsize = np.dtype(dtype).itemsize # Byte
    if verbose: print(f" > {npart} tracers & {lenout} outputs")
    if verbose: print(f" > Array size: {npart*lenout*itemsize/1024/1024/1024:.2f} GB")
    newarr = np.empty(npart*lenout, dtype=dtype)
    argsort = np.argsort(tracers['id'])
    tracers = tracers[argsort]
    newarr['id'] = np.repeat(tracers['id'], lenout)
    newarr['iout'] = np.tile(input_iouts, npart)
    if verbose: print(f" > Result dtype: {newarr.dtype}")
    

    Nrow = 100000
    prefixs = (tracers['id']-minid)//Nrow
    ufixs, counts = np.unique(prefixs, return_counts=True)
    if verbose: print(f"Check {len(ufixs)} file bricks...")


    chunks = np.unique(input_iouts//100)
    ccursor = 0
    for ic, chunk in enumerate(chunks):
        dirname1 = f"/storage5/TRACER/{100*chunk:03d}"
        couts = input_iouts[(input_iouts >= 100*chunk)&(input_iouts < 100*(chunk+1))]
        irows = iout_table; irows = irows[irows >= 100*chunk]; irows = np.where(np.isin(irows, couts))[0]
        if verbose: print(f"Time-chunk {ic+1}: {couts[0]}({irows[0]}th) - {couts[-1]}({irows[-1]}th)")

        icursor = 0
        jcursor = 0
        for ufix, count in tqdm(zip(ufixs, counts), total=len(ufixs)):
            ipart = tracers[icursor : icursor+count]
            whichcols = (ipart['id']-minid)%Nrow

            for name, dtype in names:
                fname = f"{dirname1}/tracer_{name}_{ufix:04d}.dat"; bsize = int(dtype[-1])
                with open(fname, 'rb') as f:
                    leng = int.from_bytes(f.read(4), byteorder='little')
                    oldrow = -1
                    for irow, cout in zip(irows, couts):
                        if (irow - oldrow)>1: f.seek(4 + irow*bsize*leng)
                        tmp = np.frombuffer(f.read(bsize*leng), dtype=dtype)
                        tmp = tmp[whichcols]
                        if validate:
                            # ! If you don't believe this function: set validate=True
                            assert np.array_equal(ipart['id'], newarr['id'][ccursor+jcursor : ccursor+jcursor + count*lenout : lenout])
                            assert np.all(newarr['iout'][ccursor+jcursor : ccursor+jcursor + count*lenout : lenout]==cout), (cout, newarr['iout'][ccursor+jcursor : ccursor+jcursor + count*lenout : lenout], ccursor+jcursor)
                        newarr[name][ccursor+jcursor : ccursor+jcursor + count*lenout : lenout] = tmp
                        jcursor += 1
                        oldrow = irow
                jcursor -= len(couts)
            
            
            icursor += count
            jcursor += count*lenout
        ccursor += len(couts)
    return newarr


def _track_tracer(shape, shmname, dtype, ikey, path, target_iouts, ids, keys, argwhere, chunk, target_fields):
    exist = shared_memory.SharedMemory(name=shmname)
    datamem = np.ndarray(shape=shape, dtype=dtype, buffer=exist.buf)

    ipath = f"{path}/iout_{chunk:03d}"
    mask = keys==ikey
    indicies = np.arange(len(ids))[mask]
    irows = ids[mask]//1000
    for name in target_fields:
        arr = load(f"{ipath}/tracer_{name}_{ikey:03d}.pkl", msg=False, format='pkl')[irows]
        for jth in range(len(indicies)):
            datamem[indicies[jth]*len(target_iouts):(indicies[jth]+1)*len(target_iouts)][name] = arr[jth][argwhere]

def track_tracer(tracer, target_iouts=None, target_fields=['x','y','z','cpu','family'], nthread=1):
    snap = tracer.snap
    path = f"{snap.repo}/TRACER"
    header = load(f"{path}/header.pkl", msg=False, format='pkl')
    minid = header['minid']
    if(timer.verbose>=1):
        print(f"\n{len(tracer)} tracers are given")
        target_iouts = header['nout'] if target_iouts is None else np.atleast_1d(target_iouts)
        print(f"Find iouts: {target_iouts}\n")
    lengt = len(target_iouts)
    ids = tracer['id']-minid
    lengi = len(ids)
    keys = np.mod(ids,1000)

    nout = header['nout']
    where = np.where(np.isin(nout, target_iouts, assume_unique=True))[0]
    dtype = [('x','f8'), ('y','f8'), ('z','f8'), ('cpu','i2'), ('family','i2')]
    dtype = [(name, form) for name, form in dtype if name in target_fields] + [('id','i4'), ('iout', 'i4')]
    bsize = np.sum([int(form[-1]) for name, form in dtype]) # byte size
    if(timer.verbose>=1):
        print(f"Allocate array ({len(target_iouts)*len(ids)*bsize / 1024**3:.2f} GB)")
    data = np.empty( len(target_iouts) *len(ids), dtype=dtype)
    data['id'] = np.repeat(ids, len(target_iouts))
    data['iout'] = np.tile(target_iouts, len(ids))

    chunks = np.unique( (nout[where]//100 * 100).astype(int) )
    for chunk in chunks:
        ipath = f"{path}/iout_{chunk:03d}"
        print(f"Check snapshots of {chunk:03d}~{chunk+100:03d}")
        used_nout = nout[(nout >= chunk) & (nout < (chunk+100))]
        argwhere = np.where(np.isin(used_nout, target_iouts, assume_unique=True))[0]
        print(f" > Snap Num.{used_nout[argwhere]}")
        if(nthread==1):
            for ikey in tqdm(range(1000), desc=f"Reading tracer bricks"):
                if not ikey in keys: continue
                mask = keys==ikey
                indicies = np.arange(lengi)[mask]
                irows = ids[mask]//1000
                for name in target_fields:
                    arr = load(f"{ipath}/tracer_{name}_{ikey:03d}.pkl", msg=False, format='pkl')[irows]
                    for jth in range(len(indicies)):
                        data[indicies[jth]*lengt:(indicies[jth]+1)*lengt][name] = arr[jth][argwhere]
        else:
            shmname = 'tracermap'
            snap.tracer_mem = shared_memory.SharedMemory(name=snap.make_shm_name(shmname), create=True, size=data.nbytes)
            snap.memory.append(snap.tracer_mem)
            data_shared = np.ndarray(data.shape, dtype=dtype, buffer=snap.tracer_mem.buf)
            ikeys = np.unique(keys)

            if(timer.verbose>=1):
                pbar = tqdm(total=len(ikeys), desc=f"Reading tracer bricks")
                def update(*a): pbar.update()
            else:
                update = None

            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            with Pool(processes=nthread) as pool:
                async_result = [pool.apply_async(
                    _track_tracer, (data_shared.shape, snap.tracer_mem.name, dtype, ikey, path, target_iouts, ids, keys, argwhere, chunk, target_fields), callback=update
                    ) for ikey in ikeys]
                iterobj = async_result
                for r in iterobj: r.get()
            signal.signal(signal.SIGTERM, snap.terminate)
            for name in target_fields:
                data[name] = data_shared[name]
            snap.tracer_mem.close()
            snap.tracer_mem.unlink()
    return data

def cpumap_tracer(tracer, target_iouts=None, extend=False, nthread=4):
    snap = tracer.snap
    path = f"{snap.repo}/TRACER/old"
    header = load(f"{path}/header.pkl", msg=False, format='pkl')
    minid = header['minid']
    if(timer.verbose>=1):
        print(f"\n{len(tracer)} tracers are given")
        if(target_iouts is None):
            target_iouts = header['nout']
        else:
            target_iouts = np.atleast_1d(target_iouts)
        print(f"Find iouts: {target_iouts}\n")
    ids = tracer['id']-minid
    keys = np.mod(ids,1000)

    nout = header['nout']
    where = np.where(np.isin(nout, target_iouts, assume_unique=True))[0]

    if(timer.verbose>=1):
        print(f"Allocate array ({len(target_iouts)*len(ids)*16 / 1024**3:.2f}GB)")
    data = np.empty((len(target_iouts), len(ids)), dtype='int16')

    if(timer.verbose>=1):
        pbar = tqdm(total=1000, desc=f"Reading tracer bricks")
        def update(*a):
            pbar.update()
    else:
        update = None
    if(nthread==1):
        for ikey in range(1000):
            mask = keys==ikey
            fname = f"{path}/tracer_{ikey:03d}.pkl"
            tmp = load(fname, msg=False, format='pkl')
            indicies = np.arange(len(ids))[mask]
            iids = ids[mask]
            cpumap = tmp['cpumap'][iids//1000]
            for iout, iwhere in zip(target_iouts, where):
                ctmp = cpumap[:, iwhere]
                data[iwhere][indicies] = ctmp.astype('int16')
            if(update is not None):
                update()
    else:
        shmname = 'tracermap'
        snap.tracer_mem = shared_memory.SharedMemory(name=snap.make_shm_name(shmname), create=True, size=data.nbytes)
        snap.memory.append(snap.tracer_mem)
        data = np.ndarray(data.shape, dtype='int16', buffer=snap.tracer_mem.buf)

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        with Pool(processes=nthread) as pool:
            async_result = [pool.apply_async(_cpumap_tracer, (data.shape, snap.tracer_mem.name, ikey, path, target_iouts, ids, keys==ikey, where), callback=update) for ikey in range(1000)]
            # iterobj = tqdm(async_result, desc=f"Reading tracer bricks") if (timer.verbose >= 1) else async_result
            iterobj = async_result
            for r in iterobj:
                r.get()
        signal.signal(signal.SIGTERM, snap.terminate)

    if(extend):
        return data
    else:
        result = {}
        for i in range(len(target_iouts)):
            iout = target_iouts[i]
            result[iout] = np.unique(data[i])
        if(nthread>1):
            snap.tracer_mem.close()
            snap.tracer_mem.unlink()
        return result

def _cpumap_tracer(shape, address, ikey, path, target_iouts, ids, mask, where):
    exist = shared_memory.SharedMemory(name=address)
    datamem = np.ndarray(shape=shape, dtype='int16', buffer=exist.buf)

    fname = f"{path}/tracer_{ikey:03d}.pkl"
    tmp = load(fname, msg=False, format='pkl')
    indicies = np.arange(len(ids))[mask]
    iids = ids[mask]
    cpumap = tmp['cpumap'][iids//1000]
    for iout, iwhere in zip(target_iouts, where):
        ctmp = cpumap[:, iwhere]
        datamem[iwhere][indicies] = ctmp.astype('int16')

def time_series(repo, iouts, halo_table, mode='none', extent=None, unit=None):
    # returns multiple snapshots from repository and array of iouts
    snaps = []
    snap = None
    for halo, iout in zip(halo_table, iouts):
        snap = RamsesSnapshot(repo, iout, mode, snap=snap)
        if (extent is None):
            extent_now = halo['rvir'] * 2
        else:
            extent_now = extent * snap.unit[unit]
        box = get_box(get_vector(halo), extent_now)
        snap.box = box
        snaps.append(snap)
    return snaps


def get_cpulist(box_unit, binlvl, maxlvl, bound_key, ndim, n_divide, ncpu=None):
    # get list of cpu domains involved in selected box using hilbert key.
    # box_unit should be in unit (0, 1)^3
    # calculate volume
    volume = np.prod([box_unit[:, 1] - box_unit[:, 0]])
    
    # set level of bin to compute hilbery curve (larger is finer, slower)
    if (binlvl is None):
        binlvl = int(np.log2(1. / (volume + 1E-20)) / ndim) + n_divide
    # avoid too large binlvl
    if (binlvl > 64 // ndim):
        binlvl = 64 // ndim - 1

    # compute the indices of minimum bounding box of the current box in binlvl
    lower, upper = np.floor(box_unit[:, 0] * 2 ** binlvl).astype(int), np.ceil(box_unit[:, 1] * 2 ** binlvl).astype(int)
    bbox = np.stack([lower, upper], axis=-1)

    # do cartesian product to list all bins within the bounding box 
    bin_list = utool.cartesian(
        np.arange(bbox[0, 0], bbox[0, 1]),
        np.arange(bbox[1, 0], bbox[1, 1]),
        np.arange(bbox[2, 0], bbox[2, 1]))  # TODO: generalize this

    if (timer.verbose >= 2):
        print("Setting bin level as %d..." % binlvl)
        print("Input box:", box_unit)
        print("Bounding box:", bbox)
        ratio = np.prod([bbox[:, 1] / 2 ** binlvl - bbox[:, 0] / 2 ** binlvl]) / volume
        print("Volume ratio:", ratio)
        print("N. of Blocks:", bin_list.shape[0])

    # compute hilbert key of all bins
    keys = hilbert3d(*(bin_list.T), binlvl, bin_list.shape[0])
    keys = np.array(keys)
    key_range = np.stack([keys, keys + 1], axis=-1)
    key_range = key_range.astype('f8')

    # check all involved domains
    involved_cpu = np.zeros(ncpu, dtype='?')
    icpu_ranges = np.searchsorted(bound_key / 2. ** (ndim * (maxlvl - binlvl + 1)), key_range)
    icpu_ranges = np.unique(icpu_ranges, axis=0)
    for icpu_range in icpu_ranges:
        involved_cpu[np.arange(icpu_range[0], icpu_range[1] + 1, dtype=int)] = True
    involved_cpu = np.where(involved_cpu)[0] + 1

    if (timer.verbose >= 2):
        print("List of involved CPUs: ", involved_cpu)
    return involved_cpu


def ckey2idx(amr_keys, nocts, levelmin, ndim=3):
    idx = 0
    poss = []
    lvls = []
    for noct, leveladd in zip(nocts, np.arange(0, nocts.size)):
        ckey = amr_keys[idx: idx + noct]
        idx += noct
        ckey = np.repeat(ckey[:, :, np.newaxis], 2 ** ndim, axis=-1)
        suboct_ind = np.arange(2 ** ndim)
        nstride = 2 ** np.arange(0, ndim)

        suboct_ind, nstride = np.meshgrid(suboct_ind, nstride)

        cart_key = 2 * ckey + np.mod(suboct_ind // nstride, 2) + 0.5
        level = levelmin + leveladd
        poss.append(cart_key / 2 ** level)
        lvls.append(np.full(noct, level))
    poss = np.concatenate(poss)
    # poss = np.mod(poss-0.5, 1)
    lvls = np.concatenate(lvls)
    return poss, lvls


def domain_slice(array, cpulist, cpulist_all, bound):
    # array should already been aligned with bound
    idxs = np.where(np.isin(cpulist_all, cpulist, assume_unique=True))[0]
    doms = np.stack([bound[idxs], bound[idxs + 1]], axis=-1)
    segs = doms[:, 1] - doms[:, 0]

    out = np.empty(np.sum(segs), dtype=array.dtype)  # same performance with np.concatenate
    now = 0
    for dom, seg in zip(doms, segs):
        out[now:now + seg] = array[dom[0]:dom[1]]
        now += seg

    return out


def bulk_sort(array):
    # Sorts the array cpu-wise, not used for now
    cpumap = array['cpu']
    idxs = compute_boundary(cpumap)
    counts = np.diff(idxs)
    idxs = idxs[:-1]
    cpulist = cpumap[idxs]

    key = np.argsort(cpulist)
    cpulist = cpulist[key]
    idxs = idxs[key]
    counts = counts[key]

    new = np.empty(array.size, dtype=array.dtype)
    now = 0
    bound_new = [0]
    for icpu, idx, count in zip(cpulist, idxs, counts):
        new[now:now + count] = array[idx:idx + count]
        now += count
        bound_new.append(now)

    return new, np.array(bound_new)


def compute_boundary(cpumap, cpulist):
    bound = np.searchsorted(cpumap, cpulist)
    return np.concatenate([bound, [cpumap.size]])


class GraficLevel(object):
    # an object to read grafic ic file of specific level
    def __init__(self, level_repo, level=None, read_pos=True):
        self.repo = level_repo
        self.read_pos = read_pos
        if (level is None):
            self.level = int(level_repo[-3:])
        else:
            self.level = level
        self.read_header('ic_deltab')
        self.set_coo()

    def read_ic(self):
        vel = []
        if (self.read_pos):
            pos = []
        else:
            self.pos = None
        self.rho = self.read_file(join(self.repo, 'ic_deltab'))
        if (exists(join(self.repo, 'ic_refmap'))):
            self.ref = self.read_file(join(self.repo, 'ic_refmap'))
        else:
            self.ref = None
        for dim in ['x', 'y', 'z']:
            vel_dim = self.read_file(join(self.repo, 'ic_velc%s' % dim))
            vel.append(vel_dim)
            if (self.read_pos):
                pos_dim = self.read_file(join(self.repo, 'ic_posc%s' % dim))
                pos.append(pos_dim)

        self.pvar = []
        for idx in self.pvar_idxs:
            self.pvar.append(self.read_file(join(self.repo, 'ic_pvar_%05d' % idx)))

        self.vel = np.stack(vel, axis=-1)
        if (self.read_pos):
            self.pos = np.stack(pos, axis=-1)

    def get_table(self):
        table_dtype = [('coo', 'f4', 3), ('vel', 'f4', 3), ('pos', 'f4', 3), ('rho', 'f4'), ('ref', 'f4')]
        table_dtype += [('pvar%03d' % idx, 'f4') for idx in self.pvar_idxs]
        table = np.zeros(self.rho.size, dtype=table_dtype)
        coo = []
        vel = []
        if (self.read_pos):
            pos = []
        for idim in [0, 1, 2]:
            coo.append(self.coo[:, :, :, idim].flatten())
            vel.append(self.vel[:, :, :, idim].flatten())
            if (self.read_pos):
                pos.append(self.pos[:, :, :, idim].flatten())
        coo = np.stack(coo, axis=-1)
        table['coo'] = coo

        vel = np.stack(vel, axis=-1)
        table['vel'] = vel

        if (self.read_pos):
            pos = np.stack(pos, axis=-1)
            table['pos'] = pos

        table['rho'] = self.rho.flatten()
        if (self.ref is not None):
            table['ref'] = self.ref.flatten()
        for (idx, pvar_idx) in zip(np.arange(len(self.pvar)), self.pvar_idxs):
            table['pvar%03d' % pvar_idx] = self.pvar[idx].flatten()
        return table

    def read_header(self, fname):
        ff = FortranFile(join(self.repo, fname))
        self.header = ff.read_record(grafic_header_dtype)

        pvar_fnames = glob.glob(join(self.repo, 'ic_pvar_' + '[0-9]' * 5))
        self.pvar_idxs = [int(pvar_fname[-5:]) for pvar_fname in pvar_fnames]

    def set_coo(self):
        nx, ny, nz = self.header['nx'], self.header['ny'], self.header['nz']
        dx = 0.5 ** self.level
        off_arr = np.array([self.header['%soff' % dim][0] for dim in ['y', 'x', 'z']])
        idxarr = np.stack(np.meshgrid(np.arange(ny) + 0.5, np.arange(nx) + 0.5, np.arange(nz) + 0.5), axis=-1)
        self.coo = ((idxarr + off_arr / self.header['dx']) * dx)[:, :, :, [1, 0, 2]]

    def read_file(self, fname):
        # reads grafic2 file format
        ff = FortranFile(join(self.repo, fname))
        header = ff.read_record(grafic_header_dtype)

        nx = int(header['nx'])
        ny = int(header['ny'])
        nz = int(header['nz'])
        data = np.zeros((nx, ny, nz), dtype='f4')

        for i in range(nz):
            data[:, :, i] = ff.read_record('f4').reshape(nx, ny, order='F')
        ff.close()
        return data

    def __getitem__(self, key):
        return self.header[key]


class GraficIC(object):
    # an object to manage multi-level grafic IC
    def __init__(self, repo=None, level_repos=None, levels=None, read_pos=True):
        self.repo = repo
        self.ic = []
        if (level_repos is None and repo is not None):
            level_repos = glob.glob(join(self.repo, 'level_' + '[0-9]' * 3))
        if (levels is None):
            levels = [int(level_repo[-3:]) for level_repo in level_repos]
        self.levels = levels
        for level, level_repo in zip(self.levels, level_repos):
            ic = GraficLevel(level_repo, level, read_pos=read_pos)
            self.ic.append(ic)

    def read_ic(self):
        for ic in self.ic:
            ic.read_ic()

    def get_table(self):
        tables = []
        for ic, level in zip(self.ic, self.levels):
            table = ic.get_table()
            table = append_fields(table, 'level', np.full(table.size, level), usemask=False)
            tables.append(table)
        return np.concatenate(tables)

    def __getitem__(self, key):
        return self.ic[key]


class Region():
    def evaluate(self, data):
        if (isinstance(data, np.ndarrray) and data.shape[-1] == 3):
            return self.isin(data)
        elif (isinstance(data, Table)):
            return self.isin(data['pos'])

    def isin(self, points):
        pass

    def get_bounding_box(self):
        pass

    __call__ = evaluate


class BoxRegion(Region):
    def __init__(self, box):
        self.box = box

    def set_center(self, center, extent=None):
        center = np.array(center)
        if (extent is None):
            extent = self.get_extent()
        elif (not np.isscalar(extent)):
            extent = np.array(extent)
        self.box = np.stack([center - extent / 2, center + extent / 2], axis=-1)

    def get_extent(self):
        return self.box[:, 1] - self.box[:, 0]

    def get_center(self):
        return np.mean(self.box, axis=-1)

    def get_bounding_box(self):
        return self

    def isin(self, points, size=0):
        box = self.box
        mask = np.all((box[:, 0] <= points + size / 2) & (points - size / 2 <= box[:, 1]), axis=-1)
        return mask


class SphereRegion(Region):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def get_bounding_box(self):
        box = BoxRegion(None)
        box.set_center(self.center, self.radius * 2)
        return box

    def isin(self, points, size=0):
        center = self.center
        radius = self.radius
        return rss(points - center) <= radius - size


def part_density(part, reso, mode='m'):
    snap = part.snap
    if (not isinstance(reso, Iterable)):
        reso = np.repeat(reso, 3)
    mhist = np.histogramdd(part['pos'], weights=part['m'], bins=reso, range=snap.box)[0]
    vol = np.prod((snap.box[:, 1] - snap.box[:, 0]) / reso)
    if (mode == 'm'):
        hist = mhist / vol
    elif (mode == 'sig'):
        vel = part['vel']
        sig2 = np.zeros(shape=reso, dtype='f8')
        for idim in np.arange(0, 2):
            mom1 = np.histogramdd(part['pos'], weights=part['m'] * vel[:, idim], bins=reso, range=snap.box)[0]
            mom2 = np.histogramdd(part['pos'], weights=part['m'] * vel[:, idim] ** 2, bins=reso, range=snap.box)[0]
            sig2 += mom2 / mhist - (mom1 / mhist) ** 2
        hist = np.sqrt(sig2)
    return hist


def get_bytes_data(array):
    barr = array.view('b').reshape((array.size, array.itemsize))
    return barr


def fromndarrays(ndarrays, dtype):
    """
    convert list of ndarray to structured array with given dtype
    faster than np.rec.fromarrays
    only works for 2d arrays for now
    """
    descr = np.dtype(dtype)

    itemsize = 0
    nitem = None
    for nda in ndarrays:
        if (nitem is None):
            nitem = nda.shape[0]
        elif (nitem != nda.shape[0]):
            raise ValueError("Array shape does not match")
        itemsize += nda.shape[1] * nda.dtype.itemsize
    if (descr.itemsize != itemsize):
        raise ValueError(f"Sum of itemsize ({itemsize}) does not match with desired dtype ({descr.itemsize})")

    array = np.empty(nitem, descr)
    barr = get_bytes_data(array)
    col = 0
    for nda in ndarrays:
        bnda = nda.view('b')
        barr[:, col:col + bnda.shape[1]] = bnda
        col += bnda.shape[1]
    return array


def quad_to_f16(by):
    # receives byte array with format of IEEE 754 quadruple float and converts to numpy.float128 array
    # because quadruple float is not supported in numpy
    # source: https://stackoverflow.com/questions/52568037/reading-16-byte-fortran-floats-into-python-from-a-file
    out = []
    asint = []
    for raw in np.reshape(by, (-1, 16)):
        asint.append(int.from_bytes(raw, byteorder='little'))
    asint = np.array(asint)
    sign = (np.float128(-1.0)) ** np.float128(asint >> 127)
    exponent = ((asint >> 112) & 0x7FFF) - 16383
    significand = np.float128((asint & ((1 << 112) - 1)) | (1 << 112))
    return sign * significand * 2.0 ** np.float128(exponent - 112)

def parse_namelist(filename):
    config = configparser.ConfigParser(allow_no_value=True)
    with open(filename, 'r') as file:
        lines = []
        for line in file:
            # Replace "&groupname" with "[groupname]"
            if line.strip().startswith("&"):
                line = "[" + line.strip()[1:] + "]\n"
            # Skip the "/" end group notation
            elif line.strip() == "/":
                continue
            lines.append(line)
        
        # Parse the adapted config
        config.read_string("".join(lines))
    
    # Convert config to dictionary format
    namelist_data = {s: dict(config.items(s)) for s in config.sections()}
    return namelist_data

def str_to_tuple(input_data):
    return tuple(map(int, input_data.split(',')))