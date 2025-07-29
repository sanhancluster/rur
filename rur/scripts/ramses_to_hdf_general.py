#!/usr/bin/env python3

import h5py
import numpy as np
import os
import time, datetime
from tqdm import tqdm
from numba import njit, prange
import argparse

from rur import uri, utool
from rur import sim as simlist
from rur.fortranfile import FortranFile
from rur.scripts.san import simulations as sim
from rur.hilbert3d import hilbert3d
import signal
from multiprocessing import Pool


converted_dtypes = {
    'star': [
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'), 
        ('m', 'f4'), ('id', 'i4'), ('level', 'i1'),
        ('family', 'i1'), ('tag', 'i1'), ('epoch', 'f4'), ('metal', 'f4'), ('m0', 'f4'),
        ('H', 'f4'), ('O', 'f4'), ('Fe', 'f4'), ('Mg', 'f4'),
        ('C', 'f4'), ('N', 'f4'), ('Si', 'f4'), ('S', 'f4'),('D', 'f4'),
        ('rho0', 'f4'), ('partp', 'i4'), ('cpu', 'i2')],

    'dm': [
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'), 
        ('m', 'f4'), ('id', 'i4'), ('level', 'i1'),
        ('family', 'i1'), ('tag', 'i1'), ('partp', 'i4'), ('cpu', 'i2')],

    'cloud': [
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'), 
        ('m', 'f4'), ('id', 'i4'), ('level', 'i1'),
        ('family', 'i1'), ('tag', 'i1'), ('partp', 'i4'), ('cpu', 'i2')],

    'tracer': [
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('m', 'f4'), ('id', 'i4'), ('level', 'i1'),
        ('family', 'i1'), ('tag', 'i1'), ('partp', 'i4'), ('cpu', 'i2')],

    'sink': [
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'),
        ('m', 'f4'), ('id', 'i4'),
        ('tform', 'f4'), ('dM', 'f4'), ('dMBH', 'f4'), ('dMEd', 'f4'), ('Esave', 'f4'),
        ('jx', 'f4'), ('jy', 'f4'), ('jz', 'f4'),
        ('sx', 'f4'), ('sy', 'f4'), ('sz', 'f4'), ('spinmag', 'f4')],
}

converted_dtype_cell = [
    ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('level', 'i1'),
    ('rho', 'f4'), ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'), ('P', 'f4'),
    ('metal', 'f4'), ('H', 'f4'), ('O', 'f4'), ('Fe', 'f4'), ('Mg', 'f4'),
    ('C', 'f4'), ('N', 'f4'), ('Si', 'f4'), ('S', 'f4'),('D', 'f4'),
    ('d1', 'f4'), ('d2', 'f4'), ('d3', 'f4'), ('d4', 'f4'),
    ('refmask', 'f4'), ('sigma', 'f4'), ('pot', 'f4'), ('cpu', 'i2')]

CYAN = "\033[36m"
GREEN = "\033[33m"
RESET = "\033[0m"
class Timestamp:
    """
    A class to export time that took to execute the script.
    """

    def __init__(self):
        self.t0 = time.time()
        self.stamps = {}
        self.stamps['start'] = self.t0
        self.verbose = 1

    def elapsed(self, name=None):
        if name is None:
            name = 'start'
        t = self.stamps[name]
        return time.time() - t

    def start(self, name=None):
        if name is None:
            name = 'last'
        self.stamps[name] = time.time()

    def message(self, msg, name=None, verbose=1):
        self.start(name)
        if verbose <= self.verbose:
            elapsed_time = self.elapsed()
            time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            print(f"{CYAN}[ {time_string} ]{RESET} {msg}")

    def record(self, msg=None, name=None, verbose=1):
        if name is None:
            name = 'last'
        if verbose <= self.verbose:
            elapsed_time = self.elapsed()
            time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            recorded_time = self.elapsed(name)
            if msg is None:
                msg = "Done."
            print(f"{CYAN}[ {time_string} ]{RESET} {msg} -> {GREEN}{recorded_time:.2f} s{RESET}")
        self.stamps.pop(name)

timer = Timestamp()

def export_part(repo:uri.RamsesRepo, iout_list=None, n_chunk:int=1000, size_load:int=60, output_path:str='hdf', cpu_list=None, dataset_kw:dict={}, overwrite:bool=True, sim_description:str='', version:str='1.0', nthread:int=8):
    ts = repo
    if iout_list is None:
        iout_list = ts.read_iout_avail()
    
    for iout in tqdm(iout_list, desc=f"Exporting particle data", disable=True):
        timer.message(f"Starting particle data extraction for iout = {iout}.", name='part_hdf')
        snap = ts[iout]

        create_hdf5_part(snap, n_chunk=n_chunk, size_load=size_load, output_path=output_path, cpu_list=cpu_list, dataset_kw=dataset_kw, overwrite=overwrite, sim_description=sim_description, version=version, nthread=nthread)
        
        snap.clear()
        timer.record(f"Particle data extraction completed for iout = {snap.iout}.", name='part_hdf')

def create_hdf5_part(snap:uri.RamsesSnapshot, n_chunk:int, size_load:int, output_path:str='hdf', cpu_list=None, dataset_kw:dict={}, overwrite:bool=True, sim_description:str='', version:str='1.0', nthread:int=8):
    if cpu_list is None:
        cpu_list = np.arange(1, snap.ncpu + 1, dtype='i4')
    else:
        cpu_list = np.sort(cpu_list)

    output_dir = os.path.join(snap.repo, output_path)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'part_{snap.iout:05d}.h5')
    if os.path.exists(output_file) and not overwrite:
        try:
            with h5py.File(output_file, 'r') as fl:
                if fl.attrs['version'] != version:
                    print(f"File {output_file} exists but with different version ({fl.attrs['version']} != {version}). Overwriting.")
                else:
                    # If the file exists and the version matches, skip creation
                    timer.message(f"File {output_file} already exists with matching version. Skipping creation.")
                    return
        except (KeyError, OSError) as e:
            print(f"File {output_file} exists but is not a valid HDF5 file. Overwriting.")
    
    timer.message(f"Generating new part dictionary for iout = {snap.iout} with {len(cpu_list)} CPUs...")
    new_part_dict, pointer_dict = get_new_part_dict(snap, cpu_list=cpu_list, size_load=size_load, nthread=nthread)
    names = new_part_dict.keys()
    
    timer.message(f"Creating HDF5 file {output_file} with {len(new_part_dict)} particle types...")
    with h5py.File(output_file, 'w') as fl:
        fl.attrs['description'] = 'Ramses particle data' \
        '\nThis file contains particle data extracted from Ramses snapshots in HDF5 format.'\
        '\nIt includes particle coordinates, velocities, masses, and other properties.' \
        '\nThe data is organized in chunks based on Hilbert keys for efficient access.'\
        '\nThe data within each chunk is sorted by level.' \
        '\nThis file is generated by rur.rur.scripts.ramses_to_hdf_nc.py script.' \
        '\nThis file contains following particle types:' \
        '\n- star: stellar particles' \
        '\n- dm: dark matter particles' \
        '\n- cloud: cloud particles, temporal particles generated for sink particles.' \
        '\n- tracer: tracer particles, includes gas, star, sink tracers.' \
        '\n- sink: sink particles, represent MBH population.' \
        '\n' + sim_description
        fl.attrs['created'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        add_basic_attrs(fl, snap)
        fl.attrs['cpulist'] = cpu_list

        n_level = snap.levelmax
        fl.attrs['n_level'] = n_level
        fl.attrs['n_chunk'] = n_chunk
        
        n_part_tot = 0
        for name in names:
            new_part = new_part_dict[name][:pointer_dict[name]]
            if new_part.size == 0:
                print(f"No particles of type {name} found in iout = {snap.iout}. Skipping export.")
                continue
            timer.message("Measuring Hilbert key for %s data..." % name)
            coordinates = np.array([new_part['x'], new_part['y'], new_part['z']]).T
            hilbert_key = get_hilbert_key(coordinates, snap.levelmax)

            if new_part.size > 1E8:
                timer.message(f"Sorting {name} data with {new_part.size} particles...")
            sort_key = np.argsort(hilbert_key)
            hilbert_key = hilbert_key[sort_key]
            new_part = new_part[sort_key]
            assert_sorted(hilbert_key)

            timer.message("Getting chunk boundaries...")
            chunk_boundary = get_chunk_boundaries(hilbert_key, n_chunk)
            assert_sorted(chunk_boundary)

            hilbert_key_max = 2**(3 * snap.levelmax)
            safe_mask = chunk_boundary < hilbert_key.size

            # generate hilbery key for each chunk boundary
            hilbert_boundary = np.empty(n_chunk+1, dtype=hilbert_key.dtype)
            hilbert_boundary[safe_mask] = hilbert_key[chunk_boundary[safe_mask]]
            hilbert_boundary[0] = 0
            hilbert_boundary[~safe_mask] = hilbert_key_max
            assert_sorted(hilbert_boundary)

            grp = fl.create_group(name)
            grp.attrs['name'] = name
            grp.attrs['size'] = new_part.size

            grp.attrs['n_level'] = n_level
            grp.attrs['n_chunk'] = n_chunk
            grp.attrs['levelmax'] = snap.levelmax
            grp.attrs['levelmin'] = snap.levelmin

            n_part_tot += new_part.size

            grp.create_dataset('hilbert_boundary', data=hilbert_boundary, **dataset_kw)
            grp.create_dataset('chunk_boundary', data=chunk_boundary, **dataset_kw)
            if 'level' in new_part.dtype.names:
                level_boundary = set_level_boundaries(new_part, chunk_boundary, n_chunk, n_level)
                grp.create_dataset('level_boundary', data=level_boundary, **dataset_kw)
            timer.message(f"Exporting {name} data with {new_part.size} particles...")
            grp.create_dataset('data', data=new_part, **dataset_kw)

        fl.attrs['size'] = n_part_tot
        fl.attrs['version'] = version

def get_new_part_dict(snap:uri.RamsesSnapshot, cpu_list, size_load, nthread=8) -> dict:
    """
    Get a new dictionary to store particle data for each type.
    """
    names = converted_dtypes.keys()

    # pre-define the new particle array based on the snapshot header
    new_part_dict = {}
    pointer_dict = {}
    header = snap.extract_header()
    for name in names:
        if name == 'sink':
            try:
                part = snap.get_sink(all=True)
                header[name] = part.size if part is not None else 0
            except ValueError:
                print(f"No sink data available for iout = {snap.iout}. Skipping sink export.")
                header[name] = 0
        if header[name] == 0:
            continue
        new_dtypes = converted_dtypes[name]
        new_part_dict[name] = np.empty(header[name], dtype=new_dtypes)
        pointer_dict[name] = 0

    names = new_part_dict.keys() # update names to only include available data
    for idx in np.arange(len(cpu_list))[::size_load]:
        cpu_list_sub = cpu_list[idx:np.minimum(idx + size_load, len(cpu_list))]
        if len(cpu_list_sub) == 0:
            continue
        snap.get_part(cpulist=cpu_list_sub, hdf=False, nthread=nthread)
        part_data = snap.part_data
        for icpu in cpu_list_sub:
            # sort the particle within each CPU by Hilbert key, this is to reduce the sorting time later
            idx = np.where(np.isin(snap.cpulist_part, [icpu], assume_unique=True))[0][0]
            bound = snap.bound_part[idx], snap.bound_part[idx+1]
            if bound[0] == bound[1]:
                continue
            sl = slice(*bound)
            part_sl = snap.part_data[sl]
            sl_coo = np.array([part_sl['x'], part_sl['y'], part_sl['z']]).T
            hilbert_key = get_hilbert_key(sl_coo, snap.levelmax)
            sort_key = np.argsort(hilbert_key)
            part_data[sl] = part_data[sl][sort_key]

        for name in names:
            new_part = new_part_dict[name]
            new_dtypes = converted_dtypes[name]
            if name == 'sink':
                if pointer_dict[name] == 0: # we load sink data only once
                    part = snap.get_sink(all=True)
                    hilbert_key = get_hilbert_key(np.array([part['x'], part['y'], part['z']]).T, snap.levelmax)
                    part = part[np.argsort(hilbert_key)] # already Particle class at this point
                else: # sink data is alrady loaded
                    continue
            else:
                part = uri.Particle(part_data, snap)[name]

            timer.message(f"Exporting {name} data with {part.size} particles..."
                  f"\nItem size: {part.table.dtype.itemsize} -> {new_part.dtype.itemsize} B ({new_part.dtype.itemsize / part.table.dtype.itemsize * 100:.2f}%)")

            for field in new_dtypes:
                new_part_dict[name][pointer_dict[name]:pointer_dict[name] + part.size][field[0]] = part[field[0]]
            pointer_dict[name] += part.size
        snap.clear()
    
    for name in names:
        if cpu_list.size == snap.ncpu or name =='sink':
            assert pointer_dict[name] == header[name], f"Number mismatch for {name}: {pointer_dict[name]} != {header[name]}"
    return new_part_dict, pointer_dict

def compute_key_boundaries(key_array: np.ndarray, n_key: int) -> np.ndarray:
    """
    Compute the boundaries for each key based on the key array.
    key must be bewteen 0 and n_key-1.
    """
    key_boundaries = np.searchsorted(key_array, np.arange(n_key), side='right')
    key_boundaries = np.append(0, key_boundaries)
    return key_boundaries


def export_cell(repo:uri.RamsesRepo, iout_list=None, n_chunk:int=1000, size_load:int=60, output_path:str='hdf', cpu_list=None, dataset_kw:dict={}, overwrite:bool=True, sim_description:str='', version:str='1.0', nthread:int=8):
    ts = repo
    if iout_list is None:
        iout_list = ts.read_iout_avail()
    
    for iout in tqdm(iout_list, desc=f"Exporting cell data", disable=True):
        timer.message(f"Starting cell data extraction for iout = {iout}.", name='cell_hdf')
        snap = ts[iout]

        create_hdf5_cell(snap, n_chunk=n_chunk, size_load=size_load, output_path=output_path, cpu_list=cpu_list, dataset_kw=dataset_kw, overwrite=overwrite, sim_description=sim_description, version=version, nthread=nthread)
        
        snap.clear()
        timer.record(f"Cell data extraction completed for iout = {snap.iout}.", name='cell_hdf')


def create_hdf5_cell(snap:uri.RamsesSnapshot, n_chunk:int, size_load:int, output_path:str='hdf', cpu_list=None, dataset_kw:dict={}, overwrite:bool=True, sim_description:str='', version:str='1.0', nthread:int=8):
    """
    Export cell data from the snapshot to HDF5 format.
    """
    if cpu_list is None:
        cpu_list = np.arange(1, snap.ncpu + 1, dtype='i4')
    else:
        cpu_list = np.array(cpu_list, dtype='i4')

    output_dir = os.path.join(snap.repo, output_path)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'cell_{snap.iout:05d}.h5')
    if os.path.exists(output_file) and not overwrite:
        try:
            with h5py.File(output_file, 'r') as fl:
                if fl.attrs['version'] != version:
                    print(f"File {output_file} exists but with different version ({fl.attrs['version']} != {version}). Overwriting.")
                else:
                    # If the file exists and the version matches, skip creation
                    timer.message(f"File {output_file} already exists with matching version. Skipping creation.")
                    return
        except (OSError, KeyError) as e:
            print(f"File {output_file} exists but is not a valid HDF5 file. Overwriting.")
    timer.message(f"Creating HDF5 file {output_file} for cells...")
    with h5py.File(output_file, 'w') as fl:
        fl.attrs['description'] = 'Ramses cell/AMR data' \
        '\nThis file contains cell data extracted from Ramses snapshots in HDF5 format.'\
        '\nIt includes cell coordinates, velocities, densities, and other properties.' \
        '\nThe data is organized in chunks based on Hilbert keys for efficient access.'\
        '\nThe data within each chunk is sorted by level.' \
        '\nThis file is generated by rur.rur.scripts.ramses_to_hdf_nc.py script.' \
        '\n- leaf: leaf cells with no children, can be used for general purpose.' \
        '\n- branch: branch cells with children, can be used for quick access to the averaged quantities.' \
        '\n' + sim_description
        fl.attrs['created'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        add_basic_attrs(fl, snap)
        fl.attrs['cpulist'] = cpu_list

        n_level = snap.levelmax
        fl.attrs['n_level'] = n_level
        fl.attrs['n_chunk'] = n_chunk
        n_cell_tot = 0

        read_branch = None
        for name in ['leaf', 'branch']:
            timer.message(f"Generating new {name} cell array for iout = {snap.iout} with {len(cpu_list)} CPUs...")
            if name == 'leaf':
                read_branch = False
            elif name == 'branch':
                read_branch = True
            new_cell, pointer = get_new_cell(snap, cpu_list=cpu_list, size_load=size_load, read_branch=read_branch)
            new_cell = new_cell[:pointer]

            timer.message("Measuring Hilbert key for cell data...")
            coordinates = np.array([new_cell['x'], new_cell['y'], new_cell['z']]).T
            hilbert_key = get_hilbert_key(coordinates, snap.levelmax)
            if new_cell.size > 1E8:
                timer.message(f"Sorting data with {new_cell.size} cells...")
            sort_key = np.argsort(hilbert_key)
            hilbert_key = hilbert_key[sort_key]
            new_cell = new_cell[sort_key]
            assert_sorted(hilbert_key)

            timer.message("Getting chunk boundaries...")
            chunk_boundary = get_chunk_boundaries(hilbert_key, n_chunk)
            assert_sorted(chunk_boundary)

            hilbert_key_max = 2**(3 * snap.levelmax)
            safe_mask = chunk_boundary < hilbert_key.size

            hilbert_boundary = np.empty(n_chunk+1, dtype=hilbert_key.dtype)
            hilbert_boundary[safe_mask] = hilbert_key[chunk_boundary[safe_mask]]
            hilbert_boundary[0] = 0
            hilbert_boundary[~safe_mask] = hilbert_key_max
            assert_sorted(hilbert_boundary)

            grp = fl.create_group(name)
            grp.attrs['name'] = name
            grp.attrs['size'] = new_cell.size

            grp.attrs['n_level'] = n_level
            grp.attrs['n_chunk'] = n_chunk
            grp.attrs['levelmax'] = snap.levelmax
            grp.attrs['levelmin'] = snap.levelmin
            n_cell_tot += new_cell.size

            grp.create_dataset('hilbert_boundary', data=hilbert_boundary, **dataset_kw)
            grp.create_dataset('chunk_boundary', data=chunk_boundary, **dataset_kw)

            if 'level' in new_cell.dtype.names:
                level_boundary = set_level_boundaries(new_cell, chunk_boundary, n_chunk, n_level)
                grp.create_dataset('level_boundary', data=level_boundary, **dataset_kw)
            
            timer.message(f"Exporting cell data with {new_cell.size} cells...")
            grp.create_dataset('data', data=new_cell, **dataset_kw)

        fl.attrs['size'] = n_cell_tot
        fl.attrs['version'] = version


def get_new_cell(snap:uri.RamsesSnapshot, cpu_list, size_load, read_branch=False, nthread=8) -> np.ndarray:
    """
    Get a new array to store cell data.
    """
    new_cell = None
    pointer = 0
    nthread = min(nthread, len(cpu_list))

    timer.message(f"Calculating total number of cells for iout = {snap.iout} with {len(cpu_list)} CPUs...")
    if nthread==1:
        n_cell = get_ncell(snap, python=True, read_branch=read_branch)
    else:
        ncpu, ndim, nx, ny, nz, nlevelmax, nboundary, boxlen, icoarse_min, jcoarse_min, kcoarse_min = snap._get_amr_info()
        amr_kwargs = {
            'nboundary': nboundary, 'nlevelmax': nlevelmax, 'ndim': ndim,
            'ncpu': ncpu, 'twotondim': 2 ** ndim, 'skip_amr': 3 * (2 ** ndim + ndim) + 1,
            'nx': nx, 'ny': ny, 'nz': nz, 'boxlen': boxlen,
            'icoarse_min': icoarse_min, 'jcoarse_min': jcoarse_min, 'kcoarse_min': kcoarse_min,}
        files = [f"{snap.snap_path}/output_{snap.iout:05d}/amr_{snap.iout:05d}.out{icpu:05d}" for icpu in cpu_list]
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        with Pool(processes=nthread) as pool:
            sizes = pool.starmap(uri._calc_ncell, [(fname, amr_kwargs, read_branch) for fname in files])
        signal.signal(signal.SIGTERM, snap.terminate)
        sizes = np.asarray(sizes, dtype=np.int32)
        n_cell = np.sum(sizes)
    print(n_cell)
    new_cell = np.empty(n_cell, dtype=converted_dtype_cell)
    new_dtypes = converted_dtype_cell

    # sub-load cell data
    for idx in np.arange(len(cpu_list))[::size_load]:
        cpu_list_sub = cpu_list[idx:np.minimum(idx + size_load, len(cpu_list))]
        if len(cpu_list_sub) == 0:
            continue
        snap.get_cell(cpulist=cpu_list_sub, read_grav=True, read_branch=read_branch, hdf=False, nthread=nthread)
        cell_data = snap.cell_data
        for icpu in cpu_list_sub:
            # sort the cell within each CPU by Hilbert key, this is to reduce the sorting time later
            idx = np.where(np.isin(snap.cpulist_cell, [icpu], assume_unique=True))[0][0]
            bound = snap.bound_cell[idx], snap.bound_cell[idx+1]
            if bound[0] == bound[1]:
                continue
            sl = slice(*bound)
            cell_sl = snap.cell_data[sl]
            sl_coo = np.array([cell_sl['x'], cell_sl['y'], cell_sl['z']]).T
            hilbert_key = get_hilbert_key(sl_coo, snap.levelmax)
            sort_key = np.argsort(hilbert_key)
            cell_data[sl] = cell_data[sl][sort_key]

        timer.message(f"Exporting cell data with {cell_data.size} cells..."
              f"\nItem size: {cell_data.dtype.itemsize} -> {new_cell.dtype.itemsize} B ({new_cell.dtype.itemsize / cell_data.dtype.itemsize * 100:.2f}%)")
        for field in new_dtypes:
            new_cell[pointer:pointer + cell_data.size][field[0]] = cell_data[field[0]]
        pointer += cell_data.size
        snap.clear()
    return new_cell, pointer


def get_ncell(snap:uri.RamsesSnapshot, python=False, read_branch=False) -> int:
    """
    Get the total number of cells in the snapshot.
    """
    if not python:
        ncell_table = snap.get_ncell(np.arange(1, snap.ncpu + 1))
        ncell = np.sum(ncell_table)
    else:
        ncpu = snap.ncpu
        ndim = snap.ndim
        twotondim = 2 ** ndim
        skip_amr = 3 * (2 ** ndim + ndim) + 1
        nlevelmax = snap.levelmax

        ncell = 0
        for icpu in range(1, ncpu + 1):
            fname = snap.get_path('amr', icpu)
            with FortranFile(fname, mode='r') as f:
                f.skip_records(5)
                nboundary, = f.read_ints()
                f.skip_records(15)
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
                            if not read_branch:
                                ncell += len(son.flatten()) - np.count_nonzero(son)
                            else:
                                ncell += np.count_nonzero(son)
                        f.skip_records(2 * twotondim)
                    else:
                        f.skip_records(skip_amr)
    return ncell


def get_hilbert_key(coordinates:np.ndarray, levelmax:int) -> np.ndarray:
    subdivisions = 2 ** levelmax
    if levelmax > 21:
        raise ValueError("Levelmax must be less than or equal to 21 to avoid overflow in Hilbert key calculation.")
    idx_list = np.floor(coordinates * subdivisions).astype(int)
    return hilbert3d(*idx_list.T, levelmax, idx_list.shape[0]).astype(np.float128)


def get_chunk_boundaries(hilbert_key:np.ndarray, n_chunk:int) -> np.ndarray:
    """
    Get the boundary indices based on the given array of Hilbert key.
    The Hilbert key must be sorted in ascending order.
    The boundaries are determined by dividing the Hilbert key into `n_chunk` equal parts.
    Indices with same Hilbert key are grouped together.
    """
    chunk_boundary_exact = np.linspace(0, len(hilbert_key), n_chunk + 1).astype(int)
    # get hilbert keys for the exact chunk boundaries
    bound_key = hilbert_key[chunk_boundary_exact[1:-1]]
    lower_boundary = np.searchsorted(hilbert_key, bound_key, side='left')
    upper_boundary = np.searchsorted(hilbert_key, bound_key, side='right')
    # get either left or right boundaries that are closer to the index
    chunk_boundary = np.select([chunk_boundary_exact[1:-1] - lower_boundary < upper_boundary - chunk_boundary_exact[1:-1], True], [lower_boundary, upper_boundary])
    chunk_boundary = np.concatenate([[0], chunk_boundary, [len(hilbert_key)]])  # ensure the last boundary is the end of the array

    return chunk_boundary.astype(np.int32)


def assert_sorted(arr: np.ndarray):
    """
    Assert that the input array is sorted in ascending order.
    """
    if not np.all(arr[:-1] <= arr[1:]):
        raise ValueError("Input array must be sorted in ascending order.")
    return True


def add_basic_attrs(fl: h5py.File, snap: uri.RamsesSnapshot):
    """
    Add basic attributes to the HDF5 file.
    """
    fl.attrs['iout'] = snap.iout
    fl.attrs['icoarse'] = snap.icoarse
    fl.attrs['ncpu'] = snap.ncpu

    fl.attrs['levelmin'] = snap.levelmin
    fl.attrs['levelmax'] = snap.levelmax
    fl.attrs['boxlen'] = snap.boxlen

    fl.attrs['time'] = snap.time
    fl.attrs['aexp'] = snap.aexp
    fl.attrs['age'] = snap.age
    fl.attrs['z'] = snap.z

    fl.attrs['omega_m'] = snap.omega_m
    fl.attrs['omega_l'] = snap.omega_l
    fl.attrs['omega_k'] = snap.omega_k
    fl.attrs['omega_b'] = snap.omega_b
    fl.attrs['H0'] = snap.H0

    fl.attrs['unit_l'] = snap.unit_l
    fl.attrs['unit_t'] = snap.unit_t
    fl.attrs['unit_d'] = snap.unit_d
    fl.attrs['unit_m'] = snap.unit_d * snap.unit_l**3
    fl.attrs['unit_v'] = snap.unit_l / snap.unit_t
    fl.attrs['unit_p'] = snap.unit_m / snap.unit_l / snap.unit_t**2


def set_level_boundaries(new_data: np.ndarray, chunk_boundary: np.ndarray, n_chunk: int, n_level: int):
    """
    Get the level boundaries for the cell data after sorting by level within each chunk.
    """
    for ichunk in range(n_chunk):
        bound = chunk_boundary[ichunk], chunk_boundary[ichunk + 1]
        if bound[0] == bound[1]:
            continue
        sl = slice(*bound)
        sort_key = np.argsort(new_data[sl]['level'])
        new_data[sl] = new_data[sl][sort_key]

    chunk_array = np.repeat(np.arange(n_chunk), chunk_boundary[1:] - chunk_boundary[:-1])
    key = chunk_array * n_level + (new_data['level'] - 1)
    assert_sorted(key)
    level_boundary = compute_key_boundaries(key, n_key=n_chunk * n_level)
    return level_boundary



def main(args):
    """
    Main function to extract data from the snapshot and save it to HDF5.
    """
    
    mode = (args.mode).lower()
    if mode is not None:
        if mode in simlist.GEM_SIMULATIONS:
            simdict = simlist.GEM_SIMULATIONS[mode]
        else:
            simdict = simlist.add_custom_snapshot(mode)
    else:
        UserWarning(f"Mode {mode} not recognized. Assume mode is `nc`")
        simdict = simlist.GEM_SIMULATIONS['nc']

    verbose = args.verbose
    debug = args.debug


    dataset_kw = {
        'compression': args.compression, # lzf
        'shuffle': True,
    }

    #repo_name = 'nc'  # or 'hagn', 'nh', 'nh2', 'yzics', etc.
    #repo = sim.get_repo(repo_name)

    # receive the snapshot from the simulation repository
    repo_path = simdict['repo']
    snap = uri.RamsesSnapshot(repo_path, iout=-1, mode=simdict['rur_mode'])
    repo = uri.RamsesRepo(snap)

    sim_publication = simdict['sim_publication'] #"https://doi.org/10.48550/arXiv.2507.06301"
    rur_repo = "https://github.com/sanhancluster/rur"
    sim_description = f"This file contains snapshot data of {simdict['name']} simulation." \
        f"\nFor any use of this data, please cite the original publication (Han et al.):" \
        f"\n{sim_publication}" \
        f"\nUse RUR library to efficiently handle and analyze the data:" \
        f"\n{rur_repo}"
    version = '1.1'

    n_chunk = args.n_chunk # 8000
    size_load = 160

    #iout_list = [10]
    #cpu_list = [1, 2]
    #export_cell(repo, iout_list=iout_list, n_chunk=n_chunk, size_load=size_load, cpu_list=cpu_list, dataset_kw=dataset_kw, sim_description=sim_description, version=version)
    #export_part(repo, iout_list=iout_list, n_chunk=n_chunk, size_load=size_load, cpu_list=cpu_list, dataset_kw=dataset_kw, sim_description=sim_description, version=version)
    iout_list = None #[30]#[10, 30, 620, 670]
    cpu_list = None
    overwrite = False
    export_cell(repo, iout_list=iout_list, n_chunk=n_chunk, size_load=size_load, cpu_list=cpu_list, dataset_kw=dataset_kw, sim_description=sim_description, version=version, overwrite=overwrite, nthread=args.nthread)
    export_part(repo, iout_list=iout_list, n_chunk=n_chunk, size_load=size_load, cpu_list=cpu_list, dataset_kw=dataset_kw, sim_description=sim_description, version=version, overwrite=overwrite, nthread=args.nthread)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Ramses snapshot data to HDF5 format.')
    # print(f"Usage: {parser.prog} [options] <repo_path>")
    parser.add_argument("--mode", "-m", help='Simulation mode (nc, nh,... or `custom`)', type=str, required=False, default='nc')
    parser.add_argument("--compression", "-c", help='Compression type for HDF5 datasets', type=str, default='lzf')
    parser.add_argument("--n_chunk", "-N", help='Number of chunks to divide the data into', type=int, default=8000)
    parser.add_argument("--nthread", "-n", help='Number of threads to use for processing', type=int, default=8)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    timer.message("Starting data export...", name='main')
    main(args)
    timer.record("Script completed successfully.", name='main')