#!/usr/bin/env python3

import h5py
import numpy as np
import os
import time, datetime
from tqdm import tqdm
import argparse

from rur import uri, utool
from rur import sim as simlist
from rur.fortranfile import FortranFile
from rur.scripts.san import simulations as sim
from rur.utool import hilbert3d_map
from rur.config import Timestamp
# from rur.hilbert3d import hilbert3d
import signal
from multiprocessing import Pool


from rur.scripts.ramses_to_hdf_nc import get_chunk_boundaries, add_basic_attrs, add_attr_with_descr, write_dataset, add_group, set_hilbert_boundaries, set_level_boundaries, assert_sorted, get_hilbert_key


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

timer = Timestamp()

def export_part(repo:uri.RamsesRepo, iout_list=None, n_chunk:int=1000, size_load:int=60, output_path:str='hdf', cpu_list=None, dataset_kw:dict={}, overwrite:bool=True, sim_description:str='', version:str='1.0', nthread:int=8):
    ts = repo
    if iout_list is None:
        iout_list = ts.read_iout_avail()['iout']
    
    for iout in tqdm(iout_list, desc=f"Exporting particle data", disable=True):
        timer.start(f"Starting particle data extraction for iout = {iout}.", name='part_hdf')
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
        "\n============================================================================" \
        "\nThis file contains particle data extracted from Ramses snapshots in HDF5 format. "\
        "It includes particle coordinates, velocities, masses, and other properties. "\
        "The data is organized in chunks based on Hilbert keys for efficient access. "\
        "The data within each chunk is sorted by level. "\
        "\nThis file is generated by rur.rur.scripts.ramses_to_hdf_nc.py script."\
        "\nThis file contains following particle types:"\
        "\n'star': stellar particles"\
        "\n'dm': dark matter particles"\
        "\n'cloud': cloud particles, temporal particles generated for sink particles."\
        "\n'tracer': tracer particles, includes gas, star, sink tracers."\
        "\n'sink': sink particles, represent MBH population."\
        "\nEach particle type has the following datasets:"\
        "\n'data': Particle data."\
        "\n'hilbert_boundary': Hilbert key boundaries for each chunk based on the Peano-Hilbery curve with levelmax resolution." \
        "\n'chunk_boundary': Chunk boundary indices for each chunk." \
        "\n'level_boundary': Level boundary indices for each level within chunks." \
        "\n" + sim_description
        fl.attrs['created'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        add_basic_attrs(fl, snap)
        add_attr_with_descr(fl, 'cpulist', cpu_list, 'List of CPU indices used for this snapshot.')

        n_level = snap.levelmax
        add_attr_with_descr(fl, 'n_level', n_level, 'Number of levels in the snapshot.')
        add_attr_with_descr(fl, 'n_chunk', n_chunk, 'Number of chunks in the snapshot.')

        n_part_tot = 0
        for name in names:
            new_part = new_part_dict[name][:pointer_dict[name]]
            if new_part.size == 0:
                print(f"No particles of type {name} found in iout = {snap.iout}. Skipping export.")
                continue

            # Add particle data to HDF5 file
            add_group(fl, name, new_part,
                      levelmin=snap.levelmin, levelmax=snap.levelmax,
                      n_chunk=n_chunk, n_level=n_level, part=True, dataset_kw=dataset_kw)

            n_part_tot += new_part.size
        
        add_attr_with_descr(fl, 'size', n_part_tot, 'Total number of particles in the snapshot.')
        add_attr_with_descr(fl, 'version', version, 'Version of the file.')

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
        if name != 'sink':
            _part_dtype = [p[0] for p in snap.part_dtype]
            new_dtypes = [field for field in new_dtypes if field[0] in _part_dtype]
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
            part_sl = part_data[sl]
            sl_coo = np.array([part_sl['x'], part_sl['y'], part_sl['z']]).T
            hilbert_key = get_hilbert_key(sl_coo, snap.levelmax)
            sort_key = np.argsort(hilbert_key)
            part_data[sl] = part_data[sl][sort_key]

        for name in names:
            new_part = new_part_dict[name]
            new_dtypes = new_part.dtype.descr
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
        iout_list = ts.read_iout_avail()['iout']
    
    for iout in tqdm(iout_list, desc=f"Exporting cell data", disable=True):
        timer.start(f"Starting cell data extraction for iout = {iout}.", name='cell_hdf')
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
        "\n============================================================================" \
        "\nThis file contains cell data extracted from Ramses snapshots in HDF5 format. " \
        "It includes cell coordinates, velocities, densities, and other properties. " \
        "The data is organized in chunks based on Hilbert keys for efficient access. " \
        "The data within each chunk is sorted by level. Please check 'attributes' for more information." \
        "\nThis file is generated by rur.rur.scripts.ramses_to_hdf_nc.py script." \
        "\nThis file contains following cell types:" \
        "\n'leaf': leaf cells with no children, can be used for general purpose." \
        "\n'branch': branch cells with children, can be used for quick access to the averaged quantities." \
        "\nEach cell type has following datasets:" \
        "\n'data': Cell data." \
        "\n'hilbert_boundary': Hilbert key boundaries for each chunk based on the Peano-Hilbery curve with levelmax resolution." \
        "\n'chunk_boundary': Chunk boundary indices for each chunk." \
        "\n'level_boundary': Level boundary indices for each level within chunks." \
        "\n" + sim_description
        fl.attrs['created'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        fl.attrs['attributes'] = "This file include following attributes:"
        add_basic_attrs(fl, snap)
        add_attr_with_descr(fl, 'cpulist', cpu_list, 'List of CPU indices used for this snapshot.')

        n_level = snap.levelmax
        add_attr_with_descr(fl, 'n_level', n_level, 'Number of levels in the snapshot.')
        add_attr_with_descr(fl, 'n_chunk', n_chunk, 'Number of chunks in the snapshot.')

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

            # Add cell data to HDF5 file
            add_group(fl, name, new_cell,
                      levelmin=snap.levelmin, levelmax=snap.levelmax,
                      n_chunk=n_chunk, n_level=n_level, part=False, dataset_kw=dataset_kw)

            n_cell_tot += new_cell.size

        add_attr_with_descr(fl, 'size', n_cell_tot, 'Total number of cells in the snapshot.')
        add_attr_with_descr(fl, 'version', version, 'Version of the file.')


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
    names = list(['x','y','z'][:snap.params['ndim']]) + snap.hydro_names + ['level', 'cpu']
    _converted_dtype_cell = [(name, dtype) for name, dtype in converted_dtype_cell if (name in names)or(name == 'pot')]
    new_cell = np.empty(n_cell, dtype=_converted_dtype_cell)
    new_dtypes = _converted_dtype_cell

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
            cell_sl = cell_data[sl]
            sl_coo = np.array([cell_sl['x'], cell_sl['y'], cell_sl['z']]).T
            hilbert_key = get_hilbert_key(sl_coo, snap.levelmax, cell_data['level'])
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


def export_snapshots(repo:uri.RamsesRepo, iout_list=None, n_chunk:int=1000, size_load:int=60, output_path:str='hdf', cpu_list=None, dataset_kw:dict={}, overwrite:bool=True, sim_description:str='', version:str='1.0', nthread:int=8, export='cp'):
    """
    Export snapshots from the repository to HDF5 format.
    This function will export both particle and cell data.
    """
    ts = repo
    if iout_list is None:
        iout_list = ts.read_iout_avail()

    for iout in tqdm(iout_list, desc=f"Exporting snapshot data", disable=True):
        # Start exporting cell and particle data for each snapshot
        if 'c' in export:
            timer.start(f"Starting cell data extraction for iout = {iout}.", name='cell_hdf')
            snap = ts[iout]
            create_hdf5_cell(snap, n_chunk=n_chunk, size_load=size_load, output_path=output_path, cpu_list=cpu_list, dataset_kw=dataset_kw, overwrite=overwrite, sim_description=sim_description, version=version, nthread=nthread)
            snap.clear()
            timer.record(f"Cell data extraction completed for iout = {snap.iout}.", name='cell_hdf')

        if 'p' in export:
            timer.start(f"Starting particle data extraction for iout = {iout}.", name='part_hdf')
            snap = ts[iout]
            create_hdf5_part(snap, n_chunk=n_chunk, size_load=size_load, output_path=output_path, cpu_list=cpu_list, dataset_kw=dataset_kw, overwrite=overwrite, sim_description=sim_description, version=version, nthread=nthread)
            snap.clear()
            timer.record(f"Particle data extraction completed for iout = {snap.iout}.", name='part_hdf')


def main(args):
    """
    Main function to extract data from the snapshot and save it to HDF5.
    """
    
    mode = (args.mode).lower()
    if mode is not None:
        if mode in simlist.GEM_SIMULATIONS:
            simdict = simlist.GEM_SIMULATIONS[mode]
        else:
            cwd = os.path.dirname(os.path.abspath(__file__))
            fname = f"{cwd}/ramses_to_hdf_custom.json"
            if os.path.exists(fname):
                print(f"Read `{fname}` for custom simulation data")
                import json
                with open(fname) as f:
                    simdicts = json.load(f)
                if mode in simdicts:
                    simdict = simdicts[mode]
                else:
                    simdict = simlist.add_custom_snapshot(mode)
            else:
                simdict = simlist.add_custom_snapshot(mode)
    else:
        UserWarning(f"Mode {mode} not recognized. Assume mode is `nc`")
        simdict = simlist.GEM_SIMULATIONS['nc']

    verbose = args.verbose
    if verbose:
        print("Simulation dictionary:")
        for key, value in simdict.items():
            print(f" > {key}: {value}")
    debug = args.debug


    dataset_kw = {
        'compression': args.compression,
        'shuffle': True,
        'fletcher32': True,
        'chunks': True,
    }

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

    iout_list = None #[30]#[10, 30, 620, 670]
    if iout_list is None:
        iout_list = repo.read_iout_avail(allow_write=True)['iout']
    imin = simdict.get("minout", 0)
    imax = simdict.get("maxout", 10000)
    iout_list = iout_list[(iout_list >= imin) & (iout_list <= imax)]
    print(f"Do for {len(iout_list)} iouts ({iout_list[0]}-{iout_list[-1]})")
    if args.sep >= 0:
        print(f"Changed using {args.sep}/{args.dsep} separation for iouts.")
        iout_list = iout_list[iout_list%args.dsep == args.sep]
        print(f"--> Do for {len(iout_list)} iouts ({iout_list[0]}-{iout_list[-1]})")
    skipout = simdict.get("skipout", [])
    if len(skipout) > 0:
        iout_list = iout_list[~np.isin(iout_list, skipout)]

    cpu_list = None
    overwrite = False
    export = args.export.lower()
    export_snapshots(repo, iout_list=iout_list, n_chunk=n_chunk, size_load=size_load,
                output_path='hdf', cpu_list=cpu_list, dataset_kw=dataset_kw,
                sim_description=sim_description, version=version, overwrite=overwrite, nthread=args.nthread, export=export)
    # if 'c' in export:
    #     export_cell(repo, iout_list=iout_list, n_chunk=n_chunk, size_load=size_load,
    #                 output_path='hdf', cpu_list=cpu_list, dataset_kw=dataset_kw,
    #                 sim_description=sim_description, version=version, overwrite=overwrite, nthread=args.nthread)
    # if 'p' in export:
    #     export_part(repo, iout_list=iout_list, n_chunk=n_chunk, size_load=size_load,
    #                 output_path='hdf', cpu_list=cpu_list, dataset_kw=dataset_kw,
    #                 sim_description=sim_description, version=version, overwrite=overwrite, nthread=args.nthread)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Ramses snapshot data to HDF5 format.')
    # print(f"Usage: {parser.prog} [options] <repo_path>")
    parser.add_argument("--mode", "-m", help='Simulation mode (nc, nh,... or `custom`)', type=str, required=False, default='nc')
    parser.add_argument("--compression", "-c", help='Compression type for HDF5 datasets', type=str, default='lzf')
    parser.add_argument("--n_chunk", "-N", help='Number of chunks to divide the data into', type=int, default=8000)
    parser.add_argument("--nthread", "-n", help='Number of threads to use for processing', type=int, default=8)
    parser.add_argument("--sep", "-s", help='Separation for iouts', type=int, default=-1, required=False)
    parser.add_argument("--dsep", "-d", help='Denominator of separations', type=int, default=2, required=False)
    parser.add_argument("--export", "-e", help='Which for export (c and/or p)', type=str, required=False, default='cp')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    print()
    print()
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(args)
    print()
    print()

    timer.start("Starting data export...", name='main')
    main(args)
    timer.record("Script completed successfully.", name='main')
