import h5py
import numpy as np
import os

from rur import uri, utool
from rur.scripts.san import simulations as sim

available_dtypes = {
    'star': [
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'), 
        ('m', 'f4'), ('id', 'i4'), ('level', 'i1'),
        ('family', 'i1'), ('tag', 'i1'), ('epoch', 'f4'), ('metal', 'f4'), ('m0', 'f4'),
        ('H', 'f4'), ('O', 'f4'), ('Fe', 'f4'), ('Mg', 'f4'),
        ('C', 'f4'), ('N', 'f4'), ('Si', 'f4'), ('S', 'f4'),('D', 'f4'),
        ('rho0', 'f4'), ('partp', 'i4'),
        ('d1', 'f4'), ('d2', 'f4'), ('d3', 'f4'), ('d4', 'f4')],

    'dm': [
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'), 
        ('m', 'f4'), ('id', 'i4'), ('level', 'i1'),
        ('family', 'i1'), ('tag', 'i1'), ('partp', 'i4')],

    'cloud': [
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'), 
        ('m', 'f4'), ('id', 'i4'), ('level', 'i1'),
        ('family', 'i1'), ('tag', 'i1'), ('partp', 'i4')],

    'tracer': [
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('m', 'f4'), ('id', 'i4'), ('level', 'i1'),
        ('family', 'i1'), ('tag', 'i1'), ('partp', 'i4')],

    'sink': [
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'),
        ('m', 'f4'), ('id', 'i4'),
        ('tform', 'f4'), ('dM', 'f4'), ('dMBH', 'f4'), ('dMEd', 'f4'), ('Esave', 'f4'),
        ('jx', 'f4'), ('jy', 'f4'), ('jz', 'f4'),
        ('sx', 'f4'), ('sy', 'f4'), ('sz', 'f4'), ('spinmag', 'f4')],
}


def export_data(snap: uri.RamsesSnapshot, cpulist: np.ndarray, output_path: str, dataset_kw: dict):
    """
    Get data from the snapshot for a specific field.
    """
    cpulist = np.array(cpulist, dtype='i4')
    snap.get_part(cpulist=cpulist)
    names = available_dtypes.keys()

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'part_{snap.iout:05d}.h5' % snap.iout)
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping creation.")
        return

    with h5py.File(output_file, 'w') as f:
        for name in names:
            if name in snap.part:
                part = snap.part[name]
            elif name == 'sink':
                part = snap.get_sink()

            new_dtypes = available_dtypes[name]
            new_part = np.empty(part.shape, dtype=new_dtypes)

            for field in new_dtypes:
                if field[0] in part.dtype.names:
                    new_part[field[0]] = part[field[0]]
                else:
                    raise ValueError(f"Field {field[0]} not found in {name} data.")

            num_levels = snap.levelmax - snap.levelmin + 1
            if name != 'sink':
                key = (part['cpu'] - 1) * num_levels + part['level'] - snap.levelmin
                sort_key = np.argsort(key)
                new_part = new_part[sort_key]
                key = key[sort_key]
                num_key = snap.ncpu * num_levels
            
            ds = f.create_dataset(name, data=new_part, **dataset_kw)
            ds.attrs['description'] = f"{name} data at iout {snap.iout}"
            ds.attrs['iout'] = snap.iout
            ds.attrs['levelmin'] = snap.levelmin
            ds.attrs['levelmax'] = snap.levelmax

            if name != 'sink':
                # compute and store the cpu boundaries
                cpu_boundaries = compute_key_boundaries(part['cpu'], snap.ncpu)
                ds.attrs['cpu_bound'] = cpu_boundaries

                # compute the key boundaries
                key_boundaries = compute_key_boundaries(part['level'], num_key)

                # reshape key_boundaries to match the number of CPUs and levels, remove the first element which is always 0
                key_boundaries = np.reshape(key_boundaries[1:], (snap.ncpu, num_levels))

                # append a last column of level boundaries from previous CPU, if not present (i.e. the first CPU), set to 0
                first_col = np.append([0], key_boundaries[:-1, -1])
                key_boundaries = np.append(first_col, key_boundaries, axis=1)
                ds.attrs['key_bound'] = key_boundaries
                ds.attrs['cpulist'] = cpulist

def compute_key_boundaries(key_array: np.ndarray, n_key: int) -> np.ndarray:
    """
    Compute the boundaries for each KEY based on the KEY array.
    """
    key_boundaries = np.searchsorted(key_array, np.arange(1, n_key + 1), side='right')
    key_boundaries = np.append(0, key_boundaries)
    return key_boundaries

output_path = '/storage7/NewCluster/hdf/'
repo_name = 'nc'
iout = 10

def main():
    """
    Main function to extract data from the snapshot and save it to HDF5.
    """
    dataset_kw = {
        'compression': 'gzip',
        'compression_opts': 9,
    }
    # receive the snapshot from the simulation repository
    ts = sim.get_repo(repo_name)
    snap = ts[iout]
    cpulist = [450, 451, 452]
    export_data(snap, cpulist, output_path, dataset_kw)

    # Clear the snapshot to free memory
    snap.clear()
    print(f"Data extraction completed for iout {snap.iout}.")
        

