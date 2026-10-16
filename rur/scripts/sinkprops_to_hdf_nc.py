import h5py
import os
import numpy as np
from rur import uri, uhmi, utool
from rur import sim
from rur.config import Timestamp
from rur.scripts.ramses_to_hdf_nc import compute_key_boundaries
import argparse

timer = Timestamp()

def convert_f8_to_f4(arr, exclude_fields=['x', 'y', 'z']):
    if isinstance(exclude_fields, str):
        exclude_fields = {exclude_fields}
    else:
        exclude_fields = set(exclude_fields)

    new_dtype = []
    for name, (dtype, offset) in arr.dtype.fields.items():
        if dtype == np.float64 and name not in exclude_fields:
            new_dtype.append((name, np.float32))
        else:
            new_dtype.append((name, dtype))

    new_arr = np.zeros(arr.shape, dtype=new_dtype)
    for name in arr.dtype.names:
        new_arr[name] = arr[name]

    return new_arr

def export_sinkprops_to_hdf_nc(repo, sinkprops_dir='SINKPROPS'):
    """
    Export sink properties to HDF5 format for NewCluster simulation.
    """
    # Define the path to the sink properties file
    sinkprops_path = os.path.join(repo, sinkprops_dir, 'sinkprops.h5')

    # Check if the file already exists
    if os.path.exists(sinkprops_path):
        print(f"Sink properties file already exists at {sinkprops_path}.")
        return

    # Load the sink properties from the simulation
    timer.start("Loading sink properties from NewCluster simulation")
    nc = uri.RamsesSnapshot(repo, iout=623, hdf=True, mode='nc')
    sp = nc.read_sinkprops()
    timer.record("Sink properties loaded successfully.")

    # Sort sink properties by 'icoarse' and 'id'
    timer.start("Sorting sink properties")
    key = sp['id'] * np.max(sp['icoarse']) + sp['icoarse']
    sp = sp[np.argsort(key)]
    bounds = compute_key_boundaries(sp['id'] - 1, n_key=np.max(sp['id']))
    timer.record("Sink properties sorted successfully.")

    # Save the sorted sink properties to an HDF5 file
    timer.start("Exporting sink properties to HDF5")
    with h5py.File(sinkprops_path, 'w') as fsp:
        fsp.attrs['description'] = 'Sink properties for NewCluster simulation'
        fsp.attrs['icoarse_max'] = np.max(sp['icoarse'])
        fsp.attrs['id_max'] = np.max(sp['id'])
        fsp.attrs['ncoarse'] = len(np.unique(sp['icoarse']))
        fsp.attrs['size'] = sp.size
        fsp.create_dataset('sinkprops', data=sp, compression='lzf', shuffle=True, chunks=True)
        fsp.create_dataset('id_boundary', data=bounds)
    timer.record(f"Sink properties exported to {sinkprops_path} successfully.")


def main(args):
    """
    Main function to execute the export of sink properties.

    """
    export_sinkprops_to_hdf_nc(args.repo, args.sinkprops_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Ramses snapshot data to HDF5 format.')
    parser.add_argument("--repo", "-r", help='Repository path', type=str)
    parser.add_argument("--sinkprops_dir", "-s", help='Sink properties directory', type=str, default='SINKPROPS')

    args = parser.parse_args()

    main(args)