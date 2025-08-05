import h5py
import os
import numpy as np
from rur import uri, uhmi, utool
from rur import sim
from rur.config import Timestamp
from rur.scripts.ramses_to_hdf_nc import compute_key_boundaries

timer = Timestamp()

def export_sinkprops_to_hdf_nc():
    """
    Export sink properties to HDF5 format for NewCluster simulation.
    """
    # Define the path to the sink properties file
    sinkprops_path = os.path.join('/storage6/hansan/shared', 'sinkprops.h5')

    # Check if the file already exists
    if os.path.exists(sinkprops_path):
        print(f"Sink properties file already exists at {sinkprops_path}.")
        return

    # Load the sink properties from the simulation
    timer.start("Loading sink properties from NewCluster simulation")
    nc = sim.NewCluster(iout=620)
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


def main():
    """
    Main function to execute the export of sink properties.
    """
    export_sinkprops_to_hdf_nc()

if __name__ == "__main__":
    main()
