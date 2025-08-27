#!/usr/bin/env python3

# this script is used to fix the hilbert key of HDF files

import h5py
import numpy as np
import os
import time, datetime
from tqdm import tqdm

from rur import uri, utool
from rur.fortranfile import FortranFile
from rur.scripts.san import simulations as sim
# from rur.hilbert3d import hilbert3d

from rur.scripts.ramses_to_hdf_nc import *

output_dir = '/storage7/NewCluster/hdf'
cell_format = 'cell_{iout:05d}.h5'
iout_min = 1
iout_max = 1000

def reset_hilbert_keys(dataset_kw:dict={}):
    """
    Main function to execute the domain fixing.
    """
    timer = Timestamp()
    timer.start("Resetting hilbert key in HDF files")

    for iout in tqdm(np.arange(iout_min, iout_max + 1), desc="Processing HDF files"):
            # Define the path to the HDF file
        hdf_path = os.path.join(output_dir, cell_format.format(iout=iout))

        # Check if the file exists
        if not os.path.exists(hdf_path):
            continue
        else:
            timer.message(f"Processing {hdf_path}")

        # Open the HDF file and fix the domain
        with h5py.File(hdf_path, 'r+') as fl:
            levelmax = fl.attrs['levelmax']
            levelmin = fl.attrs['levelmin']
            for name in ['leaf', 'branch']:
                grp = fl[name]
                n_level = grp.attrs['n_level']
                n_chunk = grp.attrs['n_chunk']
                
                timer.start(f"Reading {name} in {hdf_path}...")
                new_data = grp['data'][:]
                timer.record("Reading completed.")

                add_group(fl, name, new_data, levelmin=levelmin, levelmax=levelmax,
                          n_chunk=n_chunk, n_level=n_level, part=False, dataset_kw=dataset_kw)

    timer.record("Hilbert key fixing completed.")

def main():
    dataset_kw = {
        'compression': 'lzf',
        'shuffle': True,
        'fletcher32': True,
    }
    reset_hilbert_keys(dataset_kw=dataset_kw)

if __name__ == "__main__":
    main()
    timer.message(f"Process completed at {datetime.datetime.now()}")