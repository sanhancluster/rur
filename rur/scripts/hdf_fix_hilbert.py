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
import argparse

from rur.scripts.ramses_to_hdf_nc import *

timer = Timestamp()

def reset_hilbert_keys(output_dir, iout_min, iout_max, dataset_kw:dict={}):
    """
    Main function to execute the domain fixing.
    """
    cell_format = 'cell_{iout:05d}.h5'
    part_format = 'part_{iout:05d}.h5'

    timer.start("Resetting hilbert key in HDF files")

    for iout in tqdm(np.arange(iout_min, iout_max + 1), desc="Processing HDF files"):
            # Define the path to the HDF file
        hdf_path = os.path.join(output_dir, 'hdf', cell_format.format(iout=iout))
        # Check if the file exists
        if os.path.exists(hdf_path):
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

        hdf_path = os.path.join(output_dir, part_format.format(iout=iout))        
        # Check if the file exists
        if os.path.exists(hdf_path):
            timer.message(f"Processing {hdf_path}")
            with h5py.File(hdf_path, 'r+') as fl:
                levelmax = fl.attrs['levelmax']
                levelmin = fl.attrs['levelmin']
                for name in ['star', 'dm', 'cloud', 'tracer', 'sink']:
                    grp = fl[name]
                    n_level = grp.attrs['n_level']
                    n_chunk = grp.attrs['n_chunk']
                    
                    timer.start(f"Reading {name} in {hdf_path}...")
                    new_data = grp['data'][:]
                    timer.record("Reading completed.")

                    add_group(fl, name, new_data, levelmin=levelmin, levelmax=levelmax,
                            n_chunk=n_chunk, n_level=n_level, part=True, dataset_kw=dataset_kw)

    timer.record("Hilbert key fixing completed.")

def main(args):
    dataset_kw = {
        'compression': 'lzf',
        'shuffle': True,
        'fletcher32': True,
        'chunks': True,
    }
    output_dir = args.repo
    iout_min = args.imin
    iout_max = args.imax

    reset_hilbert_keys(output_dir=output_dir, iout_min=iout_min, iout_max=iout_max, dataset_kw=dataset_kw)

if __name__ == "__main__":
    repo_path_default = '/storage7/NewCluster'
    parser = argparse.ArgumentParser(description='Fix hilbert ordering of existing RAMSES HDF5 file.')
    parser.add_argument("--repo", "-r", help='Repository path', type=str, default=repo_path_default)
    parser.add_argument("--imin", "-i", help='Minimum output index to process (default: 1)', type=int, default=1)
    parser.add_argument("--imax", "-I", help='Maximum output index to process (default: 1)', type=int, default=1)
    parser.add_argument("--nocolor", "-c", help='Disable color on output messages', action='store_true')

    args = parser.parse_args()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('-----------------------------------------------')
    print(f"HDF5 fix hilbert: Script started at {now}")
    print(f"Input arguments: {args}")
    print('-----------------------------------------------')
    timer.use_color = not args.nocolor

    #timer.start("Starting data export...", name='main')
    main(args)
    #timer.record("Script completed successfully.", name='main')

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('-----------------------------------------------')
    print(f"HDF5 fix hilbert: Script completed at {now}")
    print('-----------------------------------------------')
    #timer.message(f"Process completed at {datetime.datetime.now()}")