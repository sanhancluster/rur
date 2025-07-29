import numpy as np
import h5py
import os
import glob

def convert(dir_path):
    # Define the path to the input text file and output HDF5 file
    input_txt_path = glob.glob(os.path.join(dir_path, '*.txt'))
    output_name = os.path.basename(dir_path)
    output_hdf5_path = os.path.join(dir_path, f'../{output_name}.h5')

    # Load data from the text file
    with h5py.File(output_hdf5_path, 'w') as f:
        for file in input_txt_path:
            data = np.loadtxt(file)
            data = data.astype(np.float32)

            # Use the file name (without extension) as the dataset name
            name = os.path.splitext(os.path.basename(file))[0]
            f.create_dataset(name, data=data, compression='lzf', shuffle=True, chunks=True)

def main():
    """
    Main function to execute the conversion.
    """
    import sys
    if len(sys.argv) != 2:
        print("Usage: python txt_to_hdf.py <directory_path>")
        sys.exit(1)

    dir_path = sys.argv[1]
    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a valid directory.")
        sys.exit(1)

    convert(dir_path)
    print(f"Conversion completed. HDF5 file created in {dir_path}.")

if __name__ == "__main__":
    main()