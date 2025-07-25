import h5py
import os
import numpy as np
from rur import uri, uhmi, utool
from rur.scripts.san import simulations
from numpy.lib.recfunctions import drop_fields


def main():
    dataset_kw = {
        'compression': 'lzf',
        'shuffle': True,
        'chunks': True,
    #   'compression_opts': 4,
    }
    repo = '/storage7/NewCluster/ptree/'
    nested_names = ['star', 'gas', 'cgas', 'wgas', 'hgas', 'misc']

    path = os.path.join(repo, 'ptree_prop.pkl')
    pp = utool.load(path)
    pp.sort(order='id')

    pp_drop = drop_fields(pp, nested_names)

    with h5py.File(os.path.join('/storage7/NewCluster/ptree/', 'ptree_prop.h5'), 'w') as fpp:
        fpp.create_dataset('galaxy', data=pp_drop, **dataset_kw)
        for name in nested_names:
            fpp.create_dataset(name, data=pp[name], **dataset_kw)

if __name__ == '__main__':
    main()
    print("Done!")
