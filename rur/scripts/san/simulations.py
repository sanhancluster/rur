from rur import uri
from rur.config import alias_dict
from collections import defaultdict

name_alias = alias_dict(
    {
        'Horizon-AGN': 'hagn',
        'HAGN': 'hagn',
        'NewHorizon': 'nh',
        'NewH': 'nh',
        'NH': 'nh',
        'NewHorizon2': 'nh2',
        'NewH2': 'nh',
        'NH2': 'nh2',
        'YZiCS': 'yzics',
        'NewCluster': 'nc',
        'NewC': 'nc',
    }
)

repos = {
    'hagn': '/storage4/Horizon_AGN',
    'nh': '/storage6/NewHorizon',
    'nh2': '/storage7/NH2',
    'yzics': '/storage3/Clusters',
    'nc': '/storage7/NewCluster',
}

iouts_last = defaultdict(lambda: -1)
iouts_last.update(
    {
        'hagn': 782,
        'nh': -1
    })

def get_repo(name):
    name = name_alias[name]
    snap = get_snapshot(name)
    return uri.RamsesRepo(snap)

def get_snapshot(name, iout=None):
    name = name_alias[name]
    if iout is None:
        iout = iouts_last[name]
    snap = uri.RamsesSnapshot(repo=repos[name], iout=iout, mode=name)
    return snap

#hagn = get_repo('hagn')