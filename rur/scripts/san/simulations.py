from rur import uri

repos = {
    'hagn': '/storage4/Horizon_AGN',
    'nh': '/storage6/NewHorizon'
}

iouts_last = {
    'hagn': 782,
    'nh': -1
}

def get_repo(name):
    snap = uri.RamsesSnapshot(repo=repos[name], iout=iouts_last[name], mode=name)
    return uri.RamsesRepo(snap)

hagn = get_repo('hagn')