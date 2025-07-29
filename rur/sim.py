from rur import uri


class NewHorizon(uri.RamsesSnapshot):
    def __init__(self, iout, box=None):
        repo = '/storage6/NewHorizon'
        super().__init__(repo, iout, mode='nh', box=box)


class NewHorizon2(uri.RamsesSnapshot):
    def __init__(self, iout, box=None):
        repo = '/storage7/NH2'
        super().__init__(repo, iout, mode='nh2', box=box)


class NewCluster(uri.RamsesSnapshot):
    def __init__(self, iout, box=None):
        repo = '/storage7/NewCluster'
        super().__init__(repo, iout, mode='nc', box=box)

GEM_SIMULATIONS = {
    'nc': {
        'name': 'NewCluster',
        'repo': '/storage7/NewCluster',
        'sim_publication': "https://doi.org/10.48550/arXiv.2507.06301",
        'ramses_ver': 4,
        'rur_mode': 'nc',
    },
    'nh': {
        'name': 'NewHorizon',
        'repo': '/storage6/NewHorizon',
        'sim_publication': "https://doi.org/10.1051/0004-6361/202039429",
        'ramses_ver': 3,
        'rur_mode': 'nh',
    },
    'nh2': {
        'name': 'NewHorizon2',
        'repo': '/storage7/NH2',
        'sim_publication': "https://doi.org/10.3847/1538-4365/ad0e71",
        'ramses_ver': 4,
        'rur_mode': 'nh2',
    },
}

def add_custom_snapshot(input_name):
    print(f"Adding new simulation `{input_name}` to GEM_SIMULATIONS.")
    name = str( input("Enter the name of the new simulation (ex: nc): ") )
    repo = str( input("Enter the repository path (ex: /storage7/NewCluster): ") )
    sim_publication = str( input("Enter the simulation publication link: ") )
    ramses_ver = int( input("Enter the RAMSES version (ex: 3 or 4): ") )
    rur_mode = str( input("Enter the RUR mode (ex: nc, nh, nh2): ") )
    added = dict(
        name=name,
        repo=repo,
        sim_publication=sim_publication,
        ramses_ver=ramses_ver,
        rur_mode=rur_mode.lower(),
    )
    if input_name in GEM_SIMULATIONS:
        print(f"Simulation `{input_name}` already exists. Updating the repository path.")
    GEM_SIMULATIONS[input_name] = added
    print(f"Simulation `{input_name}` added/updated successfully.")
    return GEM_SIMULATIONS[input_name]