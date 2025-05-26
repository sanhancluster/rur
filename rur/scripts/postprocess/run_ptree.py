#!/usr/env/python
import os
os.nice(10)
from rur.uhmi import PhantomTree
from rur import uri
import numpy as np
assert np.__version__ < '2.0.0', 'numpy version should be less than 2.0.0'


print("example: `python3 run_ptree_NC.py 2>&1 | tee -a ptree_00613.log`")
####################################################
# Below is the example for the NewCluster simulation
####################################################


# 1. Set the following parameters
# -------------------------------
repo = '/storage7/NewCluster' # path to the simulation repo (See `rur/scripts/san/simulations.py`)
iout = 623 # last snapshot number
mode = 'nc' # rur mode
start_on_middle= True # whether to start_on_middle the ptree brick
galaxy = True


# 2. Setting path
# ---------------
if(galaxy):
    full_path_ptree = f'{repo}/ptree'
    full_path_halomaker = f'{repo}/galaxy'
    path_in_repo_halomaker = 'galaxy'
    dp = True
else:
    full_path_ptree = f'{repo}/ptree_dm'
    full_path_halomaker = f'{repo}/halo'
    path_in_repo_halomaker = 'halo'
    dp = True

# 3. Check if the ptree brick exists and compare to halomaker bricks
# -----------------------------------------------------------
halomaker_exist = os.listdir(full_path_halomaker)
halomaker_exist = [f for f in halomaker_exist if f.startswith('tree_bricks')]
halomaker_exist.sort()
fouts = np.array([int(f[-5:]) for f in halomaker_exist])
if(iout < np.max(fouts)):
    yess = ['y', 'yes', 'Y', 'Yes', 'YES']
    ans = input(f"Warning!\n\tiout={iout} < the largest iout={np.max(fouts)}\n\tin `{full_path_halomaker}`.\n\tChange to {np.max(fouts)}? (y/n)")
    if(ans in yess):
        iout = np.max(fouts)
pbrick_exist = os.listdir(full_path_ptree)
pbrick_exist = [p for p in pbrick_exist if(p.startswith('ptree_0'))and(p.endswith('.pkl'))]
if(start_on_middle):
    # if start_on_middle, then pbrick_exist should be less than halomaker_exist
    assert len(pbrick_exist) < len(halomaker_exist)

# 4. Main run
# -----------
snaps = []
def do_phantom_tree(repo):
    snap = uri.RamsesSnapshot(repo=repo, iout=iout, path_in_repo='snapshots', mode=mode)
    PhantomTree.path_in_repo='ptree'

    # Save each snapshot's ptree brick to `ptree_xxxxx.pkl`
    #   Added columns: `desc`, `npass`
    PhantomTree.from_halomaker(
        snap, 4, 4,
        galaxy=galaxy,
        double_precision=dp,
        start_on_middle=start_on_middle,
        skip_jumps=True,
        path_in_repo_halomaker=path_in_repo_halomaker,
        full_path_halomaker=full_path_halomaker,
        full_path_ptree=full_path_ptree)

    # Merge all ptree bricks to `ptree.pkl`
    #   Removed columns: `mcontam`
    #   Changed columns: `id`->`hmid`
    PhantomTree.merge_ptree(repo, snap.iout, full_path=full_path_ptree, skip_jumps=True)

    # Build tree and save as `ptree_stable.pkl`
    #   Added columns: `fat`, `score_fat`, `son`, `score_son`
    ptree = PhantomTree.load(snap.repo, ptree_file='ptree.pkl',full_path=full_path_ptree)
    ptree = PhantomTree.build_tree(ptree, overwrite=True)
    PhantomTree.save(ptree, snap.repo, ptree_file='ptree_stable.pkl', full_path=full_path_ptree)

    # Add tree info and save as `ptree_stable.pkl`
    #   Added columns: `nprog`, `ndesc`, `first`, `last`, `first_rev`, `last_rev`
    ptree = PhantomTree.load(snap.repo, ptree_file='ptree_stable.pkl',full_path=full_path_ptree)
    ptree = PhantomTree.add_tree_info(ptree, overwrite=True)
    PhantomTree.save(ptree, snap.repo, ptree_file='ptree_stable.pkl', full_path=full_path_ptree)

    # Not used currently?
    #PhantomTree.save(ptree, snap.repo, ptree_file='ptree_stable.pkl')
    #PhantomTree.measure_star_prop(snap, ptree_file='ptree_stable.pkl', output_file='ptree_SFR.pkl')
    #PhantomTree.measure_gas_prop(snap, ptree_file='ptree_SFR.pkl', output_file='ptree_stable.pkl',
    #                             backup_freq=10, subload_limit=5000, n_jobs=40, iout_start=49)

do_phantom_tree(repo)