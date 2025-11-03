print("ex: $ python3 run_prepare.py /storage7/NewCluster --galaxy")
import numpy as np
assert np.__version__ < '2.0.0', "This script is not compatible with numpy 2.0.0 or later."
from rur import uri,uhmi
uri.timer.verbose=0
from rur.utool import *

import os, sys, glob, pwd, grp
from tqdm import tqdm
from multiprocessing import shared_memory, Pool
import argparse, time
import _guide as guide
from _pfunc import vprint, yess, bs, be, filemode


# ---------------------------------------------
# Arguments
# ----------------------------------------------
parser = argparse.ArgumentParser(description='Fast tracer timeseries')
parser.add_argument("repo", help='repo', type=str)
parser.add_argument("--galaxy", action='store_true', default=False)
parser.add_argument("--halo", action='store_true', default=False)
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()
repo = args.repo
galaxy = args.galaxy
halo = args.halo
if galaxy and halo:
    print("You cannot use both --galaxy and --halo at the same time")
    sys.exit(1)
if not galaxy and not halo:
    print("You must use either --galaxy or --halo")
    sys.exit(1)
verbose = args.verbose
debug = args.debug
gstr = 'Galaxy' if galaxy else 'Halo'


print("\n\n")
print(f"=================================================")
print(f"{bs} Post-processing HaloMaker (syj3514@yonsei.ac.kr) {be}")
print(f"=================================================")
if debug:
    print(f" DEBUG MODE")
    print(f"=================================================")
print(f" > repo : `{repo}`")
print(f" > mode : {gstr}")
print(f"=================================")
if verbose:
    print(f"OUTLINE")
    print(f" 1. Check new snapshots")
    print(f" 2. Check halomaker done")
    print(f" 3. Check merger tree")
    print(f" 4. Check extended catalog")
    print(f" 5. Check tracer")
print()

print(f"=================================")
print(f" User Info")
# Current user's UID and GID
uid = os.getuid()
gid = os.getgid()
# Username and group name
user_info = pwd.getpwuid(uid)
group_info = grp.getgrgid(gid)
print(f" > User: {user_info.pw_name} ({uid})")
print(f" > Group: {group_info.gr_name} ({gid})")
all_groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]
print(f" > Groups: {all_groups}")
if not 'gem' in all_groups:
    print(f" !! {bs}You are not in the 'gem' group{be} !!")
    print(f" !! Please ask your system manager to add you to the 'gem' group. !!")
print(f"=================================")
print("")


snaprepo = f"{repo}/snapshots"
print(f"=================================")
print(f"{bs} 1. Check new snapshots {be}")
print(f"=================================")
# Recorded iout table from txt
snap = uri.RamsesSnapshot(repo, 1)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
vprint(f" > Check `{repo}/list_iout_avail.txt`", verbose)
full_iouts = snaps.iout_avail['iout'] # type: ignore
vprint(f" > Recorded iouts: {full_iouts[0]}~{full_iouts[-1]} ({len(full_iouts)})", verbose)
# Manual check iout table
vprint(f" > Read info files...", verbose)
fnames = glob.glob(f"{snaprepo}/output*")
fnames = [f for f in fnames if f[-5:].isdigit()]
fnames = [int(f[-5:]) for f in fnames]
fnames.sort()
vprint(f" > Found iouts: {fnames[0]}~{fnames[-1]} ({len(fnames)})", verbose)
check_link = [f for f in fnames if os.path.islink(f"{snaprepo}/output_{f:05d}")]
if len(check_link)>0:
    vprint(f" \t> Found {len(check_link)} links in snapshots ({check_link[0]}~{check_link[-1]})", verbose)
    realpath = os.path.realpath(f"{snaprepo}/output_{check_link[0]:05d}")
    realpath = os.path.dirname(realpath)
    vprint(f" \t> Check `{realpath}`", verbose)
    fnames2 = glob.glob(f"{realpath}/output*")
    fnames2 = [int(f[-5:]) for f in fnames2]
    fnames2.sort()
    fnames2 = [f for f in fnames2 if f not in fnames]
    if len(fnames2)>0:
        print(f" \t> You have (maybe) new snapshots in `{realpath}`")
        print(f" \t> Found iouts: {fnames2[0]}~{fnames2[-1]} ({len(fnames2)})")
        ans = input(f" \t> Do you want to link them? (y/n): ")
        if ans in yess:
            uid, gid, permissions = filemode(realpath)
            if not permissions:
                print(f" > {bs}You don't have permission to link files in `{realpath}`{be}")
                print(f" > Please ask your system manager to change the permission.")
            for f2 in fnames2:
                os.system(f"ln -s {realpath}/output_{f2:05d} {snaprepo}/")
                print(f"ln -s {realpath}/output_{f2:05d} {snaprepo}/")
        fnames += fnames2
        fnames = np.unique(fnames)
# Check if all iouts are available
new_iouts = []
for iout in fnames:
    if not iout in full_iouts: new_iouts.append(iout)
if len(new_iouts)==0:
    print(f" > {bs}All iouts are recorded in `list_iout_avail.txt` ({full_iouts[0]}~{full_iouts[-1]}){be}")
else:
    uid, gid, permissions = filemode(f"{repo}/list_iout_avail.txt")
    if not permissions:
        print(f" > {bs}You don't have permission to write `list_iout_avail.txt`{be}")
        print(f" > Please ask your system manager to change the permission.")
    uid, gid, permissions = filemode(f"{repo}/list_iout_avail.txt")
    if not permissions:
        print(f" > {bs}You don't have permission to write `list_iout_avail.txt`{be}")
        print(f" > Please ask your system manager to change the permission.")
    print(f" > Missing iouts in `list_iout_avail.txt`")
    print(f" > Missing iouts: {new_iouts}")
    if not debug: snaps.write_iout_avail(use_cache=True)
    print(f" > `list_iout_avail.txt` updated")
snaps.read_iout_avail()
full_iouts = snaps.iout_avail['iout'] # type: ignore

catrepo = f"{repo}/{gstr.lower()}"
print()
print(f"=================================")
print(f"{bs} 2. Check halomaker done {be}")
print(f"=================================")
vprint(f" > Check `{catrepo}`", verbose)
tree_bricks = os.listdir(f"{catrepo}")
tree_bricks = [f for f in tree_bricks if f.startswith('tree_bricks')]
tree_bricks.sort()
vprint(f" > Found {len(tree_bricks)} tree_bricks", verbose)
new_iouts = []
for iout in full_iouts:
    if not f"tree_bricks{iout:05d}" in tree_bricks:
        if iout>30:
            new_iouts.append(iout)
if len(new_iouts)==0:
    print(f" > {bs}All halomaker done{be}")
else:
    print(f" > {bs}Need to run HaloMaker!{be}")
    print(f" > Missing iouts: {new_iouts}")
    guide.howtodo_halomaker(catrepo, new_iouts, galaxy=galaxy, snaprepo=snaprepo)
    print()
    print("COME BACK AFTER HALOMAKER IS DONE")
    sys.exit(0)


ptree_repo = f"{repo}/ptree"
if not galaxy: ptree_repo = f"{repo}/ptree_dm"
print()
print(f"=================================")
print(f"{bs} 3. Check merger tree {be}")
print(f"=================================")
vprint(f" > Check `{ptree_repo}`", verbose)
pbricks = os.listdir(ptree_repo)
pbricks = [f for f in pbricks if f.endswith('pkl')]
pbricks.sort()
vprint(f" > Found {len(pbricks)} pkl files", verbose)
new_iouts = []
for iout in full_iouts:
    if not f"ptree_{iout:05d}.pkl" in pbricks:
        if iout>30:
            new_iouts.append(iout)
if len(new_iouts)==0:
    print(f" > {bs}All PhantomTree done{be}")
else:
    print(f" > Need to run PhantomTree!")
    print(f" > Missing iouts: {new_iouts}")
    guide.howtodo_ptree(ptree_repo, catrepo, new_iouts, galaxy=galaxy, debug=debug, snaprepo=snaprepo)

print()
print(f"=================================")
print(f"{bs} 4. Check extended catalog {be}")
print(f"=================================")
extrepo = f"{catrepo}/extended"
vprint(f" > Check `{extrepo}`", verbose)
extended = os.listdir(extrepo)
extended = [f for f in extended if f.startswith('0')]
extended.sort()
vprint(f" > Found {len(extended)} extended catalogs", verbose)

# Separated files check
new_iouts = []
for iout in full_iouts:
    if not f"{iout:05d}" in extended:
        if iout>30:
            new_iouts.append(iout)
if len(new_iouts)==0:
    print(f" > {bs}All extended catalogs are separately done{be}")
    check_combine=True
else:
    print(f" > Need to run extended catalog!")
    print(f" > Missing iouts: {new_iouts}")
    guide.howtodo_extend(catrepo, new_iouts, galaxy=galaxy, debug=debug)
    check_combine=False

if check_combine:
    # Combined files check
    print()
    vprint(f" > Check whether results are combined", verbose)
    vprint(f" > Check `{extrepo}/avail.pkl`", verbose)
    avail = load(f"{extrepo}/avail.pkl", msg=False)
    nout = avail['nout']
    notyet = full_iouts[~np.isin(full_iouts, nout)]
    names = avail['names']
    avail = avail['avail']
    incomplete = nout[np.sum(avail, axis=1) < len(names)]
    incomplete = incomplete[incomplete>30]
    incomplete = np.unique(np.hstack((incomplete, notyet)))
    if len(incomplete)==0:
        print(f" > {bs}All extended catalogs are combined{be}")
    else:
        print(f" > Need to combine extended catalogs!")
        print(f" > Missing iouts: {incomplete}")
        print(f" > {bs}Run `run_combine_{gstr.lower()}.py`{be}")


print()
print(f"=================================")
print(f"{bs} 5. Check tracer {be}")
print(f"(only NC available for now)")
print(f"=================================")
assert "NewCluster" in repo, "Tracer is only available for NewCluster"
tracer_repo = "/storage5/TRACER"
vprint(f" > Check `{tracer_repo}`", verbose)
header = load(f"{tracer_repo}/header.pkl", msg=False)
mode = header['mode']
minid = header['minid']
desc = header['desc']
tracer_nout = header['nout']

new_iouts = []
for iout in full_iouts:
    if not iout in tracer_nout:
        if iout>30:
            new_iouts.append(iout)
if len(new_iouts)==0:
    print(f" > {bs}All tracer done{be}")
else:
    print(f" > Need to run tracer!")
    print(f" > Missing iouts: {new_iouts}")
    print(f" > {bs}Run `run_tracer.py`{be}")
