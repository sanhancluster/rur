import os
import shutil
import sys

yess = ['y', 'Y', 'Yes', 'yes', 'ok']

print("python3 setup_treemaker.py <mode> <abspath>")
assert len(sys.argv)>2
mode = sys.argv[1]
abspath = sys.argv[2]
path = f"{abspath}/{mode}"
ok = input(f"mode={mode} & path={abspath}\n > {path}\n\tConfirm? [Y/N]")
if not ok in yess:
    raise AssertionError("Confirm mode/path!")

fnames = os.listdir(path)
fnames = [fname for fname in fnames if fname.startswith("tree_bricks")]
fnames.sort()
nsteps = len(fnames)
n_tree_files = 0  # single `tree.dat`` if <= 1, else multiple bush `tree_file_...`

datfile = f"{path}/input_TreeMaker.dat"

if os.path.isfile(datfile):
    ok = input(f"`{datfile}` already exist! Delete? [Y/N]")
    if ok:
        os.remove(datfile)
    else:
        sys.exit()

with open(datfile, "w") as f:
    f.write(f"{nsteps}\t{n_tree_files}\n")
    for fname in fnames:
        f.write(f"'{path}/{fname}'\n")


repo = "/storage6/sources/treemaker_dp"
if(mode == 'galaxy'):
    exe = "TreeMaker_star"
elif(mode == 'halo'):
    exe = "TreeMaker_dm"
else:
    raise ValueError(f"Invalid mode: {mode}!")


if not os.path.isfile(f"{path}/{exe}"):
    os.symlink(f"{repo}/{exe}", f"{path}/{exe}")
