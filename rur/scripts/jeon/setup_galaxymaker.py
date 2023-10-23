import os
import shutil
import sys

# names = ['01605','04466','05420','05427','06098','07206','10002','17891','24954','29172','29176','35663','36413','36415','39990','49096']
# names = ['04466','05420','05427','06098','07206','10002','17891']#,'24954','29172','29176','35663','36413','36415','39990','49096']
#names = ['04466']#,'24954','29172','29176','35663','36413','36415','39990','49096']
#names = sys.argv[1:]
print("\npython3 setup_galaxy.maker <mode> <snappath> <abspath> ...")
print("    mode: galaxy or halo")
print("    snappath: should have `snapshots` directory")
print("    abspath: will have `galaxy` and `halo` directories\n")
mode = sys.argv[1]
ok = input(f"mode={mode}. Confirm? [Y/N]")
yess = ['y', 'Y', 'Yes', 'yes', 'ok']
if not ok in yess:
    raise AssertionError("Confirm the mode [galaxy/halo]!")

gmodes = ['galaxy', 'Galaxy', 'g', 'G', 'Gal', 'gal']
hmodes = ['halo', 'Halo', 'h', 'H', 'Hal', 'hal']
if mode in gmodes:
    mode = 'galaxy'
if mode in hmodes:
    mode = 'halo'


#if len(sys.argv)>2:
#    names = sys.argv[2:]
#else:
#    names = ['nc']
#abspath = sys.argv[2]
base = sys.argv[2]
savedir = sys.argv[3]
#base = '/storage7/NewCluster'
#savedir = "/storage8/jeon/NC"



lbox = 142.04546 # [Mpc] For Others
# lbox = 142.04546*2 # [Mpc] For YZiCS
npart = 100
#dcell_min = 1/1000 # [Mpc] 1kpc For YZiCS & HAGN
dcell_min = 80/1000000 # [Mpc] 40pc for NH
nchem = 0 # For YZiCS
nsteps = 1026 # More than n_output
version = 'Ra4' # Ra, Ra3, Ra4
ncore = 48
dump = True
method = 'MSM' 


if mode == 'halo':
    repo = "/storage6/sources/halomaker_dp"
    exe = "HaloMaker"
else:
    repo = "/storage6/sources/galaxymaker_dp"
    exe = "GalaxyMaker"

params = f"{repo}/input_HaloMaker.dat"
with open(params, mode='r') as f:
    lines = f.readlines()
    newlines = []
    for i, line in enumerate(lines):
        if "lbox" in line[:15]:
            arg = line.find("!")
            newline = f"{line[:17]}{lbox} {line[arg:]}"
        elif "npart" in line[:15]:
            arg = line.find("!")
            newline = f"{line[:17]}{npart} {line[arg:]}"
        elif "method" in line[:15]:
            arg = line.find("!")
            newline = f"{line[:17]}{method} {line[arg:]}"
        elif "dcell_min" in line[:15]:
            arg = line.find("!")
            newline = f"{line[:17]}{dcell_min} {line[arg:]}"
        elif "nchem" in line[:15]:
            arg = line.find("!")
            newline = f"{line[:17]}{nchem} {line[arg:]}"
        elif "nsteps" in line[:15]:
            arg = line.find("!")
            newline = f"{line[:17]}{nsteps} {line[arg:]}"
        elif "dump_" in line[:15]:
            arg = line.find("!")
            newline = f"{line[:17]}.true. {line[arg:]}"
        else:
            newline = line
        print(f"{newline[:-1]}")
        newlines.append(newline)
ok = input(f"Confirm? [Y/N]")
if not ok in yess:
    raise AssertionError("Confirm parameters!")


tmp = f"{params}~"
if os.path.isfile(tmp):
    os.remove(tmp)
with open(tmp, mode='w') as f:
    f.writelines(newlines)

def makeinput(base, galpath, version='Ra3', ncore=1):
    global startout
    path = os.path.abspath(f"{base}/snapshots")
    fnames = os.listdir(path)
    fnames.sort()
    fnames = [file for file in fnames if file.startswith("output_")]

    if os.path.isfile(f"{galpath}/inputfiles_HaloMaker.dat"):
        raise ValueError("`inputfiles_HaloMaker.dat` already exists!")
    with open(f"{galpath}/inputfiles_HaloMaker.dat", "w") as f:
        for fname in fnames:
            iout = fname[-5:]
            if int(iout)>=startout:
                abspath = f"'{path}/{fname}/'"
                line = f"{abspath}\t{version}\t{ncore}\t{iout}\n"
                f.write(line)



for name in names:
    if name[0]=='y':
        path = f"{base}/{name}"
    else:
        path = base
    #galpath = os.path.abspath(f"{path}/{mode}_new")
    galpath = os.path.abspath(f"{savedir}/{mode}")
    if os.path.isdir(galpath):
        ok = input(f"`{galpath}` is already existed! Delete? [Y/N]")
        if ok in yess:
            shutil.rmtree(galpath)
        else:
            pass
    else:
        print(f"`{galpath}` is created")
    if not os.path.isdir(galpath):
        os.mkdir(galpath)

    print(os.listdir(galpath))
    startout=0
    ans = input(f"Start iout > {startout}. Change?")
    try:
        startout = int(ans)
    except:
        pass
    print(f"Start from {startout}")
    
    if not os.path.isfile(f"{galpath}/inputfiles_HaloMaker.dat"):
        makeinput(base,galpath, version=version, ncore=ncore)
    else:
        os.remove(f"{galpath}/inputfiles_HaloMaker.dat")
        makeinput(base,galpath, version=version, ncore=ncore)
   
    if not os.path.isfile(f"{galpath}/input_HaloMaker.dat"):
        shutil.copy(tmp, f"{galpath}/input_HaloMaker.dat")
    
    if not os.path.isfile(f"{galpath}/{exe}"):
        os.symlink(f"{repo}/{exe}", f"{galpath}/{exe}")

    fnames = os.listdir(galpath)
    fnames.sort()
    fnames_tree = [file for file in fnames if file.startswith("tree_bricks")]
    iouts_tree = [int(file[-5:]) for file in fnames_tree if int(file[-5:])>=startout]
    galname = "GAL" if mode=='galaxy' else "HAL"
    fnames_GAL = [file for file in fnames if file.startswith(f"{galname}_")]
    iouts_GAL = [int(file[-5:]) for file in fnames_GAL if int(file[-5:])>=startout]
    print(f"{len(iouts_tree)} tree_bricks and {len(iouts_GAL)} {galname}_xxxxx directories (>={startout})")
    leng = len(iouts_tree)+len(iouts_GAL)
    if leng>0:
        ok = input(f"Delete? [Y/N]")
        if ok in yess:
            for file in fnames_tree:
                iout = int(file[-5:])
                if iout>=startout:
                    os.remove(f"{galpath}/{file}")
            for file in fnames_GAL:
                iout = int(file[-5:])
                if iout>=startout:
                    shutil.rmtree(f"{galpath}/{file}")
        else:
            pass
    print(f"`{galpath}` ready\n\n")
