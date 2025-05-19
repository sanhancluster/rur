import numpy as np
import os
from _pfunc import cprint, yess, bs, be




def makeinput(youts, snappath, halpath, version='Ra3', ncore=1):
    # Check write permission
    if(os.access(halpath, os.W_OK)==False):
        raise PermissionError(f"Error: `{halpath}` is not writable!\nAsk admin to change permission.")

    if(os.path.exists(f"{halpath}/inputfiles_HaloMaker.dat")):
        os.remove(f"{halpath}/inputfiles_HaloMaker.dat")
    with open(f"{halpath}/inputfiles_HaloMaker.dat", "w") as f:
        for yout in youts:
            line = f"'{snappath}/output_{yout:05d}/'\t{version}\t{ncore}\t{yout:05d}\n"
            f.write(line)

def howtodo_halomaker(catrepo, new_iouts, galaxy=False, debug=False, snaprepo="/"):
    gstr = "Galaxy" if galaxy else "Halo"
    cprint(f"---------------------------------")
    cprint(f" GUIDE: How to run {gstr}Maker")
    cprint(f"---------------------------------")
    cprint(f"  Hi, user!")
    cprint(f"  You may need to run more {gstr}Maker")
    cprint(f"  To run this, you need three files:")
    cprint("- - - - - - - - - - - - - - - - - - - - -")
    cprint(f"  1. Prepare `{gstr}Maker` execution file")
    cprint(f"    This is fortran-compiled executable.")
    cprint(f"    Usually, it is compiled via Intel Fortran.")
    cprint(f"    If it does not work, please use GNU Fortran version.")
    fname = f"{catrepo}/{gstr}Maker"
    if os.path.exists(fname):
        cprint(f"{bs}    There is already `{gstr}Maker` file, so use it!{be}")
    else:
        cprint(f"    There is no `{gstr}Maker` in `{catrepo}`.")
        if galaxy:
            original = "/storage6/sources/galaxymaker_dp/GalaxyMaker"
        else:
            original = "/storage6/sources/halomaker_dp/HaloMaker_ifort"
        try:
            os.link(original, fname)
            cprint(f"    {bs}But don't worry! I linked `{gstr}Maker` file!{be}")
        except:
            cprint(f"    Oops, maybe you don't have a permission to link.")
            cprint(f"    {bs}Please ask your system manager to link `{original}` there.{be}")

    cprint("- - - - - - - - - - - - - - - - - - - - -")
    cprint(f"  2. Prepare `input_HaloMaker.dat`")
    cprint(f"    This is setting file for `{gstr}Maker`.")
    cprint(f"    You can find this file in `{catrepo}`.")
    cprint(f"    If you want to change the setting, please edit this file.")
    cprint(f"    If you don't know how to edit, please ask your system manager.")
    fname = f"{catrepo}/input_HaloMaker.dat"
    if os.path.exists(fname):
        cprint(f"    {bs}There is already `input_HaloMaker.dat` file, so use it!{be}")
    else:
        cprint(f"    There is no `input_HaloMaker.dat` in `{catrepo}`.")
        original = f"/storage6/sources/{gstr.lower()}maker_dp/input_HaloMaker.dat"
        try:
            os.link(original, fname)
            cprint(f"    {bs}But don't worry! I linked `input_HaloMaker.dat` file!{be}")
        except:
            cprint(f"    Oops, maybe you don't have a permission to link.")
            cprint(f"    {bs}Please ask your system manager to link `{original}` there.{be}")

    cprint("- - - - - - - - - - - - - - - - - - - - -")
    cprint(f"  3. Prepare `inputfiles_HaloMaker.dat`")
    cprint(f"    This is output snapshot list to run {gstr}Maker.")
    cprint(f"    Example:")
    cprint(f"      snapshot_path                                RamsesVer  Ncpu    iout")
    cprint(f"      '/storage7/NewCluster/snapshots/output_00624/'  Ra4     48      00624")
    fname = f"{catrepo}/inputfiles_HaloMaker.dat"
    make_file = f"{catrepo}/make_inputfiles_{gstr.lower()}.py"
    if not os.path.exists(make_file):
        make_file = f"/storage6/sources/{gstr.lower()}maker_dp/make_inputfiles_{gstr.lower()}.py"
    cprint(f"    Option 1.")
    cprint(f"      Manually edit `{fname}`.")
    cprint(f"    Option 2.")
    cprint(f"      Run `{make_file}`.")
    cprint(f"    Option 3.")
    cprint(f"      May I generate this file for you? (Be sure the access permission!)")
    cprint(f"      If yes, I'll do like this:")
    cprint(f"          '{snaprepo}/output_00624/'  Ra4     48      iout:05d")
    cprint(f"          ...")
    if not debug:
        cprint(f"      iout={new_iouts[0]}~{new_iouts[-1]} ({len(new_iouts)})")
    answer = input(f"      {bs}Do you want me to generate this file? (y/n) {be}")
    if answer in yess:
        cprint(f"      Okay, I will generate this file!")
        if not debug:
            makeinput(new_iouts, snaprepo, catrepo, version='Ra4', ncore=48)
        cprint(f"      {bs}`{fname}` prepared!{be}")
    else:
        cprint(f"      {bs}Okay, then choose Option 1 or 2.{be}")

    cprint("- - - - - - - - - - - - - - - - - - - - -")
    cprint(f"  4. Run `{gstr}Maker`")
    hostname = os.uname()[1]
    cprint(f"    In {bs}{hostname}{be} machine,")
    if 'tardis' in hostname:
        cprint(f"        Use jobscript.")
        cprint(f"        Here is my example:")
        cprint("\t\t#PBS -q workq")
        cprint("\t\t#PBS -N GalaxyMaker")
        cprint("\t\t#PBS -S /bin/bash")
        cprint("\t\t#PBS -l select=1:ncpus=48")
        cprint("\t\t#PBS -l walltime=24:00:00")
        cprint("\t\t#PBS -V")
        cprint("\t\t")
        cprint("\t\tcd $PBS_O_WORKDIR")
        cprint("\t\t")
        cprint("\t\tmodule purge")
        cprint("\t\tmodule load intel20/compiler-20")
        cprint("\t\tmodule load intel20/mvapich2-2.3.4")
        cprint("\t\tmodule load intel20/impi-20")
        cprint("\t\texport MV2_ENABLE_AFFINITY=0")
        cprint("\t\texport OMP_STACKSIZE=100G")
        cprint("\t\t")
        cprint("\t\tcd /storage8/NC/galaxy")
        cprint("\t\tiout1='0610'")
        cprint("\t\tiout2='0614'")
        cprint("\t\t./GalaxyMaker &> galaxy_${iout1}_${iout2}.log")

    else:
        cprint(f"        (Recommend to run in tmux)")
        cprint(f"        $ cd {catrepo}")
        cprint(f"        $ ./{gstr}Maker 2>&1 | tee halomaker.log")
    cprint()
    cprint(f"  {bs}You will find results as `{catrepo}/tree_bricksxxxxx`{be}")
    cprint(f"---------------------------------")
    cprint(f"---------------------------------")




def howtodo_ptree(ptree_repo, catrepo, new_iouts, galaxy=False, debug=False, snaprepo="/"):
    gstr = "Galaxy" if galaxy else "Halo"
    cprint(f"---------------------------------")
    cprint(f" GUIDE: How to run PhantomTree for {gstr}")
    cprint(f"---------------------------------")
    cprint(f"  Hi, user!")
    cprint(f"  You want to build merger tree for {gstr}.")
    use_runfile = False
    runfiles = os.listdir(ptree_repo)
    runfiles = [f for f in runfiles if f.endswith('.py')]
    runfiles = [f for f in runfiles if f.startswith('run_ptree')]
    if len(runfiles)==1:
        # Good
        runfile = runfiles[0]
        use_runfile = True
    if use_runfile and (not debug):
        cprint(f"  {bs}You can run `{ptree_repo}/{runfile}`{be}")
        cprint(f"   However, you should modify 18th line of this file.")
        cprint(f"     iout <- Set to be last snapshot number")
    else:
        runfile = "w/h/e/r/e/rur/rur/scripts/jeon/run_ptree.py"
        cprint(f"  {bs}Check this: `{runfile}`{be}")
        cprint(f"  You must check `repo`, `iout`, `mode`, `start_on_middle`, `galaxy`")
    cprint(f"")
    cprint(f"  {bs}You will find results as `{ptree_repo}/ptree_xxxxx.pkl` and ptree_stable.pkl{be}")
    if np.__version__ > "2":
        cprint(f"  !!! WARNING! Do not use Numpy > 2 !!!")
    cprint(f"---------------------------------")
    cprint(f"---------------------------------")


def howtodo_extend(catrepo, new_iouts, galaxy=False, debug=False):
    gstr = "Galaxy" if galaxy else "Halo"
    cprint(f"---------------------------------")
    cprint(f" GUIDE: How to run {gstr}Extend")
    cprint(f"---------------------------------")
    cprint(f"  Hi, user!")
    cprint(f"  You may need to extend the {gstr.lower()} catalog.")
    cprint(f"  Before start, you must have the catalogs done.")
    extfile = f"run_extend_{gstr.lower()}.py"
    cprint()
    cprint(f"  Just run this file: `{extfile}`")
    
    cprint(f"{bs}  ex: $ python3 {extfile} -m nc -n 24 --verbose{be}")
    cprint(f"  Arguments:")
    cprint(f"    -m, --mode:     [Default=nc] nc, nh, nh2, ncdust2 (you can add in `inhouse` in `_extend_{gstr.lower()}.py`)")
    cprint(f"    -n, --nthread:  [Default= 8] Number of cores")
    cprint(f"    --verbose:      Turn on verbose for messages")
    cprint(f"    (If you want to save memory)")
    cprint(f"    -p, --partition:[Default= 0] Partitioning domain (0=No, 1=x, 2=xy, 3=xyz)")
    cprint(f"    (If you want to run parallel for outputs)")
    cprint(f"    -s, --sep:      [Optional]   Run only iout%nsep==sep")
    cprint(f"    -N, --nsep:     [Optional]   Run only iout%nsep==sep")
    if galaxy:
        cprint(f"    (If you don't need particular info)")
        cprint(f"    --nocell:       Turn off cell data")
        cprint(f"    --nochem:       Turn off chemical data")
    else:
        cprint(f"    (If you don't need particular info)")
        cprint(f"    --onlymem:      Use only member data")
    cprint()
    cprint(f"  {bs}You will find results as `{catrepo}/extended/xxxxx`{be}")
    cprint(f"---------------------------------")
    cprint(f"---------------------------------")
        
