import numpy as np


bs = "\033[1m"
be = "\033[0;0m"
yess = ["yes", "Yes", "YES", "y", "Y"]

def vprint(txt, verbose):
    if verbose:
        print(txt)

def cprint(txt=""):
    print(f"  |{txt}")

def filemode(fname):
    import os
    # import pwd
    # import grp
    import stat
    file_stat = os.stat(fname)
    # Get owner UID and GID
    uid = file_stat.st_uid
    gid = file_stat.st_gid

    # Get username and group name
    # try:
    #     user = pwd.getpwuid(uid).pw_name
    # except:
    #     user = str(uid)
    # group = grp.getgrgid(gid).gr_name

    # Get file permission in octal
    permissions = bool(file_stat.st_mode & stat.S_IWGRP)
    return uid, gid, permissions