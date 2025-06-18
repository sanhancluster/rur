import numpy as np


bs = "\033[1m"
be = "\033[0;0m"
yess = ["yes", "Yes", "YES", "y", "Y"]

def vprint(txt, verbose):
    if verbose:
        print(txt)

def cprint(txt=""):
    print(f"  |{txt}")