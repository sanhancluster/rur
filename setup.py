from numpy.distutils.core import Extension, setup
from os.path import dirname, abspath, join
from os import system

with open("README.md", "r") as fh:
    long_description = fh.read()

#mypath = dirname(abspath(__file__))
#system(join(mypath, 'f2py.sh'))

readr = Extension(name='readr', sources=['readr.f90'], language='f90')
readhtm = Extension(name='readhtm', sources=['readhtm.f90'], language='f90')
hilbert3d = Extension(name='hilbert3d', sources=['hilbert3d.f90'], language='f90')
if __name__ == "__main__":
    setup(
        name='rur',
        version='0.1',
        author="San Han",
        author_email="sn1994a@gmai.com",
        description="A package for reading and analyzing the RAMSES data",
        ext_modules=[readr, readhtm, hilbert3d],
        long_description=long_description
    )
