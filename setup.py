from numpy.distutils.core import Extension, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

readr = Extension(name='rur.readr', sources=['rur/readr.f90'], language='f90')
readhtm = Extension(name='rur.readhtm', sources=['rur/readhtm.f90'], language='f90')
hilbert3d = Extension(name='rur.hilbert3d', sources=['rur/hilbert3d.f90'], language='f90')
if __name__ == "__main__":
    setup(
        name='rur',
        version='0.1',
        author="San Han",
        author_email="sn1994a@gmai.com",
        description="A package for reading and analyzing the RAMSES data",
        packages=['rur'],
        ext_packagess=['rur'],
        ext_modules=[readr, readhtm, hilbert3d],
        long_description=long_description,
        package_data={'rur': ['colormaps/data/*.csv']},
    )
