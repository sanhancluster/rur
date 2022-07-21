#!/bin/bash

FILES='readr.f90 readhtm.f90 hilbert3d.f90 io_ramses.f90'
F2PY=f2py
FORT=gfortran
BASEDIR=$(dirname "$0")

cd $BASEDIR/rur
for f in $FILES
do
    bn=$(basename "$f" .f90)
    $FORT -x f95-cpp-input -c $f
    $F2PY -c --f90exec=$FORT --f77exec=$FORT $f -m $bn --opt='-O3 -x f95-cpp-input'
done
rm *.o *.mod

#For VR part
cd ..
FILES='find_domain_py.f90  get_flux_py.f90  get_ptcl_py.f90  jsamr2cell_py.f90  jsamr2cell_totnum_py.f90  js_gasmap_py.f90  js_getpt_ft.f90'

cd $BASEDIR/rur/vr/fortran
for f in $FILES
do
    bn=$(basename "$f" .f90)
    $F2PY -m $bn --fcompiler=$FORT --f90flags='-fopenmp' -lgomp -c $f
done

