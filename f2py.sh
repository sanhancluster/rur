#!/bin/bash

FILES='readr.f90 readhtm.f90 hilbert3d.f90 io_ramses.f90'
F2PY=f2py3
FORT=gfortran
BASEDIR=$(dirname "$0")

cd $BASEDIR/rur
for f in $FILES
do
    bn=$(basename "$f" .f90)
    $FORT -x f95-cpp-input -c $f
    $F2PY -c --f90exec=$FORT $f -m $bn --opt='-O3 -x f95-cpp-input'
done
rm *.o *.mod

