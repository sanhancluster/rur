#!/bin/bash

FILES='readr.f90 readhtm.f90 hilbert3d.f90'
F2PY=f2py
FORT=gfortran
BASEDIR=$(dirname "$0")

cd $BASEDIR/rur
for f in $FILES
do
    bn=$(basename "$f" .f90)
    $FORT -c $f
    $F2PY -c $f -m $bn --opt='-O3'
done
rm *.o *.mod

