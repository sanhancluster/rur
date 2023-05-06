#!/bin/bash

FILES='readr.f90 readhtm.f90 hilbert3d.f90 io_ramses.f90'
#FILES='readr.f90'
F2PY=f2py
FORT=gfortran
BASEDIR=$(dirname "$0")

cd $BASEDIR/rur
for f in $FILES
do
    bn=$(basename "$f" .f90)
    if [[ $FORT == "gfortran" ]]; then
        # OMP on
	$FORT -x f95-cpp-input -c -fopenmp $f
	# In tardis
	export CFLAGS="-fPIC -O2 -std=c99"
	$F2PY -lgomp --f90exec=$FORT --f77exec=$FORT --f90flags='-fopenmp -O3 -x f95-cpp-input'  -c $f -m $bn
	# In GC
        #$F2PY -lgomp --f90exec=$FORT --f77exec=$FORT --f90flags='-fopenmp -O3 -x f95-cpp-input'  -c $f -m $bn
        # OMP off
        # $FORT -x f95-cpp-input -c $f
        # $F2PY -c -lgomp --f90exec=$FORT --f77exec=$FORT $f -m $bn --opt='-O3 -x f95-cpp-input'
    fi

    if [[ $FORT == "ifort" ]]; then
        # OMP on
        $FORT -O3 -foptimize-sibling-calls -c $f
	# In tardis
	$F2PY -c $f -m $bn --compiler=intelem --fcompiler=intelem --opt='-O3 -heap-arrays -foptimize-sibling-calls -fpp -m64 -free -fopenmp -std=c99' -liomp5
	# In GC
        #$F2PY -c $f -m $bn --fcompiler=intelem --opt='-O3 -heap-arrays -foptimize-sibling-calls -fpp -m64 -free -fopenmp -D_GNU_SOURCE' -liomp5
        # OMP off
        # $FORT -O3 -foptimize-sibling-calls -c $f
        # $F2PY -c $f -m $bn --fcompiler=intelem --opt='-O3 -heap-arrays -foptimize-sibling-calls -fpp -m64 -free' -liomp5
    fi
done
rm *.o *.mod


#For VR part
cd ..
FILES='find_domain_py.f90  get_flux_py.f90  get_ptcl_py.f90  jsamr2cell_py.f90  jsamr2cell_totnum_py.f90  js_gasmap_py.f90  js_getpt_ft.f90  jsrd_part_totnum_py.f90  jsrd_part_py.f90'

#FORT=gfortran
#cd $BASEDIR/rur/vr/fortran
#for f in $FILES
#do
#   bn=$(basename "$f" .f90)
#   $F2PY -m $bn --fcompiler=$FORT --f90flags='-fopenmp' -lgomp -c $f
#done

#For VR part
