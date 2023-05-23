#!/bin/bash

FILES='readr.f90 readhtm.f90 hilbert3d.f90 io_ramses.f90'
# FILES='readhtm.f90'
F2PY=f2py
FORT=gfortran # or "ifort"
MACHINE="gc" # or "tardis"
BASEDIR=$(dirname "$0")

cd $BASEDIR/rur
for f in $FILES
do
    bn=$(basename "$f" .f90)
    echo -e "\n\n\nCompiling $bn\n\n\n"
    if [ $FORT == "gfortran" ]; then
        # If not OMP, remove `-fopenmp` and `-lgomp`
        $FORT -x f95-cpp-input -c -fopenmp $f 
        if [ $MACHINE == "tardis" ]; then
            export CFLAGS="-fPIC -O2 -std=c99"
        fi
        $F2PY -lgomp --f90exec=$FORT --f77exec=$FORT --f90flags='-fopenmp -O3 -x f95-cpp-input'  -c $f -m $bn

    elif [ $FORT == "ifort" ]; then
        # If not OMP, remove `-fopenmp` and `-liomp5`
        $FORT -O3 -foptimize-sibling-calls -c $f
        if [ $MACHINE == "tardis" ]; then
            $F2PY -c $f -m $bn --compiler=intelem --fcompiler=intelem --opt='-O3 -heap-arrays -foptimize-sibling-calls -fpp -m64 -free -fopenmp -std=c99' -liomp5
        elif [ $MACHINE == "gc" ]; then
            $F2PY -c $f -m $bn --fcompiler=intelem --opt='-O3 -heap-arrays -foptimize-sibling-calls -fpp -m64 -free -fopenmp -D_GNU_SOURCE' -liomp5
        else
            echo "NotImplementedError: unknown machine '$MACHINE'"
            exit 2
        fi
    fi
done
rm *.o *.mod

# exit 0

For VR part
cd ..
FILES='find_domain_py.f90  get_flux_py.f90  get_ptcl_py.f90  jsamr2cell_py.f90  jsamr2cell_totnum_py.f90  js_gasmap_py.f90  js_getpt_ft.f90  jsrd_part_totnum_py.f90  jsrd_part_py.f90 js_getsfe_ft.f90'

FORT=gfortran
cd $BASEDIR/rur/vr/fortran
for f in $FILES
do
  bn=$(basename "$f" .f90)
  echo -e "\n\n\nCompiling $bn\n\n\n"
  if [$MACHINE == "tardis" ]; then
    export CFLAGS="-fPIC -O2 -std=c99"
  fi
  $F2PY -m $bn --fcompiler=$FORT --f90flags='-fopenmp' -lgomp -c $f
done
