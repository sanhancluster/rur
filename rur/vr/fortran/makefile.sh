#!/bin/bash

#f2py3 -m find_domain_py --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c find_domain_py.f90
#f2py3 -m get_ptcl_py --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c get_ptcl_py.f90
#f2py3 -m get_flux_py --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c get_flux_py.f90
#f2py3 -m jsamr2cell_totnum_py --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c jsamr2cell_totnum_py.f90
#f2py3 -m jsamr2cell_py --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c jsamr2cell_py.f90
f2py3 -m js_gasmap_py --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c js_gasmap_py.f90

#To be updated based on rur repo (https://bitbucket.org/sanhancluster/rur). Thanks San
#FILES='find_domain_py.f90'
#F2PY=f2py
#FORT=gfortran
#BASEDIR=$(dirname "$0")
#
#for f in $FILES
#do
#    bn=$(basename "$f" .f90)
#    $FORT -x f95-cpp-input -c $f
#    $F2PY -c --f90exec=$FORT $f -m $bn --opt='-O3 -x f95-cpp-input'
#done
#rm *.o *.mod

#!/bin/bash

#f2py3 -c find_domain_py.f90 -m find_domain_py -lgomp
#
#f2py3 -c get_ptcl_py.f90 -m get_ptcl_py -lgomp


