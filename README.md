[1]: https://www.anaconda.com/
[2]: https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html

RAMSES Universal Reader
=============================

By San Han

A package for reading and computing various versions of RAMSES hydro and particle data.

Setup
-----------------------------

Download the code from bitbucket repository, or use clone it from the latest repository using git.
```bash
git clone https://github.com/sanhancluster/rur.git
```
### Installing
rur requires python version to be >= 3.8
#### Using pip
Use the following command to install the package using pip.
```bash
cd rur
pip install .
```
This requires numpy to be already installed.
#### Using Conda environment
Use the following command to install the package on current python environment.
```bash
cd rur
python3 setup.py install
```
if you want to develop the source code, 
use included bash script [f2py.sh](f2py.sh) and [conda develop][2] instead.
```bash
cd rur
./f2py.sh
conda develop .
```
[f2py.sh](f2py.sh) should be run whenever there is a change in Fortran (*.f90) files.

Usage
-----------------------------

### Reading a full volume data
```python
from rur import uri
iout = 136 # arbitrary snapshot number, from output_XXXXX
snap = uri.RamsesSnapshot('my_RAMSES_repo', iout, mode='none', path_in_repo='')

# reads cell and particle data in the selected bounding box, if the box is not specified, loads the whole volume.
cell = snap.get_cell()
part = snap.get_part()
```

### Reading data from halo position

If Rockstar halo finder data is located in `my_RAMSES_repo/path_to_files_in_repo`,
```python
from rur import uri
from rur.uhmi import Rockstar
iout = 136
snap = uri.RamsesSnapshot('my_RAMSES_repo', iout, mode='none')
rst = Rockstar.load(snap, path_in_repo='path_to_rockstar_in_repo')

target_halo = rst[123] # arbitrary halo number
snap.set_box_halo(target_halo, radius=1)
cell = snap.get_cell()
part = snap.get_part()
```
gives you cell and particle table of bounding box of the selected halo.

### Configuring cell and particle data column

Cell and particle column data differ by RAMSES versions. This can be configured by manually modifying 
[`config.py`](rur/config.py). Alternatively, cell column list can be changed by following code.
```python
from rur import uri
iout = 136
snap = uri.RamsesSnapshot('my_RAMSES_repo', iout, mode='none')
snap.hydro_names = ['rho', 'x', 'y', 'z', 'P', 'some', 'additional', 'columns']
cell = snap.get_cell()
```
Note that if you changed particle column in config.py, particle-reading subroutine in [`readr.f90`](rur/readr.f90) 
also need to be changed, or you will get an error.

### Drawing gas / particle map

Cell and particle data can be drawn directly from built-in module [`rur.painter`](rur/painter.py), which relies on 
`matplotlib.pyplot` module.
```python
from rur import uri, painter
import matplotlib.pyplot as plt
iout = 136
snap = uri.RamsesSnapshot('my_RAMSES_repo', iout, mode='none')

snap.set_box(center=[0.5, 0.5, 0.5], extent=[0.1, 0.2, 0.1]) # bounding box of the region to draw
proj = [0, 2]

plt.figure()
snap.get_part()
plt.subplot(121)
plt.title('Stars')
painter.draw_partmap(snap.part['star'], proj=proj, shape=1000)
painter.set_ticks_unit(snap, proj, 'kpc')

snap.get_cell()
plt.subplot(122)
plt.title('Gas')
painter.draw_gasmap(snap.cell, proj=proj, mode='rho', shape=1000)
plt.show()
```
![Tutorial](rur_tutorial.png)
