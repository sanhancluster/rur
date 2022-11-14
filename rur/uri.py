
from os.path import join, exists, getsize
from numpy.core.records import fromarrays as fromarrays

from scipy.integrate import cumtrapz
from collections.abc import Iterable
from scipy.spatial import cKDTree as KDTree
from numpy.lib.recfunctions import append_fields

from rur.fortranfile import FortranFile
from rur.hilbert3d import hilbert3d
from rur.readr import readr
from rur.io_ramses import io_ramses
from rur.config import *
from rur import utool
import numpy as np
import warnings
import glob
import re

class TimeSeries(object):
    """
    A class to manage multiple snapshots in the same repository
    """
    def __init__(self, snap):
        self.snaps = {}
        self.basesnap = snap
        self.snaps[snap.iout] = snap
        self.icoarse_table = None

    def get_snap(self, iout):
        if(iout in self.snaps):
            return self.snaps[iout]
        else:
            self.snaps[iout] = self.basesnap.switch_iout(iout)
            return self.snaps[iout]

    def __getitem__(self, item):
        return self.get_snap(item)

    def __getattr__(self, item):
        return self.basesnap.__getattribute__(item)

    def set_box_halo(self, halo, radius=1, use_halo_radius=True, radius_name='rvir', iout_name='timestep'):
        snap = self.get_snap(halo[iout_name])
        snap.set_box_halo(halo, radius=radius, use_halo_radius=use_halo_radius, radius_name=radius_name)
        return snap

    def icoarse_to_aexp(self, icoarse):
        if(self.icoarse_table is None):
            output_names = glob.glob(join(self.basesnap.snap_path, output_glob))
            iouts = [int(arr[-5:]) for arr in output_names]
            iouts.sort()
            self.iout_table = iouts
            icoarses = []
            aexps = []
            for iout in iouts:
                snap = self[iout]
                icoarses.append(snap.nstep_coarse)
                aexps.append(snap.aexp)
            self.icoarse_table = np.array(icoarses)
            self.aexp_table = np.array(aexps)

        return np.interp(icoarse, self.icoarse_table, self.aexp_table)

    def write_iout_avail(self, use_cache=False):
        path = join(self.repo, 'list_iout_avail.txt')
        timer.start("Writing available timesteps in %s..." % path, 1)
        iouts = self.basesnap.get_iout_avail()
        if(use_cache and exists(path)):
            self.read_iout_avail()
        iout_table = np.zeros(len(iouts), dtype=iout_avail_dtype)
        i = 0
        for iout in iouts:
            if(use_cache and iout in self.iout_avail['iout']):
                iout_table[i] = self.iout_avail[np.searchsorted(self.iout_avail['iout'], iout)]
            else:
                try:
                    snap = self.get_snap(iout)
                except:
                    continue
                iout_table[i]['iout'] = iout
                iout_table[i]['aexp'] = snap.aexp
                iout_table[i]['age'] = snap.age
                iout_table[i]['icoarse'] = snap.nstep_coarse
                iout_table[i]['time'] = snap.time
                i += 1
        iout_table = iout_table[:i]
        names = iout_table.dtype.names
        self.iout_avail = iout_table
        np.savetxt(path, iout_table,
                   fmt='%18d %18.9e %18.9e %18d %18.9e', header=('%16s'+' %18s'*(len(names)-1)) % names)
        timer.record()

    def read_iout_avail(self):
        self.iout_avail = np.loadtxt(join(self.repo, 'list_iout_avail.txt'), dtype=iout_avail_dtype)

    def clear(self):
        self.snaps = None
        self.basesnap = None

class RamsesSnapshot(object):
    """A handy object to store RAMSES AMR/Particle snapshot data.

    One RamsesSnapshot represents single snapshot of the simulation. Efficient reads from RAMSES binary are guaranteed
    by only touching necessary files.

    Parameters
    ----------
    repo : string
        Repository path, path to directory where RAMSES raw data is stored.
    iout : integer
        RAMSES output number
    mode : 'none', 'yzics', 'nh', 'gem', etc.
        RAMSES version. Define it in config.py if necessary
    box : numpy.array with shape (3, 2)
        Bounding box of region to open in the simulation, in code unit, should be within (0, 1)
    path_in_repo : string
        Path to actual snapshot directories in the repository path, path where 'output_XXXXX' is stored.
    full_path : string
        Overrides repo and path_in_repo, full path to directory where 'output_XXXXX' is stored.
    snap : RamsesSnapshot object
        Previously created snapshot object from same simulation (but different output number) saves time to compute
        cosmology.

    Attributes
    ----------
    part : RamsesSnapshot.Particle object
        Loaded particle table in the box
    cell : RamsesSnapshot.Cell object
        Loaded particle table in the box
    box : numpy.array with shape (3, 2)
        Current bounding box. Change this value if bounding box need to be changed.
    part_data : numpy.recarray
        Currently cached particle data, if data requested by get_part() have already read, retrieve from this instead.
    cell_data : numpy.recarray
        Currently cached cell data, if data requested by get_cell() have already read, retrieve from this instead.

    Examples
    ----------
    To read data from ramses raw binary data:

    >>> from rur import uri
    >>> repo = "path/to/your/ramses/outputs"
    >>> snap = uri.RamsesSnapshot(repo, iout=1, path_in_repo='')
    >>> snap.box = np.array([[0, 1], [0, 1], [0, 1]])

    To read particle and cell data:

    >>> snap.get_part()
    >>> snap.get_cell()
    >>> print(snap.part.dtype)
dtype((numpy.record, [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'), ('m', '<f8'), ('epoch', '<f8'), ('metal', '<f8'), ('id', '<i4'), ('level', 'u1'), ('cpu', '<i4')]))

    >>> print(snap.cell.dtype)
dtype((numpy.record, [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('rho', '<f8'), ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'), ('P', '<f8'), ('metal', '<f8'), ('zoom', '<f8'), ('level', '<i4'), ('cpu', '<i4')]))

    >>> snap.clear()

    """

    def __init__(self, repo, iout, mode='none', box=None, path_in_repo=default_path_in_repo['snapshots'], snap=None, longint=False):
        self.repo = repo
        self.path_in_repo = path_in_repo
        self.snap_path = join(repo, path_in_repo)

        if(iout<0):
            iouts = self.get_iout_avail()
            iout = iouts[iout]
        self.iout = iout

        self.path = join(self.snap_path, output_format.format(snap=self))
        self.params = {}

        self.mode = mode
        self.info_path = join(self.path, info_format[mode].format(snap=self))
        self.data_path = join(self.path, data_format[mode].format(snap=self))

        self.part_data = None
        self.cell_data = None

        self.part = None
        self.cell = None

        self.pcmap = None

        self.longint = longint

        if(self.longint is None):
            if(mode == 'fornax'):
                self.longint = True
            else:
                self.longint = False

        if(mode == 'ng'):
            self.classic_format = False
        else:
            self.classic_format = True

        self.read_params(snap)
        if(box is not None):
            self.box = np.array(box)
        else:
            self.box = default_box
        self.region = BoxRegion(self.box)
        self.box_cell = None
        self.box_part = None

    def get_iout_avail(self):
        output_names = glob.glob(join(self.snap_path, output_glob))
        iouts = [int(arr[-5:]) for arr in output_names]
        return np.sort(iouts)

    def switch_iout(self, iout):
        # returns to other snapshot while maintaining repository, box, etc.
        return RamsesSnapshot(self.repo, iout, self.mode, self.box, self.path_in_repo, snap=self, longint=self.longint)

    def __getitem__(self, item):
        return self.params[item]

    def __getattr__(self, item):
        if item.startswith('__') and item.endswith('__'):
            return super(RamsesSnapshot, self).__getattr__(item)
        return self.__getitem__(item)

    def get_path(self, type='amr', icpu=1):
        return self.data_path.format(type=type, icpu=icpu)

    def set_cosmology(self, params=None, n=5000, snap=None):
        # calculates cosmology table with given cosmology paramters
        if(params is None):
            params = self.params

        if(snap is None):
            # Integrate manually because astropy cosmology calculation is too slow...
            aarr = np.linspace(0, 1, n)[1:] ** 2
            aarr_st = (aarr[:-1] + aarr[1:])/2
            duda = 1. / (aarr_st ** 3 * np.sqrt(params['omega_m'] * aarr_st ** -3 + params['omega_l']))
            dtda = 1. / (params['H0'] * km * Gyr / Mpc * aarr_st * np.sqrt(params['omega_m'] * aarr_st ** -3 + params['omega_l']))
            aarr = aarr[1:]

            uarr = cumtrapz(duda[::-1], aarr[::-1], initial=0)[::-1]
            tarr = cumtrapz(dtda, aarr, initial=0)
            self.cosmo_table = fromarrays([aarr, tarr, uarr], dtype=[('aexp', 'f8'), ('t', 'f8'), ('u', 'f8')])
        else:
            self.cosmo_table = snap.cosmo_table

        self.params['age'] = np.interp(params['time'], self.cosmo_table['u'], self.cosmo_table['t'])
        self.params['lookback_time'] = self.cosmo_table['t'][-1] - self.params['age']

        if(timer.verbose>=1):
            print('Age of the universe (now/z=0): %.3f / %.3f Gyr, z = %.5f (a = %.4f)' % (self.params['age'], self.cosmo_table['t'][-1], params['z'], params['aexp']))

    def aexp_to_age(self, aexp):
        return np.interp(aexp, self.cosmo_table['aexp'], self.cosmo_table['t'])

    def age_to_aexp(self, age):
        return np.interp(age, self.cosmo_table['t'], self.cosmo_table['aexp'])

    def epoch_to_age(self, epoch):
        table = self.cosmo_table
        ages = np.interp(epoch, table['u'], table['t'])
        return ages

    def epoch_to_aexp(self, epoch):
        table = self.cosmo_table
        aexp = np.interp(epoch, table['u'], table['aexp'])
        return aexp

    def aexp_to_dtdu(self, aexp):
        # returns dt over du (derivative of proper time t in function of ramses time unit u)
        return aexp**2 / (self['H0'] * km * Gyr / Mpc)

    def set_extra_fields(self):
        self.cell_extra, self.part_extra = custom_extra_fields(self)

    def set_unit(self):
        custom_units(self)

    def set_box(self, center, extent, unit=None):
        """set center and extent of the current target bounding box of the simulation.
        if unit is None, it is recognized as code unit
        """
        if(unit is not None):
            extent = extent / self.unit[unit]
            center = center / self.unit[unit]
        self.box = get_box(center, extent)
        if(self.box.shape != (3, 2)):
            raise ValueError("Incorrect box shape: ", self.box.shape)
        self.region = BoxRegion(self.box)

    def set_box_halo(self, halo, radius=1, use_halo_radius=True, radius_name='rvir'):
        if(isinstance(halo, np.ndarray)):
            warnings.warn("numpy.ndarray is passed instead of np.void in halo parameter. Assuming first row as input halo...", UserWarning)
            halo = halo[0]
        center = get_vector(halo)
        if(use_halo_radius):
            extent = halo[radius_name] * radius * 2
        else:
            extent = radius * 2
        self.set_box(center, extent)

    def read_params(self, snap):

        opened = open(self.info_path, mode='r')

        int_regex = re.compile(r'(?P<name>\w+)\s*=\s*(?P<data>\d+)')
        float_regex = re.compile(r'(?P<name>\w+)\s*=\s*(?P<data>.+)')
        str_regex = re.compile(r'(?P<name>\w+)\s*=\s*(?P<data>.+)')
        domain_regex = re.compile(r'(?P<domain>.+)\s+(?P<ind_min>.+)\s+(?P<ind_max>.+)')

        params = {}
        # read integer data
        for _ in range(6):
            line = opened.readline().strip()
            matched = int_regex.search(line)
            if(not matched):
                raise ValueError("A line in the info file is not recognized: %s" % line)
            params[matched.group('name')] = int(matched.group('data'))

        opened.readline()

        # read float data
        for _ in range(11):
            line = opened.readline().strip()
            matched = float_regex.search(line)
            if(not matched):
                raise ValueError("A line in the info file is not recognized: %s" % line)
            params[matched.group('name')] = float(matched.group('data'))

        # some cosmological calculations
        params['unit_m'] = params['unit_d'] * params['unit_l']**3
        params['h'] = params['H0']/100
        params['z'] = 1/params['aexp'] - 1
        params['boxsize'] = params['unit_l'] * params['h'] / Mpc / params['aexp']
        params['boxsize_physical'] = params['boxsize'] / (params['h']) * params['aexp']
        params['boxsize_comoving'] = params['boxsize'] / (params['h'])

        self.part_dtype = part_dtype[self.mode]
        self.hydro_names = hydro_names[self.mode]
        if(self.mode == 'nh2')&(self.iout<60):
            self.hydro_names = hydro_names['y3']
            self.part_dtype = part_dtype['y3']

        if(self.classic_format):
            opened.readline()
            line = opened.readline().strip()
            params['ordering'] = str_regex.search(line).group('data')
            opened.readline()
            if(params['ordering'] == 'hilbert'):
                bound_key = []
                for icpu in np.arange(1, params['ncpu']):
                    line = opened.readline()
                    bound_key.append(float(domain_regex.search(line).group('ind_max')))
                self.bound_key = np.array(bound_key)

            if(exists(self.get_path('part', 1))):
                self.params['nstar'] = self._read_nstar()
                self.params['star'] = self.params['nstar']>0
            else:
                self.params['nstar'] = 0
                self.params['star'] = False

            # check if star particle exists
            if(not self.params['star']):
                # This only applies to old RAMSES particle format
                if(self.mode == 'nh'):
                    self.part_dtype = part_dtype['nh_dm_only']
                elif (self.mode == 'yzics'):
                    self.part_dtype = part_dtype['yzics_dm_only']
            if(self.longint):
                if(self.mode == 'iap' or self.mode == 'gem' or self.mode == 'fornax' or self.mode == 'y2' or self.mode == 'y3' or self.mode == 'y4' or self.mode == 'nc'):
                    self.part_dtype = part_dtype['gem_longint']
        else:
            self.params['star'] = True

        # initialize cpu list and boundaries
        self.cpulist_cell = np.array([], dtype='i4')
        self.cpulist_part = np.array([], dtype='i4')
        self.bound_cell = np.array([0], dtype='i4')
        self.bound_part = np.array([0], dtype='i4')
        self.params.update(params)

        self.set_cosmology(snap=snap)
        self.set_extra_fields()
        self.set_unit()

    def get_involved_cpu(self, box=None, binlvl=None, n_divide=5):
        """Get the list of involved cpu domain for specific region.

        """
        if(box is None):
            box = self.box
        if(self.classic_format and not box is None):
            box = np.array(box)
            maxlvl = self.params['levelmax']

            involved_cpu = get_cpulist(box, binlvl, maxlvl, self.bound_key, self.ndim, n_divide)
        else:
            involved_cpu = np.arange(self.params['ncpu']) + 1
        return involved_cpu


    def read_sink_table(self):
        if(self.mode=='nh'):
            table = np.genfromtxt(self.path+'/sink_%05d.csv' % self.iout, dtype=sink_table_dtype, delimiter=',')
        else:
            raise ValueError('This function works only for NH-version RAMSES')
        return table

    def read_specific(self, pname:str, cpulist:Iterable, mode:str, target_fields=None):
        # 0) Mode check
        import os
        modes = ['hagn', 'yzics', 'nh', 'fornax', 'y2', 'y3', 'y4', 'nc', 'nh2']
        if mode not in modes:
            raise ValueError(f"{mode} is not supported! \n(currently only {modes} are available)")
        
        # 1) Read or skip functions
        def readorskip_real(f:FortranFile, dtype:type, key:str, search:Iterable):
            if key in search:
                return f.read_reals(dtype)
            else:
                f.skip_records()
        def readorskip_int(f:FortranFile, dtype:type, key:str, search:Iterable):
            if key in search:
                return f.read_ints(dtype)
            else:
                f.skip_records()
        
        # 2) Chemical elements list
        chems = {
            'hagn':['H','O','Fe', 'C', 'N', 'Mg', 'Si'], 
            'yzics':[], 'nh':[], "fornax":[], "y2":[], 
            "y3":['H', 'O', 'Fe', 'Mg', 'C', 'N', 'Si', 'S'], 
            "y4":['H', 'O', 'Fe', 'Mg', 'C', 'N', 'Si', 'S', 'D'], 
            "nc":['H', 'O', 'Fe', 'Mg', 'C', 'N', 'Si', 'S', 'D'],
            'nh2':['H', 'O', 'Fe', 'Mg', 'C', 'N', 'Si', 'S', 'D']}
        chem = chems[mode]

        # 3) Check numbers of particles from txt (or from output file)
        files = glob.glob(f"{self.snap_path}/output_{self.iout:05d}/part*out*")
        header = f"{self.snap_path}/output_{self.iout:05d}/header_{self.iout:05d}.txt"
        sinkinfo = f"{self.snap_path}/output_{self.iout:05d}/sink_{self.iout:05d}.info"

        isfamily = False
        if os.path.isfile(header): # (NH, NH2, Fornax, NC)
            with open(header, "rt") as f:
                temp = f.readline()
                if "Family" in temp: # (Fornax, NH2, NC)
                    isfamily = True
                    ntracer_tot = int( f.readline()[14:] ) # other_tracer
                    for _ in range(5):
                        # tracers of debris, cloud, star, other, gas
                        ntracer_tot += int( f.readline()[14:] ) # debris_tracer
                    ndm_tot = int( f.readline()[14:] )
                    nstar_tot = int( f.readline()[14:] )
                    ncloud_tot = int( f.readline()[14:] )
                    npart_tot = ntracer_tot + ndm_tot + nstar_tot + ncloud_tot
                    for _ in range(3):
                        # debris, other, undefined
                        npart_tot += int( f.readline()[14:] ) # debris
                    
                    if os.path.isfile(sinkinfo): # (NH2, NC)
                        with open(sinkinfo, 'rt') as f:
                            nsink_tot = int(f.readline().split()[-1])
                    else: # (Fornax)
                        with FortranFile(f"{files[0]}", mode='r') as f:
                            f.skip_records(7)
                            nsink_tot = f.read_ints(np.int32)[0]
                else: # (NH)
                    npart_tot = int(f.readline()); f.readline()
                    ndm_tot = int(f.readline()); f.readline()
                    nstar_tot = int(f.readline()); f.readline()
                    nsink_tot = int(f.readline()); f.readline()
                    ncloud_tot = nsink_tot * 2109
                    ntracer_tot = 0
        else: # (hagn, yzics)
            with FortranFile(f"{files[0]}", mode='r') as f:
                f.skip_records(4) # ncpu, ndim, npart, localseed(+tracer_seed)
                nstar_tot = f.read_ints(np.int32)[0] # nstar
                f.skip_records(2) # mstar_tot, mstar_lost
                nsink_tot = f.read_ints(np.int32)[0] # nsink
                ncloud_tot = 2109 * nsink_tot
            ndm_tot = 0
            ntracer_tot = 0
            if(pname == 'dm'):
                npart_tot = 0
                for file in files:
                    with FortranFile(f"{file}", mode='r') as f:
                        f.skip_records(2)
                        npart_tot += f.read_ints(np.int32)[0]
                ndm_tot = npart_tot - nstar_tot - ncloud_tot

        # 4) Allocate base array
        part_dtype = self.part_dtype
        if target_fields is not None:
            if 'cpu' not in target_fields:
                target_fields = np.append(target_fields, 'cpu')
            part_dtype = [idtype for idtype in part_dtype if idtype[0] in target_fields]
        else:
            target_fields = [idtype[0] for idtype in part_dtype]
        tracers = ["tracer","cloud_tracer","star_tracer","gas_tracer"]
        if(pname == 'star'):
            size = nstar_tot
        elif(pname == 'dm'):
            size = ndm_tot
        elif(pname == 'sink'):
            size = ncloud_tot
        elif(pname in tracers):
            size = ntracer_tot
        else:
            raise ValueError(f"{pname} is currently not supported!")
        part = np.empty(size, dtype=part_dtype)

        # 5) Read output part files
        cursor = 0
        temp = 0
        for fname in files:
            icpu = int( fname[-5:] )
            if icpu in cpulist:
                with FortranFile(f"{fname}", mode='r') as f:
                    # Read fortran file
                    #   (read) header
                    f.skip_records(8) # ncpu, ndim, npart, localseed(+tracer_seed), nstar, mstar_tot, mstar_lost, nsink
                    #   (read) pos, vel, m
                    x = readorskip_real(f, np.float64, 'x', target_fields)
                    y = readorskip_real(f, np.float64, 'y', target_fields)
                    z = readorskip_real(f, np.float64, 'z', target_fields)
                    vx = readorskip_real(f, np.float64, 'vx', target_fields)
                    vy = readorskip_real(f, np.float64, 'vy', target_fields)
                    vz = readorskip_real(f, np.float64, 'vz', target_fields)
                    m = f.read_reals(np.float64)
                    #   (read) id, level
                    id = f.read_ints(np.int32) # id
                    level = readorskip_int(f, np.int32, 'level', target_fields)
                    #   (read) family, tag
                    if isfamily:
                        family = f.read_ints(np.int8) # family
                        tag = readorskip_int(f, np.int8, 'tag', target_fields) # tag
                    else: pass                    
                    #   (read) epoch, metal
                    epoch = f.read_reals(np.float64) # epoch
                    metal = readorskip_real(f, np.float64, 'metal', target_fields)
                    #############################################
                    ###     Mask stars (From `classify_part`)
                    #############################################
                    if(isfamily):
                        # Do a family-based classification
                        mask = np.isin(family, part_family[pname])
                    elif(nstar_tot>0):
                        # Do a parameter-based classification
                        if(pname == 'dm'):
                            mask = (epoch == 0) & (id > 0)
                        elif(pname == 'star'):
                            mask = ((epoch < 0) & (id > 0))\
                                   | ((epoch != 0) & (id < 0))
                        elif(pname == 'sink' or pname == 'cloud'):
                            mask = (id < 0) & (m > 0) & (epoch == 0)
                        elif(pname in tracers):
                            mask = (id < 0) & (m == 0)
                        else:
                            mask = False
                    else:
                        # DM-only simulation
                        if(pname == 'dm'):
                            mask =  id > 0
                        elif(pname in tracers):
                            mask = (id < 0) & (m == 0)
                        else:
                            mask = False
                    #############################################
                    
                    # Write masked data only for requested columns
                    nsize = len(mask[mask])
                    temp += nsize
                    if('x' in target_fields):part['x'][cursor:cursor+nsize] = x[mask]
                    if('y' in target_fields):part['y'][cursor:cursor+nsize] = y[mask]
                    if('z' in target_fields):part['z'][cursor:cursor+nsize] = z[mask]
                    if('vx' in target_fields):part['vx'][cursor:cursor+nsize] = vx[mask]
                    if('vy' in target_fields):part['vy'][cursor:cursor+nsize] = vy[mask]
                    if('vz' in target_fields):part['vz'][cursor:cursor+nsize] = vz[mask]
                    if('m' in target_fields):part['m'][cursor:cursor+nsize] = m[mask]
                    if('epoch' in target_fields):part['epoch'][cursor:cursor+nsize] = epoch[mask]
                    if('metal' in target_fields):part['metal'][cursor:cursor+nsize] = metal[mask]
                    if('id' in target_fields):part['id'][cursor:cursor+nsize] = id[mask]
                    if('level' in target_fields):part['level'][cursor:cursor+nsize] = level[mask]
                    if('family' in target_fields):part['family'][cursor:cursor+nsize] = family[mask]
                    if('tag' in target_fields):part['tag'][cursor:cursor+nsize] = tag[mask]
                    newtypes = ["m0", "rho0", "partp"] + chem
                    if True in np.isin(newtypes, target_fields):
                        # (read & write) initial mass
                        if(mode=='y2') or (mode=='y3') or (mode=='y4') or (mode=='nc'):
                            if('m0' in target_fields): part['m0'][cursor:cursor+nsize] = f.read_reals(np.float64)[mask]
                            else: f.read_reals(np.float64)
                            # (read & write) chemical elements
                        if(mode=='y2') or (mode=='y3') or (mode=='y4') or (mode=='nc') or (mode=='hagn'):
                            if len(chem)>0:
                                for ichem in chem:
                                    if(ichem in target_fields): part[ichem][cursor:cursor+nsize] = f.read_reals(np.float64)[mask]
                                    else: f.read_reals(np.float64)
                        # (read & write) stellar density at formation
                        if(mode=='y3') or (mode=='y4') or (mode=='nc'):
                            if('rho0' in target_fields): part['rho0'][cursor:cursor+nsize] = f.read_reals(np.float64)[mask]
                            else: f.read_reals(np.float64)
                        # (read & write) parent indices
                        if(mode=='y2') or (mode=='y3') or (mode=='y4') or (mode=='nc'):
                            if('partp' in target_fields): part['partp'][cursor:cursor+nsize] = f.read_ints(np.int32)[mask]
                            else: f.read_ints(np.int32)
                    # Write cpu info
                    part['cpu'][cursor:cursor+nsize] = icpu

                    cursor += nsize
        return part[:cursor]

    def read_part(self, target_fields=None, cpulist=None, pname=None):
        """Reads particle data from current box.

        Parameters
        ----------
        target_fields: list of string
            target field name to read. If None is passed, read all fields.
        cpulist: list of integer
            target list of cpus, if specified, read selected cpus regardless of current box.
        Returns
        -------
        part : RamsesSnapshot.Particle object
            particle data, can be accessed as attribute also.

        """
        if(cpulist is None):
            cpulist = self.get_involved_cpu()
        if (self.part_data is not None):
            if pname == self.part.ptype:
                if(timer.verbose>=1): print('Searching for extra files...')
                cpulist = cpulist[np.isin(cpulist, self.cpulist_part, assume_unique=True, invert=True)]

        if (cpulist.size > 0):
            filesize = 0
            for icpu in cpulist:
                filesize += getsize(self.get_path('part', icpu))
            timer.start('Reading %d part files (%s) in %s... ' % (cpulist.size, utool.format_bytes(filesize), self.path), 1)
            if pname is not None:
                part = self.read_specific(pname, cpulist, self.mode, target_fields=target_fields)
            else:
                progress_bar = cpulist.size > progress_bar_limit and timer.verbose >= 1
                mode = self.mode
                if mode == 'nc':
                    mode = 'y4'
                readr.read_part(self.snap_path, self.iout, cpulist, mode, progress_bar, self.longint)
                timer.record()

                timer.start('Building table for %d particles... ' % readr.integer_table.shape[1], 1)
                dtype = self.part_dtype
                if(target_fields is not None):
                    if(self.longint):
                        arr = [*readr.real_table, *readr.long_table, *readr.integer_table, *readr.byte_table]
                    else:
                        arr = [*readr.real_table, *readr.integer_table, *readr.byte_table]

                        target_idx = np.where(np.isin(np.dtype(dtype).names, target_fields))[0]
                        arr = [arr[idx] for idx in target_idx]
                        dtype = [dtype[idx] for idx in target_idx]
                    part = fromarrays(arr, dtype=dtype)
                else:
                    if(self.longint):
                        arrs = [readr.real_table.T, readr.long_table.T, readr.integer_table.T, readr.byte_table.T]
                    else:
                        arrs = [readr.real_table.T, readr.integer_table.T, readr.byte_table.T]
                    part = fromndarrays(arrs, dtype)

                readr.close()
            bound = compute_boundary(part['cpu'], cpulist)
            if (self.part_data is None):
                self.part_data = part
            else:
                self.part_data = np.concatenate([self.part_data, part])

            self.bound_part = np.concatenate([self.bound_part[:-1], self.bound_part[-1] + bound])
            self.cpulist_part = np.concatenate([self.cpulist_part, cpulist])
            timer.record()

        else:
            if (timer.verbose >= 1):
                print('CPU list already satisfied.')

    def read_cell(self, target_fields=None, cpulist=None):
        """Reads amr data from current box.

        Parameters
        ----------
        target_fields: list of string
            target field name to read. If None is passed, read all fields.
        cpulist: list of integer
            target list of cpus, if specified, read selected cpus regardless of current box.
        Returns
        -------
        cell : RamsesSnapshot.Cell object
            amr data, can be accessed as attribute also.

        """
        if(cpulist is None):
            cpulist = self.get_involved_cpu()
        else:
            cpulist = np.array(cpulist)
        if(self.cell_data is not None):
            if(timer.verbose>=1):
                print('Searching for extra files...')
            cpulist = cpulist[np.isin(cpulist, self.cpulist_cell, assume_unique=True, invert=True)]

        if(cpulist.size > 0):
            filesize = 0
            for icpu in cpulist:
                filesize += getsize(self.get_path('amr', icpu))
                filesize += getsize(self.get_path('hydro', icpu))
            timer.start('Reading %d AMR & hydro files (%s) in %s... ' % (cpulist.size, utool.format_bytes(filesize), self.path), 1)

            progress_bar = cpulist.size > progress_bar_limit and timer.verbose >= 1
            readr.read_cell(self.snap_path, self.iout, cpulist, self.mode, progress_bar)
            self.params['nhvar'] = int(readr.nhvar)
            timer.record()

            formats = ['f8'] * self.params['ndim'] + ['f8'] * self.params['nhvar'] + ['i4'] * 2
            names = list(dim_keys[:self.params['ndim']]) + self.hydro_names + ['level', 'cpu']

            arr = [*readr.real_table, *readr.integer_table]

            if(len(arr) != len(names)):
                raise ValueError(
                    "Number of fields and size of the hydro array does not match\n"
                    "Consider changing the content of RamsesSnapshot.hydro_names\n"
                    "Recieved: %d, Desired: %d" % (len(names), len(arr)))

            timer.start('Building table for %d cells... ' % (arr[0].size), 1)
            if target_fields is not None:
                target_idx = np.where(np.isin(names, target_fields))[0]
                arr = [arr[idx] for idx in target_idx]
                formats = [formats[idx] for idx in target_idx]
                names = [names[idx] for idx in target_idx]

                cell = fromarrays(arr, formats=formats, names=names)
            else:
                dtype = np.format_parser(formats=formats, names=names, titles=None).dtype
                arrs = [readr.real_table.T, readr.integer_table.T]
                cell = fromndarrays(arrs, dtype)
            readr.close()

            bound = compute_boundary(cell['cpu'], cpulist)
            if (self.cell_data is None):
                self.cell_data = cell
            else:
                self.cell_data = np.concatenate([self.cell_data, cell])

            self.bound_cell = np.concatenate([self.bound_cell[:-1], self.bound_cell[-1] + bound])
            self.cpulist_cell = np.concatenate([self.cpulist_cell, cpulist])
            timer.record()

        else:
            if(timer.verbose>=1):
                print('CPU list already satisfied.')

    def read_ripses(self, target_fields=None, cpulist=None):
        """Reads ripses output data from current box.

        Parameters
        ----------
        target_fields: list of string
            target field name to read. If None is passed, read all fields.
        cpulist: list of integer
            target list of cpus, if specified, read selected cpus regardless of current box.
        Returns
        -------
        cell : RamsesSnapshot.Cell object
            amr data, can be accessed as attributes also.

        """
        if(cpulist is None):
            cpulist = self.get_involved_cpu()
        else:
            cpulist = np.array(cpulist)
        if(self.cell_data is not None):
            if(timer.verbose>=1):
                print('Searching for extra files...')
            cpulist = cpulist[np.isin(cpulist, self.cpulist_cell, assume_unique=True, invert=True)]

        if(cpulist.size > 0):
            timer.start('Reading %d grid files in %s... ' % (cpulist.size, self.path), 1)

            io_ramses.read_ripses_cell(self.snap_path, self.iout, cpulist)
            self.params['nhvar'] = int(io_ramses.nvar)
            timer.record()

            formats = ['f8'] * self.params['ndim'] + ['f8'] * self.params['nhvar'] + ['i4'] * 2

            names = list(dim_keys[:self.params['ndim']]) + self.hydro_names + ['level', 'cpu']

            arr = [*io_ramses.xc.T, *io_ramses.uc.T, io_ramses.lvlc, io_ramses.cpuc]

            if(len(arr) != len(names)):
                io_ramses.close()
                raise ValueError(
                    "Number of fields and size of the hydro array does not match\n"
                    "Consider changing the content of RamsesSnapshot.hydro_names\n"
                    "nvar = %d, recieved: %d" % (len(names), len(arr)))

            if target_fields is not None:
                target_idx = np.where(np.isin(names, target_fields))[0]
                arr = [arr[idx] for idx in target_idx]
                formats = [formats[idx] for idx in target_idx]
                names = [names[idx] for idx in target_idx]

            timer.start('Building table for %d cells... ' % (arr[0].size), 1)
            cell = fromarrays(arr, formats=formats, names=names)
            io_ramses.close()

            bound = compute_boundary(cell['cpu'], cpulist)
            if (self.cell_data is None):
                self.cell_data = cell
            else:
                self.cell_data = np.concatenate([self.cell_data, cell])

            self.bound_cell = np.concatenate([self.bound_cell[:-1], self.bound_cell[-1] + bound])
            self.cpulist_cell = np.concatenate([self.cpulist_cell, cpulist])
            timer.record()

        else:
            if(timer.verbose>=1):
                print('CPU list already satisfied.')

    def read_sink_raw(self, icpu, path_in_repo=''):
        path = join(self.repo, path_in_repo)
        readr.read_sink(path, self.iout, icpu, self.levelmin, self.levelmax)
        arr = [readr.integer_table.T, readr.real_table.T]
        readr.close()
        return arr

    def search_sinkprops(self, path_in_repo='SINKPROPS'):
        sinkprop_regex = re.compile(r'sink_\s*(?P<icoarse>\d+).dat')
        path = join(self.repo, path_in_repo)
        sinkprop_names = glob.glob(join(path, sinkprop_glob))
        if(timer.verbose>=1):
            print('Found %d sinkprop files' % len(sinkprop_names))

        icoarses = []
        for name in sinkprop_names:
            matched = sinkprop_regex.search(name)
            icoarses.append(int(matched.group('icoarse')))
        return np.array(icoarses)

    def read_sinkprop(self, path_in_repo='SINKPROPS', icoarse=None, drag_part=True, raw_data=False, return_aexp=False):
        """Reads single sinkprop file from given coarse step number,
        if icoarse not specified, reads the step number of current snapshot
        if file is not found, tries to search for sinkprop file with nearest step number
        """
        if(icoarse is None):
            icoarses = self.search_sinkprops(path_in_repo)
            icoarse = icoarses[np.argmin(np.abs((self.nstep_coarse) - icoarses))]
            if(icoarse != self.nstep_coarse):
                warnings.warn('Targeted SINKPROP file not found with icoarse = %d\nFile with icoarse = %d is loaded instead.' % (self.nstep_coarse, icoarse))
        path = join(self.repo, path_in_repo)
        check = join(path, sinkprop_format.format(icoarse=icoarse))
        if(not exists(check)):
            raise FileNotFoundError('Sinkprop file not found: %s' % check)
        readr.read_sinkprop(path, icoarse, drag_part, self.mode)
        arrs = [readr.integer_table.T, readr.real_table.T]

        timer.start('Building table for %d smbhs... ' % arrs[0].shape[0], 1)
        if(raw_data):
            return arrs
        if(drag_part):
            dtype = sink_prop_dtype_drag
        else:
            dtype = sink_prop_dtype
        if(self.mode == 'fornax'):
            dtype = sink_prop_dtype_drag_fornax
        if(self.mode == 'y2' or self.mode == 'y3' or self.mode == 'y4' or self.mode == 'nc'):
            dtype = sink_prop_dtype_drag_y2
        if(arrs[0].shape[1] + arrs[1].shape[1] != len(dtype)):
            readr.close()
            raise ValueError('Number of fields mismatch\n'
                             'Received: %d, Allocated: %d' % (arrs[0].shape[1] + arrs[1].shape[1], len(dtype)))
        sink = fromndarrays(arrs, dtype=dtype)
        timer.record()
        aexp = np.copy(readr.aexp)
        readr.close()

        if(return_aexp):
            return sink, aexp
        else:
            return sink

    def read_sinkprops(self, path_in_repo='SINKPROPS', drag_part=True, use_cache=False, cache_name='sinkprops.pkl',
                       reset_cache=False, cache_format='pkl', progress=False):
        """Searches and reads all files in the sinkprops directory and returns as single table.
        if use_cache=True, it saves pickle file to save reading time
        """
        icoarses = self.search_sinkprops(path_in_repo)
        path = join(self.repo, path_in_repo)

        cache = None
        if(use_cache and not reset_cache):
            cache_file =join(path, cache_name)
            if(exists(cache_file)):
                cache = utool.load(cache_file, format=cache_format)
                icoarses = icoarses[~np.isin(icoarses, np.unique(cache['icoarse']))]
                if(icoarses.size == 0):
                    print('Found cached file: %s' % cache_file)
                    return cache

        int_table = []
        real_table = []
        nsinks = []
        aexps = []

        timer.start('Reading files...')
        if(progress):
            iterator = tqdm(icoarses)
        else:
            iterator = icoarses
        for icoarse in iterator:
            readr.read_sinkprop(path, icoarse, drag_part, self.mode)
            nsink = readr.integer_table.shape[1]
            nsinks.append(nsink)

            int_table.append(np.copy(readr.integer_table.T))
            real_table.append(np.copy(readr.real_table.T))
            aexps.append(np.copy(readr.aexp))

        int_table = np.concatenate(int_table)
        real_table = np.concatenate(real_table)

        aexp_table = np.repeat(aexps, nsinks)
        icoarse_table = np.repeat(icoarses, nsinks)

        arrs = [int_table, real_table]

        if(drag_part):
            dtype = sink_prop_dtype_drag
        else:
            dtype = sink_prop_dtype
        timer.record()

        if(self.mode == 'nh'):
            dtype = sink_prop_dtype
        if(self.mode == 'fornax'):
            dtype = sink_prop_dtype_drag_fornax
        if(self.mode == 'y2' or self.mode == 'y3' or self.mode == 'y4' or self.mode == 'nc'):
            dtype = sink_prop_dtype_drag_y2
        if(arrs[0].shape[1] + arrs[1].shape[1] != len(dtype)):
            readr.close()
            raise ValueError('Number of fields mismatch\n'
                             'Received: %d, Allocated: %d' % (arrs[0].shape[1] + arrs[1].shape[1], len(dtype)))

        timer.start('Building table for %d smbhs...' % arrs[0].shape[0])
        sink = fromndarrays(arrs, dtype=dtype)
        sink = append_fields(sink, ['aexp', 'icoarse'], [aexp_table, icoarse_table], usemask=False)

        timer.record()
        readr.close()
        if cache is not None:
            sink = np.concatenate([cache, sink])

        if(reset_cache or use_cache):
            cache_file = join(path, cache_name)
            utool.dump(sink, cache_file, format=cache_format)

        return sink


    def clear(self, part=True, cell=True):
        """Clear exsisting cache from snapshot data.

        Parameters
        ----------
        part : bool
            if True, clear particle cache
        cell : bool
            if True, clear amr cache

        Returns
        -------

        """
        self.box = None
        self.pcmap = None
        if(part):
            self.part_data = None
            self.part = None
            self.box_part = None
            self.cpulist_part = np.array([], dtype='i4')
            self.bound_part = np.array([0], dtype='i4')
        if(cell):
            self.cell_data = None
            self.cell = None
            self.box_cell = None
            self.cpulist_cell = np.array([], dtype='i4')
            self.bound_cell = np.array([0], dtype='i4')

    def _read_nstar(self):
        part_file = FortranFile(self.get_path('part', 1))
        part_file.skip_records(4)
        if(not self.longint):
            return part_file.read_ints()
        else:
            return part_file.read_longs()

    def read_hydro_ng(self):
        # not used

        hydro_uolds = []
        for icpu in np.arange(1, self.params['ncpu'] + 1):
            path = self.get_path(type='hydro', icpu=icpu)
            opened = open(path, mode='rb')

            header = np.fromfile(opened, dtype=np.int32, count=4)
            ndim, nvar, levelmin, nlevelmax = header # nothing to do

            nocts = np.fromfile(opened, dtype=np.int32, count=nlevelmax - levelmin + 1)
            hydro_uold = np.fromfile(opened, dtype=np.float64)

            hydro_uolds.append(hydro_uold)

            opened.close()

        hydro_uolds = np.concatenate(hydro_uolds)
        hydro_uolds = np.reshape(hydro_uolds, [-1, nvar, 2 ** ndim])
        hydro_uolds = np.swapaxes(hydro_uolds, 1, 2)

        self.params['nvar'] = nvar
        self.var = hydro_uolds

    def read_amr_ng(self):
        amr_poss = []
        amr_lvls = []
        amr_refs = []
        amr_cpus = []

        for icpu in np.arange(1, self.params['ncpu'] + 1):
            amr_path = self.get_path(type='amr', icpu=icpu)
            opened = open(amr_path, mode='rb')

            header = np.fromfile(opened, dtype=np.int32, count=3)
            ndim, levelmin, nlevelmax = header

            nocts = np.fromfile(opened, dtype=np.int32, count=nlevelmax - levelmin + 1)
            amr_size = np.sum(nocts)

            block_size = ndim + 2 ** ndim
            amr = np.fromfile(opened, dtype=np.int32)
            amr = np.reshape(amr, [-1, block_size])

            poss, lvls = ckey2idx(amr[:, :ndim], nocts, levelmin)
            amr_poss.append(poss)
            amr_lvls.append(lvls)
            amr_refs.append(amr[:, ndim:].astype(np.bool))
            amr_cpus.append(np.full(amr_size, icpu))

            opened.close()

        self.x = np.swapaxes(np.concatenate(amr_poss), 1, 2)
        self.lvl = np.concatenate(amr_lvls)
        self.ref = np.concatenate(amr_refs)
        self.cpu = np.concatenate(amr_cpus)

    def get_cell(self, box=None, target_fields=None, domain_slicing=True, exact_box=True, cpulist=None, ripses=False):
        if(box is not None):
            # if box is not specified, use self.box by default
            self.box = box
        if(cpulist is None):
            if(self.box is None or np.array_equal(self.box, default_box)):
                # box is default box or None: load the whole volume
                domain_slicing = False
                exact_box = False
        else:
            # if cpulist is set,
            if(not domain_slicing):
                warnings.warn("cpulist cannot be set without domain_slicing!", UserWarning)
                domain_slicing = True
            exact_box = False

        if(self.box is None or not np.array_equal(self.box, self.box_cell) or cpulist is not None):
            if(cpulist is None):
                cpulist = self.get_involved_cpu()
            else:
                domain_slicing=False
                exact_box=False
            if(not ripses):
                self.read_cell(target_fields=target_fields, cpulist=cpulist)
            else:
                self.read_ripses(target_fields=target_fields, cpulist=cpulist)
            if(domain_slicing):
                cell = domain_slice(self.cell_data, cpulist, self.cpulist_cell, self.bound_cell)
            else:
                cell = self.cell_data

            if(exact_box):
                mask = box_mask(get_vector(cell), self.box, size=self.cell_extra['dx'](cell))
                timer.start('Masking cells... %d / %d (%.4f)' % (np.sum(mask), mask.size, np.sum(mask)/mask.size), 1)
                cell = cell[mask]
                timer.record()

            cell = self.Cell(cell, self)
            self.box_cell = self.box
            self.cell = cell
        return self.cell

    class Cell(Table):
        def __getitem__(self, item):
            cell_extra = self.snap.cell_extra
            if isinstance(item, str):
                if item in cell_extra.keys():
                    return cell_extra[item](self)
                else:
                    return self.table[item]
            elif isinstance(item, tuple):
                letter, unit = item
                return self.__getitem__(letter) / self.snap.unit[unit]
            else:
                return RamsesSnapshot.Cell(self.table[item], self.snap)


    def get_part(self, box=None, target_fields=None, domain_slicing=True, exact_box=True, cpulist=None, pname=None):
        if(box is not None):
            # if box is not specified, use self.box by default
            self.box = box
        if(cpulist is None):
            if(self.box is None or np.array_equal(self.box, default_box)):
                # box is default box or None: load the whole volume
                domain_slicing = False
                exact_box = False
        else:
            # if cpulist is set,
            if(not domain_slicing):
                warnings.warn("cpulist cannot be set without domain_slicing!", UserWarning)
                domain_slicing = True
            exact_box = False
        do=False
        if self.part is not None:
            if pname != self.part.ptype:
                print(f"\nYou loaded part only `{self.part.ptype}` but now you want `{pname}`!\nIt forces to remove `{self.part.ptype}` data and retry get_part (so it's inefficient!)\n")
                self.part_data=None
                self.part=None
                self.box_part = None
                self.cpulist_part = np.array([], dtype='i4')
                self.bound_part = np.array([0], dtype='i4')
                do=True
        if(self.box is None or not np.array_equal(self.box, self.box_part) or cpulist is not None or do):
            if(cpulist is None):
                cpulist = self.get_involved_cpu()
            else:
                domain_slicing = True
                exact_box = False
            self.read_part(target_fields=target_fields, cpulist=cpulist, pname=pname)
            if(domain_slicing):
                part = domain_slice(self.part_data, cpulist, self.cpulist_part, self.bound_part)
            else:
                part = self.part_data
            if(self.box is not None):
                if(exact_box):
                    mask = box_mask(get_vector(part), self.box)
                    timer.start('Masking particles... %d / %d (%.4f)' % (np.sum(mask), mask.size, np.sum(mask)/mask.size), 1)
                    part = part[mask]
                    timer.record()
            part = self.Particle(part, self, ptype=pname)
            self.box_part = self.box
            self.part = part
        return self.part

    class Particle(Table):
        def __getitem__(self, item):
            part_extra = self.snap.part_extra
            if isinstance(item, str):
                if item in part_extra.keys():
                    return part_extra[item](self)
                elif item in part_family.keys():
                    if self.ptype is not None:
                        if item == self.ptype:
                            return self
                        else:
                            print(f"\nYou loaded part only `{self.ptype}` but now you want `{item}`!\nIt forces to clear `{self.ptype}` data and retry get_part (so it's inefficient!)\n")
                            self.snap.part_data = None
                            self.snap.part = None
                            self.snap.box_part = None
                            cpulist = np.unique(self.snap.cpulist_part) if(self.snap.box is None or np.array_equal(self.snap.box, default_box)) else None
                            self.snap.cpulist_part = np.array([], dtype='i4')
                            self.snap.bound_part = np.array([0], dtype='i4')
                            part = self.snap.get_part(box=self.snap.box, target_fields=self.table.dtype.names, domain_slicing=True, exact_box=True, cpulist=cpulist, pname=item)
                            return part
                    else:
                        return RamsesSnapshot.Particle(classify_part(self.table, item, ptype=self.ptype), self.snap, ptype=item)
                elif item == 'smbh':
                    return RamsesSnapshot.Particle(find_smbh(self.table), self.snap, ptype='smbh')
                else:
                    return self.table[item]
            elif isinstance(item, tuple):
                letter, unit = item
                return self.__getitem__(letter) / self.snap.unit[unit]
            else:
                return RamsesSnapshot.Particle(self.table[item], self.snap, ptype=self.ptype)


    def get_halos_cpulist(self, halos, radius=3., use_halo_radius=True, radius_name='rvir', n_divide=4):
        # returns cpulist that encloses given list of halos
        cpulist = []
        for halo in halos:
            if(use_halo_radius):
                extent = halo[radius_name]*radius*2
            else:
                extent = radius*2
            box = get_box(get_vector(halo), extent)
            cpulist.append(get_cpulist(box, None, self.levelmax, self.bound_key, self.ndim, n_divide))
        return np.unique(np.concatenate(cpulist))

    def get_cpulist_from_part(self, ids, path_in_repo='part_cpumap', mode='init', filename='%s_cpumap_%05d.pkl'):
        """
        reads particle-cpumap file (if there's any) and returns appropriate cpulist of domains
        that encompass selected id list of paritcles
        mode can either be 'init'(dm + tracer) or 'star'
        """
        if(self.pcmap is None):
            path = join(self.repo, path_in_repo, filename % (mode, self.iout))
            self.pcmap = utool.load(path)
        return np.unique(self.pcmap[ids]).astype('i8')

    def diag(self):
        # prints a brief description of the current status of snapshot.
        dm_tot = 0
        star_tot = 0
        gas_tot = 0
        smbh_tot = 0
        if(self.box is not None):
            volume = np.prod(self.box[:, 1]-self.box[:, 0]) / (self.unit['Mpc']/self.params['h'])**3 / self.params['aexp']**3
            print('=============================================')
            print('Description of the snapshot %05d (%s)' % (self.iout, self.repo))
            print('---------------------------------------------')
            print('Redshift (z) = %.5f (a = %.5f), Age of the Universe = %.4f Gyr' % (self.z, self.aexp, self.params['age']))
            print('Comoving volume of the box: %.3e (Mpc/h)^3' % (volume))
        if(self.part is not None):
            part = self.part
            part = part[box_mask(get_vector(part), self.box)]
            print('---------------------------------------------')
            print('Total  number of particles: %d' % part.size)
            dm = part['dm']

            dm_tot = np.sum(dm['m', 'Msol'])
            dm_min = np.min(dm['m', 'Msol'])

            print('Number of     DM particles: %d with total mass of %.3e Msol, Min. particle mass: %.3e Msol' % (dm.size, dm_tot, dm_min))

            contam = np.sum(dm[dm['m'] > np.min(dm['m'])]['m'] * 1.01) / np.sum(dm['m'])
            if(contam>0.0):
                print('DM Contamination fraction within the box: %.3f %%' % (contam*100))

            tracer = part['tracer']

            if(tracer.size>0):
                tracer_tot = np.sum(tracer['m', 'Msol'])
                tracer_min = np.min(tracer['m', 'Msol'])

                print('Number of tracer particles: %d with total mass of %.3e Msol, Min. particle mass: %.3e Msol' % (tracer.size, tracer_tot, tracer_min))

            if(self.params['star']):
                star = part['star']
                smbh = part['smbh']

                star_tot = np.sum(star['m', 'Msol'])
                star_min = np.min(star['m', 'Msol'])

                print('---------------------------------------------')

                print('Number of       star particles: %d with total mass of %.3e Msol, Min. particle mass: %.3e Msol' % (star.size, star_tot, star_min))
                if(star.size>0):
                    sfr100 = np.sum(star[star['age', 'Myr']<100]['m', 'Msol'])/1E8
                    sfr10 = np.sum(star[star['age', 'Myr']<10]['m', 'Msol'])/1E7
                    sfr1 = np.sum(star[star['age', 'Myr']<1]['m', 'Msol'])/1E6
                    print('SFR within the box (last 100, 10, 1Myr): %.3e, %.3e %.3e Msol/yr' % (sfr100, sfr10, sfr1))

                if(smbh.size>0):
                    smbh_tot = np.sum(smbh['m', 'Msol'])
                    smbh_max = np.max(smbh['m', 'Msol'])

                    print('Number of       SMBH particles: %d with total mass of %.3e Msol, Max. SMBH mass: %.3e Msol' % (smbh.size, smbh_tot, smbh_max))
                print('DM/Stellar mass ratio is %.3f' % (dm_tot / star_tot))

                star_den = star_tot/volume
                print('Stellar Mass density is %.3e Msol / (Mpc/h)^3' % (star_den))

        if(self.cell is not None):
            cell = self.cell
            cell = cell[box_mask(get_vector(cell), self.box, size=cell['dx'])]
            print('---------------------------------------------')
            print('Min. spatial resolution = %.4f pc (%.4f pc/h in comoving)' % (np.min(self.cell['dx', 'pc']), self.boxsize*1E6*0.5**np.max(self.cell['level'])))
            print('Total number of cells: %d' % cell.size)
            gas_tot = np.sum(cell['rho'] * (cell['dx'])**3) / self.unit['Msol']
            print('Total gas mass: %.3e Msol' % gas_tot)
            print('Max. gas density    : %.3e H/cc' % np.max(self.cell['rho', 'H/cc']))
            print('Max. gas temperature: %.3e K' % np.max(self.cell['T', 'K']))
            if('refmask' in cell.dtype.names):
                contam = 1.-np.sum(cell[cell['refmask']>0.01]['m'])/np.sum(cell['m'])
                if(contam>0.):
                    print('Cell Contamination fraction within the box: %.3f %%' % (contam*100))

        if(self.cell is not None and self.part is not None):
            print('Baryonic fraction: %.3f' % ((gas_tot+star_tot+smbh_tot) / (dm_tot+gas_tot+star_tot+smbh_tot)))

def trace_parts(part_ini, cropped):
    return part_ini[np.isin(part_ini['id'], cropped['id'], True)]

def write_zoomparts_music(part_ini: RamsesSnapshot.Particle, cropped: RamsesSnapshot.Particle,
                          filepath: str, reduce: int=None):
    """
    writes position table of particles in MUSIC format.
    """
    cropped_ini = part_ini[np.isin(part_ini['id'], cropped['id'], True)]
    if reduce is not None:
        cropped_ini = np.random.choice(cropped_ini, cropped_ini.size//reduce, replace=False)
    np.savetxt(filepath, get_vector(cropped_ini))
    return cropped_ini

def write_parts_rockstar(part: RamsesSnapshot.Particle, snap: RamsesSnapshot, filepath: str):
    """
    writes particle data in ASCII format that can be read by Rockstar
    """
    timer.start('Writing %d particles in %s... ' % (part.size, filepath), 1)

    pos = get_vector(part) * snap.params['boxsize']
    vel = get_vector(part, 'v') * snap.get_unit('v', 'km/s')
    table = fromarrays([*pos.T, *vel.T, part['id']], formats=['f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'i4'])
    np.savetxt(filepath, table, fmt=('%.16e',)*6 + ('%d',))

    timer.record()

def write_snaps_rockstar(repo: str, start: int, end: int, mode='none',
                         path_in_repo='snapshots', ncpu=48, min_halo_particles=100):
    # write particles in format that can be read by Rockstar
    path = join(repo, 'rst')
    dm_flist = []
    star_flist = []
    for iout in np.arange(start, end):
        snap = RamsesSnapshot(repo, iout, mode, path_in_repo=path_in_repo)
        part = snap.get_part()

        filepath = join(path, 'dm_%05d.dat' % iout)
        write_parts_rockstar(part['dm'], snap, filepath)
        dm_flist.append(filepath)

        if(snap.params['star']):
            filepath = join(path, 'star_%05d.dat' % iout)
            write_parts_rockstar(part['star'], snap, filepath)
            star_flist.append(filepath)

    with open(join(path, 'dmlist.dat'), 'w') as opened:
        for fname in dm_flist:
            opened.write(fname+'\n')

    with open(join(path, 'dm.cfg')) as opened:
        opened.write('SNAPSHOT_NAMES = %s\n' % join(path, 'dmlist.dat'))
        opened.write('NUM_WRITERS = %d\n' % ncpu)
        opened.write('FILE_FORMAT = ASCII\n')

        opened.write('BOX_SIZE = %.3f\n' % snap.params['boxsize'])
        opened.write('PARTICLE_MASS = %.3f\n' % np.min(snap.part['m']) * snap.get_unit('m', 'Msun') * snap.params['h'])
        opened.write('h0 = %.4f\n' % snap.params['h'])
        opened.write('Ol = %.4f\n' % snap.params['omega_l'])
        opened.write('Om = %.4f\n' % snap.params['omega_m'])

        opened.write('MIN_HALO_PARTICLES = %d\n' % min_halo_particles)

    if (snap.params['star']):

        with open(join(path, 'starlist.dat'), 'w') as opened:
            for fname in star_flist:
                opened.write(fname+'\n')

        with open(join(path, 'star.cfg')) as opened:
            opened.write('SNAPSHOT_NAMES = %s\n' % join(path, 'starlist.dat'))
            opened.write('NUM_WRITERS = %d\n' % ncpu)
            opened.write('FILE_FORMAT = ASCII\n')

            opened.write('BOX_SIZE = %.3f\n' % snap.params['boxsize'])
            opened.write('PARTICLE_MASS = %.3f\n' % np.min(snap.part['m']) * snap.get_unit('m', 'Msun') * snap.params['h'])
            opened.write('h0 = %.4f\n' % snap.params['h'])
            opened.write('Ol = %.4f\n' % snap.params['omega_l'])
            opened.write('Om = %.4f\n' % snap.params['omega_m'])

            opened.write('MIN_HALO_PARTICLES = %d\n' % min_halo_particles)

def save_part_cpumap(snap, full_box=False, icpu_dtype='u2', path_in_repo='part_cpumap', mode='init', filename='%s_cpumap_%05d.pkl'):
    # writes cpumap that tells what particle belongs to which cpu
    if(full_box):
        snap.box = None
    snap.get_part()
    if(mode == 'init'): # dm and tracer
        part = snap.part['init']
    elif(mode == 'star'):
        part = snap.part['star']
    else:
        raise ValueError("Unknown mode: %s" % mode)
    if(part.size > 0):
        size = np.max(part['id']) + 1
        pcmap = np.zeros(size, dtype=icpu_dtype)
        pcmap[part['id']] = part['cpu']
        path = join(snap.repo, path_in_repo, filename % (mode, snap.iout))
        utool.dump(pcmap, path)
    else:
        print("No particle detected, skipping..")

def cut_spherical(table, center, radius, prefix='', ndim=3, inverse=False):
    distances = rss(center - get_vector(table, prefix, ndim))
    if(inverse):
        mask = distances > radius
    else:
        mask = distances <= radius
    return table[mask]

def cut_halo(table, halo, radius=1, use_halo_radius=True, inverse=False, radius_name='rvir'):
    center = get_vector(halo)
    if(use_halo_radius):
        radius = halo[radius_name] * radius
    else:
        radius = radius
    return cut_spherical(table, center, radius, inverse=inverse)

def classify_part(part, pname, ptype=None):
    # classify particles, if familty exists in the data, use it.
    # if not, use id, mass and epoch instead.
    timer.start('Classifying %d particles... ' % part.size, 2)
    if (ptype is not None):
        if isinstance(ptype, str):
            if(pname == ptype):
                return part
            else:
                return np.array([], dtype=part.dtype)
        else:
            raise TypeError(f"Invalid type of given `ptype`({type(ptype)}) instead of `list` or `str`!")

    names = part.dtype.names
    if('family' in names):
        # Do a family-based classification
        mask = np.isin(part['family'], part_family[pname])

    elif('epoch' in names):
        # Do a parameter-based classification
        if(pname == 'dm'):
            mask = (part['epoch'] == 0) & (part['id'] > 0)
        elif(pname == 'star'):
            mask = ((part['epoch'] < 0) & (part['id'] > 0))\
                   | ((part['epoch'] != 0) & (part['id'] < 0))
        elif(pname == 'sink' or pname == 'cloud'):
            mask = (part['id'] < 0) & (part['m'] > 0) & (part['epoch'] == 0)
        elif(pname == 'tracer'):
            mask = (part['id'] < 0) & (part['m'] == 0)
        else:
            mask = False
    elif('id' in names):
        # DM-only simulation
        if(pname == 'dm'):
            mask =  part['id'] > 0
        elif(pname == 'tracer'):
            mask = (part['id'] < 0) & (part['m'] == 0)
        else:
            mask = False
    else:
        # No particle classification is possible
        raise ValueError('Particle data structure not classifiable.')
    output = part[mask]

    timer.record()
    return output

def find_smbh(part, verbose=None):
    # Find SMBHs by merging sink (cloud) particles
    verbose_tmp = timer.verbose
    if(verbose is not None):
        timer.verbose = verbose
    timer.start('Searching for SMBHs in %d particles...' % part.size, 2)
    sink = classify_part(part, 'cloud')
    names = part.dtype.names
    ids = np.unique(sink['id'])

    smbh = []
    for id in ids:
        smbh_cloud = sink[sink['id'] == id]
        row = ()
        for name in names:
            if (name == 'm'):
                val = np.sum(smbh_cloud[name])
            elif (name == 'family'):
                val = 4
            elif (name == 'tag'):
                val = 0
            else:
                val = np.average(smbh_cloud[name])
            row += (val,)
        smbh.append(row)
    smbh = np.array(smbh, dtype=part.dtype)
    timer.record()
    if(timer.verbose >=2):
        print('Found %d SMBHs.' % smbh.size)
    timer.verbose = verbose_tmp
    return smbh

def box_mask(coo, box, size=None, exclusive=False):
    # masking coordinates based on the box
    if size is not None:
        size = expand_shape(size, [0], 2)
    else:
        size = 0
    if(exclusive):
        size *= -1
    box = np.array(box)
    box_mask = np.all((box[:, 0] <= coo+size/2) & (coo-size/2 <= box[:, 1]), axis=-1)
    return box_mask

def interpolate_part(part1, part2, name, fraction=0.5, periodic=False):
    # Interpolates two particle snapshots based on their position and fraction
    timer.start('Interpolating %d, %d particles...' % (part1.size, part2.size), 2)

    id1 = part1['id']
    id2 = part2['id']

    id1 = np.abs(id1)
    id2 = np.abs(id2)

    part_size = np.maximum(np.max(id1), np.max(id2))+1

    val1 = part1[name]
    val2 = part2[name]

    if(name == 'pos' or name == 'vel'):
        pool = np.zeros((part_size, 3), dtype='f8')
    else:
        pool = np.zeros(part_size, dtype=val1.dtype)

    mask1 = np.zeros(part_size, dtype='?')
    mask2 = np.zeros(part_size, dtype='?')

    mask1[id1] = True
    mask2[id2] = True

    active_mask = mask1 & mask2
    if(periodic):
        diff = np.zeros((part_size, 3), dtype='f8')
        diff[id1] += val1
        diff[id2] -= val2

        # if val1-val2 >> 0, the particle moved outside the boundary 1
        # if val1-val2 << 0, the particle moved outside the boundary 0
        val2 = val2.copy()

        val2[diff[id2] > 0.5] += 1.
        val2[diff[id2] < -0.5] -= 1.

    pool[id1] += val1 * (1. - fraction)
    pool[id2] += val2 * fraction
    val = pool[active_mask]

    if(periodic):
        val = np.mod(val, 1.)

    if(timer.verbose>=2):
        print("Particle interpolation - part1[%d], part2[%d], result[%d]" % (id1.size, id2.size, np.sum(active_mask)))
    timer.record()

    return val

def interpolate_part_pos(part1, part2, Gyr_interp, fraction=0.5):
    # Interpolates two particle snapshots based on their position and fraction
    timer.start('Interpolating %d, %d particles...' % (part1.size, part2.size), 2)

    id1 = part1['id']
    id2 = part2['id']

    id1 = np.abs(id1)
    id2 = np.abs(id2)

    part_size = np.maximum(np.max(id1), np.max(id2))+1

    pos1 = part1['pos']
    pos2 = part2['pos']

    vel1 = part1['vel']
    vel2 = part2['vel']

    pool = np.zeros((part_size, 3), dtype='f8')

    mask1 = np.zeros(part_size, dtype='?')
    mask2 = np.zeros(part_size, dtype='?')

    mask1[id1] = True
    mask2[id2] = True

    active_mask = mask1 & mask2

    time_interval = (part2.snap.age - part1.snap.age) * Gyr_interp

    pool[id1] += interp_term(pos1, vel1, fraction, time_interval, 1)
    pool[id2] += interp_term(pos2, vel2, 1-fraction, time_interval, -1)
    val = pool[active_mask]

    if(timer.verbose>=2):
        print("Particle interpolation - part1[%d], part2[%d], result[%d]" % (id1.size, id2.size, np.sum(active_mask)))
    timer.record()

    return val

def interp_term(pos, vel, fraction, time_interval, vel_sign=1):
    fun = lambda x: -np.cos(x*np.pi)/2 + 0.5 # arbitrary blending function I just invented...
    return (pos + time_interval * fraction * vel * vel_sign) * fun(1-fraction)

def sync_tracer(tracer, cell, copy=False, **kwargs):
    tid, cid = match_tracer(tracer, cell, **kwargs)
    tracer[tid] = utool.set_vector(tracer, cell[cid]['vel'], prefix='v', copy=copy)
    if(copy):
        return tracer

def match_part_to_cell(part, cell, n_search=16):
    tree = KDTree(cell['pos'])
    dists, idx_cell = tree.query(part['pos'], k=n_search, n_jobs=-1, p=np.inf)

    star_pos = utool.expand_shape(part['pos'], [0, 2], 3)
    dists_cand = np.max(np.abs(cell[idx_cell]['pos'] - star_pos), axis=-1) / cell[idx_cell]['dx']

    min_idxs = np.argmin(dists_cand, axis=-1)
    min_dists = np.min(dists_cand, axis=-1)
    if(np.any(min_dists>0.5)):
        print(min_dists)
        raise RuntimeWarning("%d particles are not matched corretly. Try increasing n_search. If it doesn't work, it could mean your cell data is incomplete." % np.sum(min_dists>0.5))

    idx_cell = idx_cell[(np.arange(part.size), min_idxs)]

    return idx_cell

def match_tracer(tracer, cell, n_jobs=-1, min_dist_pc=1, use_cell_size=False):
    # match MC gas tracer particles to cell
    timer.start("Matching %d tracers and %d cells..." % (tracer.size, cell.size), 1)
    tree = KDTree(tracer['pos'])
    dists, idx_tracer = tree.query(cell['pos'], p=1, n_jobs=n_jobs)

    if(use_cell_size):
        mask = dists < min_dist_pc*cell.snap.unit['pc']
    else:
        mask = dists < cell['dx']

    idx_cell = np.arange(cell.size)
    idx_cell = idx_cell[mask]
    idx_tracer = idx_tracer[mask]

    print("%d / %d tracers are matched to %d / %d cells"
          % (np.unique(idx_tracer).size, tracer.size, np.unique(idx_cell).size, cell.size))
    timer.record()
    return idx_tracer, idx_cell

def time_series(repo, iouts, halo_table, mode='none', extent=None, unit=None):
    # returns multiple snapshots from repository and array of iouts
    snaps = []
    snap = None
    for halo, iout in zip(halo_table, iouts):
        snap = RamsesSnapshot(repo, iout, mode, snap=snap)
        if(extent is None):
            extent_now = halo['rvir']*2
        else:
            extent_now = extent * snap.unit[unit]
        box = get_box(get_vector(halo), extent_now)
        snap.box = box
        snaps.append(snap)
    return snaps

def get_cpulist(box, binlvl, maxlvl, bound_key, ndim, n_divide):
    # get list of cpus involved in selected box.
    volume = np.prod([box[:, 1] - box[:, 0]])
    if (binlvl is None):
        binlvl = int(np.log2(1. / (volume + 1E-20)) / ndim) + n_divide
    if (binlvl > 64 // ndim):
        binlvl = 64 // ndim - 1
    lower, upper = np.floor(box[:, 0] * 2 ** binlvl).astype(int), np.ceil(box[:, 1] * 2 ** binlvl).astype(int)
    bbox = np.stack([lower, upper], axis=-1)

    bin_list = utool.cartesian(
        np.arange(bbox[0, 0], bbox[0, 1]),
        np.arange(bbox[1, 0], bbox[1, 1]),
        np.arange(bbox[2, 0], bbox[2, 1]))  # TODO: generalize this

    if (timer.verbose >= 2):
        print("Setting bin level as %d..." % binlvl)
        print("Input box:", box)
        print("Bounding box:", bbox)
        ratio = np.prod([bbox[:, 1] / 2 ** binlvl - bbox[:, 0] / 2 ** binlvl]) / volume
        print("Volume ratio:", ratio)
        print("N. of Blocks:", bin_list.shape[0])

    keys = hilbert3d(*(bin_list.T), binlvl, bin_list.shape[0])
    keys = np.array(keys)
    key_range = np.stack([keys, keys + 1], axis=-1)
    key_range = key_range.astype('f8')

    involved_cpu = []
    for icpu_range, key in zip(
            np.searchsorted(bound_key / 2. ** (ndim * (maxlvl - binlvl + 1)), key_range),
            key_range):
        involved_cpu.append(np.arange(icpu_range[0], icpu_range[1] + 1))
    involved_cpu = np.unique(np.concatenate(involved_cpu)) + 1
    if (timer.verbose >= 2):
        print("List of involved CPUs: ", involved_cpu)
    return involved_cpu


def ckey2idx(amr_keys, nocts, levelmin, ndim=3):
    idx = 0
    poss = []
    lvls = []
    for noct, leveladd in zip(nocts, np.arange(0, nocts.size)):
        ckey = amr_keys[idx : idx+noct]
        idx += noct
        ckey = np.repeat(ckey[:,:,np.newaxis], 2**ndim, axis=-1)
        suboct_ind = np.arange(2**ndim)
        nstride = 2**np.arange(0, ndim)

        suboct_ind, nstride = np.meshgrid(suboct_ind, nstride)

        cart_key = 2*ckey+np.mod(suboct_ind//nstride, 2) + 0.5
        level = levelmin+leveladd
        poss.append(cart_key/2**level)
        lvls.append(np.full(noct, level))
    poss = np.concatenate(poss)
    #poss = np.mod(poss-0.5, 1)
    lvls = np.concatenate(lvls)
    return poss, lvls

def domain_slice(array, cpulist, cpulist_all, bound):
    # array should already been aligned with bound
    idxs = np.where(np.isin(cpulist_all, cpulist))[0]
    doms = np.stack([bound[idxs], bound[idxs+1]], axis=-1)
    segs = doms[:, 1] - doms[:, 0]

    out = np.empty(np.sum(segs), dtype=array.dtype) # same performance with np.concatenate
    now = 0
    for dom, seg in zip(doms, segs):
        out[now:now + seg] = array[dom[0]:dom[1]]
        now += seg

    return out


def bulk_sort(array):
    # Sorts the array cpu-wise, not used for now
    cpumap = array['cpu']
    idxs = compute_boundary(cpumap)
    counts = np.diff(idxs)
    idxs = idxs[:-1]
    cpulist = cpumap[idxs]

    key = np.argsort(cpulist)
    cpulist = cpulist[key]
    idxs = idxs[key]
    counts = counts[key]

    new = np.empty(array.size, dtype=array.dtype)
    now = 0
    bound_new = [0]
    for icpu, idx, count in zip(cpulist, idxs, counts):
        new[now:now+count] = array[idx:idx+count]
        now += count
        bound_new.append(now)

    return new, np.array(bound_new)

def compute_boundary(cpumap, cpulist):
    bound = np.searchsorted(cpumap, cpulist)
    return np.concatenate([bound, [cpumap.size]])

class GraficLevel(object):
    # an object to read grafic ic file of specific level
    def __init__(self, level_repo, level=None, read_pos=True):
        self.repo = level_repo
        self.read_pos = read_pos
        if(level is None):
            self.level = int(level_repo[-3:])
        else:
            self.level = level
        self.read_header('ic_deltab')
        self.set_coo()

    def read_ic(self):
        vel = []
        if(self.read_pos):
            pos = []
        else:
            self.pos = None
        self.rho = self.read_file(join(self.repo, 'ic_deltab'))
        if(exists(join(self.repo, 'ic_refmap'))):
            self.ref = self.read_file(join(self.repo, 'ic_refmap'))
        else:
            self.ref = None
        for dim in ['x', 'y', 'z']:
            vel_dim = self.read_file(join(self.repo, 'ic_velc%s' % dim))
            vel.append(vel_dim)
            if(self.read_pos):
                pos_dim = self.read_file(join(self.repo, 'ic_posc%s' % dim))
                pos.append(pos_dim)

        self.pvar = []
        for idx in self.pvar_idxs:
            self.pvar.append(self.read_file(join(self.repo, 'ic_pvar_%05d' % idx)))

        self.vel = np.stack(vel, axis=-1)
        if(self.read_pos):
            self.pos = np.stack(pos, axis=-1)

    def get_table(self):
        table_dtype = [('coo', 'f4', 3), ('vel', 'f4', 3), ('pos', 'f4', 3), ('rho', 'f4'), ('ref', 'f4')]
        table_dtype += [('pvar%03d' % idx, 'f4') for idx in self.pvar_idxs]
        table = np.zeros(self.rho.size, dtype=table_dtype)
        coo = []
        vel = []
        if(self.read_pos):
            pos = []
        for idim in [0, 1, 2]:
            coo.append(self.coo[:, :, :, idim].flatten())
            vel.append(self.vel[:, :, :, idim].flatten())
            if(self.read_pos):
                pos.append(self.pos[:, :, :, idim].flatten())
        coo = np.stack(coo, axis=-1)
        table['coo'] = coo

        vel = np.stack(vel, axis=-1)
        table['vel'] = vel

        if(self.read_pos):
            pos = np.stack(pos, axis=-1)
            table['pos'] = pos

        table['rho'] = self.rho.flatten()
        if(self.ref is not None):
            table['ref'] = self.ref.flatten()
        for (idx, pvar_idx) in zip(np.arange(len(self.pvar)), self.pvar_idxs):
            table['pvar%03d' % pvar_idx] = self.pvar[idx].flatten()
        return table

    def read_header(self, fname):
        ff = FortranFile(join(self.repo, fname))
        self.header = ff.read_record(grafic_header_dtype)

        pvar_fnames = glob.glob(join(self.repo, 'ic_pvar_'+'[0-9]'*5))
        self.pvar_idxs = [int(pvar_fname[-5:]) for pvar_fname in pvar_fnames]

    def set_coo(self):
        nx, ny, nz = self.header['nx'], self.header['ny'], self.header['nz']
        dx = 0.5**self.level
        off_arr = np.array([self.header['%soff' % dim ][0] for dim in ['y', 'x', 'z']])
        idxarr = np.stack(np.meshgrid(np.arange(ny)+0.5, np.arange(nx)+0.5, np.arange(nz)+0.5), axis=-1)
        self.coo = ((idxarr + off_arr / self.header['dx']) * dx)[:, :, :, [1, 0, 2]]

    def read_file(self, fname):
        # reads grafic2 file format
        ff = FortranFile(join(self.repo, fname))
        header = ff.read_record(grafic_header_dtype)

        nx = int(header['nx'])
        ny = int(header['ny'])
        nz = int(header['nz'])
        data = np.zeros((nx,ny,nz), dtype='f4')

        for i in range(nz):
            data[:, :, i] = ff.read_record('f4').reshape(nx, ny, order='F')
        ff.close()
        return data

    def __getitem__(self, key):
        return self.header[key]

class GraficIC(object):
    # an object to manage multi-level grafic IC
    def __init__(self, repo=None, level_repos=None, levels=None, read_pos=True):
        self.repo = repo
        self.ic = []
        if(level_repos is None and repo is not None):
            level_repos = glob.glob(join(self.repo, 'level_'+'[0-9]'*3))
        if(levels is None):
            levels = [int(level_repo[-3:]) for level_repo in level_repos]
        self.levels = levels
        for level, level_repo in zip(self.levels, level_repos):
            ic = GraficLevel(level_repo, level, read_pos=read_pos)
            self.ic.append(ic)

    def read_ic(self):
        for ic in self.ic:
            ic.read_ic()

    def get_table(self):
        tables = []
        for ic, level in zip(self.ic, self.levels):
            table = ic.get_table()
            table = append_fields(table, 'level', np.full(table.size, level), usemask=False)
            tables.append(table)
        return np.concatenate(tables)

    def __getitem__(self, key):
        return self.ic[key]

class Region():
    def evaluate(self, data):
        if(isinstance(data, np.ndarrray) and data.shape[-1] == 3):
            return self.isin(data)
        elif(isinstance(data, Table)):
            return self.isin(data['pos'])
    def isin(self, points):
        pass

    def get_bounding_box(self):
        pass

    __call__ = evaluate

class BoxRegion(Region):
    def __init__(self, box):
        self.box = box

    def set_center(self, center, extent=None):
        center = np.array(center)
        if(extent is None):
            extent = self.get_extent()
        elif(not np.isscalar(extent)):
            extent = np.array(extent)
        self.box = np.stack([center-extent/2, center+extent/2], axis=-1)

    def get_extent(self):
        return self.box[:, 1] - self.box[:, 0]

    def get_center(self):
        return np.mean(self.box, axis=-1)

    def get_bounding_box(self):
        return self

    def isin(self, points, size=0):
        box = self.box
        mask = np.all((box[:, 0] <= points+size/2) & (points-size/2 <= box[:, 1]), axis=-1)
        return mask

class SphereRegion(Region):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def get_bounding_box(self):
        box = BoxRegion(None)
        box.set_center(self.center, self.radius*2)
        return box

    def isin(self, points, size=0):
        center = self.center
        radius = self.radius
        return rss(points-center) <= radius - size

def part_density(part, reso, mode='m'):
    snap = part.snap
    if(not isinstance(reso, Iterable)):
        reso = np.repeat(reso, 3)
    mhist = np.histogramdd(part['pos'], weights=part['m'], bins=reso, range=snap.box)[0]
    vol = np.prod((snap.box[:, 1] - snap.box[:, 0]) / reso)
    if(mode == 'm'):
        hist = mhist / vol
    elif(mode == 'sig'):
        vel = part['vel']
        sig2 = np.zeros(shape=reso, dtype='f8')
        for idim in np.arange(0, 2):
            mom1 = np.histogramdd(part['pos'], weights=part['m']*vel[:, idim], bins=reso, range=snap.box)[0]
            mom2 = np.histogramdd(part['pos'], weights=part['m']*vel[:, idim]**2, bins=reso, range=snap.box)[0]
            sig2 += mom2/mhist - (mom1/mhist)**2
        hist = np.sqrt(sig2)
    return hist

def get_bytes_data(array):
    barr = array.view('b').reshape((array.size, array.itemsize))
    return barr

def fromndarrays(ndarrays, dtype):
    """
    convert list of ndarray to structured array with given dtype
    faster than np.rec.fromarrays
    only works for 2d arrays for now
    """
    descr = np.dtype(dtype)

    itemsize = 0
    nitem = None
    for nda in ndarrays:
        if(nitem is None):
            nitem = nda.shape[0]
        elif(nitem != nda.shape[0]):
            raise ValueError("Array shape does not match")
        itemsize += nda.shape[1] * nda.dtype.itemsize
    if(descr.itemsize != itemsize):
        raise ValueError(f"Sum of itemsize ({itemsize}) does not match with desired dtype ({descr.itemsize})")

    array = np.zeros(nitem, descr)
    barr = get_bytes_data(array)
    col = 0
    for nda in ndarrays:
        bnda = nda.view('b')
        barr[:, col:col+bnda.shape[1]] = bnda
        col += bnda.shape[1]
    return array