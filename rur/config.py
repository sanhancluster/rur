from rur.sci.geometry import rss, ss
import numpy as np
from collections import defaultdict
import time
from tqdm.autonotebook import tqdm

def defnone(): return None
class Table:
    """
    Table class to store RAMSES particle/AMR data.
    Basically acts as numpy.recarray, but some functions do not work.
    """

    def __init__(self, table, snap, units=None):
        self.table = table
        self.snap = snap
        self.extra_fields = custom_extra_fields(snap, 'common')
        if units is None:
            units = {}
        # unit of table data in dict form, returns 'None' by default, which indicates code unit
        self.units = defaultdict(defnone)
        self.units.update(units)
        self.size = self.table.size

    def __getitem__(self, item, return_code_unit=False):
        if isinstance(item, str):
            if item in self.extra_fields.keys():
                return self.extra_fields[item](self)
            else:
                if return_code_unit:
                    return self.table[item] * self.snap.unit[self.units[item]]
                else:
                    return self.table[item]
        elif isinstance(item, tuple):  # if unit is given
            letter, unit = item
            if(letter in dim_keys)or(letter in vel_keys):
                if(self.snap.unitmode != 'code'):
                    print(f"Warning! Current unit is already physical! Don't trust this result!")
            return self.__getitem__(letter, return_code_unit=True) / self.snap.unit[unit]
        else:
            return self.__copy__(self.table[item])

    def __len__(self):
        return len(self.table)

    def __str__(self):
        return self.table.__str__()

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.__dict__)

    def __setitem__(self, key, value):
        return self.table.__setitem__(key, value)

    def __copy__(self, table=None, snap=None, units=None):
        if table is None:
            table = self.table
        if snap is None:
            snap = self.snap
        if units is None:
            units = self.units
        return self.__class__(table, snap, units)
    
    @property
    def shape(self):
        return self.table.shape
    @property
    def dtype(self):
        return self.table.dtype


class Timer:
    def __init__(self, unitl='s', verbose=0):
        self.t = time.time()
        self.unitl = unitl
        if (unitl == 'h'):
            self.unit = 3600
        elif (unitl == 'm'):
            self.unit = 60
        else:
            self.unit = 1
        self.verbose = verbose
        self.verbose_lim = 1

    def start(self, message=None, verbose_lim=None, tab=0):
        if (verbose_lim is not None):
            self.verbose_lim = verbose_lim

        if self.verbose >= self.verbose_lim and message is not None:
            ntab = '\t'*tab
            print(f"{ntab}{message}")
        self.t = time.time()

    def time(self):
        return (time.time() - self.t) / self.unit

    def record(self, verbose_lim=None, tab=0):
        if verbose_lim is not None:
            self.verbose_lim = verbose_lim

        if self.verbose >= self.verbose_lim:
            # print('Done (%.3f%s).' % (self.time(), self.unitl))
            ntab = '\t'*tab
            print(f"{ntab}Done ({self.time():.3f}{self.unitl}).")

    def measure(self, func, message=None, tab=0, **kwargs):
        self.start(message, tab=tab)
        result = func(**kwargs)
        self.record(tab=tab)
        return result

CYAN = "\033[36m"
GREEN = "\033[33m"
RESET = "\033[0m"
class Timestamp:
    """
    A class to export time that took to execute the script.
    """

    def __init__(self):
        self.t0 = time.time()
        self.stamps = {}
        self.stamps['start'] = self.t0
        self.stamps['last'] = self.t0
        self.verbose = 1

    def elapsed(self, name=None):
        if name is None:
            name = 'start'
        t = self.stamps[name]
        return time.time() - t
    
    def time(self):
        """
        Returns the elapsed time since the start or a specific name.
        """
        return self.elapsed(name='start')

    def start(self, message=None, name=None):
        if name is None:
            name = 'last'
        self.stamps[name] = time.time()
        if message is not None:
            self.message(message)

    def message(self, message, verbose_lim=1):
        if verbose_lim <= self.verbose:
            time = self.elapsed()
            time_string = self.get_time_string(time, add_units=True)
            print(f"{CYAN}[ {time_string} ]{RESET} {message}")

    def record(self, message=None, name=None, verbose_lim=1):
        if name is None:
            name = 'last'
        if verbose_lim <= self.verbose:
            time = self.elapsed()
            time_string = self.get_time_string(time, add_units=True)
            recorded_time = self.elapsed(name)
            recorded_time_string = self.get_time_string(recorded_time, add_units=True)
            if message is None:
                message = "Done."
            print(f"{CYAN}[ {time_string} ]{RESET} {message} -> {GREEN}{recorded_time_string}{RESET}")
        #self.stamps.pop(name)

    def get_time_string(self, elapsed_time, add_units=False):
        """
        Convert elapsed time in seconds to a formatted string.
        """
        time_format = "%H:%M:%S"
        if elapsed_time < 60:
            if add_units:
                return f"{elapsed_time:05.2f}s"
            else:
                return f"{elapsed_time:05.2f}"
        elif elapsed_time < 3600:
            if add_units:
                time_format = "%Mm %Ss"
            else:
                time_format = "%M:%S"
        elif elapsed_time < 86400:
            if add_units:
                time_format = "%Hh %Mm %Ss"
            else:
                time_format = "%H:%M:%S"
        else:
            elapsed_day = elapsed_time // 86400  # Convert to days
            if add_units:
                time_format = f"{elapsed_day}d %Hh %Mm %Ss"
            else:
                time_format = f"{elapsed_day} %H:%M:%S"
        return time.strftime(time_format, time.gmtime(elapsed_time))

    def measure(self, func, message=None, **kwargs):
        self.start(message)
        result = func(**kwargs)
        self.record()
        return result

def get_vector(table, prefix='', ndim=3):
    return np.stack([table[prefix + key] for key in dim_keys[:ndim]], axis=-1)

class alias_dict(dict):
    def __missing__(self, key):
        return key

oct_offset = np.array([
    -0.5,  0.5, -0.5,  0.5, -0.5,  0.5, -0.5,  0.5,
    -0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5,
    -0.5, -0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5 
    ]).reshape(3,8).T
oct_x = oct_offset[:, 0].reshape(8, 1)
oct_y = oct_offset[:, 1].reshape(8, 1)
oct_z = oct_offset[:, 2].reshape(8, 1)

# path_related parameters
# avaiable modes: none, ng, nh, etc.
default_path_in_repo = {
    'snapshots': 'snapshots',
    'GalaxyMaker': 'galaxy',
    'HaloMaker': 'halo',
}

output_format = 'output_{snap.iout:05d}'
output_regex = r'output_(?P<iout>\d{5})'
output_glob = 'output_[0-9][0-9][0-9][0-9][0-9]'
sinkprop_glob = 'sink_[0-9][0-9][0-9][0-9][0-9].dat'



sinkprop_format = 'sink_{icoarse:05d}.dat'

default = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'), ('m', 'f8')]

desc2dtype = {
    'position_x':'x', 'position_y':'y', 'position_z':'z',
    'velocity_x':'vx', 'velocity_y':'vy', 'velocity_z':'vz',
    'mass':'m',
    'identity':'id', 'levelp':'level',
    'family':'family', 'tag':'tag',
    'birth_time':'epoch', 'metallicity':'metal', 'initial_mass':'m0',
    'chem_H':'H', 'chem_O':'O', 'chem_Fe':'Fe',
    'chem_Mg':'Mg', 'chem_C':'C', 'chem_N':'N',
    'chem_Si':'Si', 'chem_S':'S', 'chem_D':'D',
    'birth_density':'rho0', 'partp':'partp',
    'density':'rho', 'pressure':'P', 'thermal_pressure':'P',
    'dust_bin01':'d1', 'dust_bin02':'d2', 'dust_bin03':'d3', 'dust_bin04':'d4',
    'refinement_scalar':'refmask', 'scalar_15':'sigma'
}

desc2dtype_sink = {
    'identity':'id', 'mass':'m',
    'position_x':'x', 'position_y':'y', 'position_z':'z',
    'velocity_x':'vx', 'velocity_y':'vy', 'velocity_z':'vz',
    'birth_time':'tform', 'dMsmbh':'dM', 'dMBH_coarse':'dMBH', 'dMEd_coarse':'dMEd',
    'Esave':'Esave', 'jsink_x':'jx', 'jsink_y':'jy', 'jsink_z':'jz',
    'spin_x':'sx', 'spin_y':'sy', 'spin_z':'sz', 'spin_magnitude':'spinmag'
}

format_f2py = {
    'd':'f8',
    'i':'i4',
    'b':'i1'
}

sink_prop_dtype_drag = [
    ('id', 'i4'), ('n_star', 'i4'), ('n_dm', 'i4'), ('m', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'),
    ('vy', 'f8'), ('vz', 'f8'),
    ('gas_jx', 'f8'), ('gas_jy', 'f8'), ('gas_jz', 'f8'), ('Mdot', 'f8'), ('Medd', 'f8'), ('dM', 'f8'),
    ('d_avgptr', 'f8'), ('c_avgptr', 'f8'), ('v_avgptr', 'f8'), ('Esave', 'f8'),
    ('jx', 'f8'), ('jy', 'f8'), ('jz', 'f8'), ('spinmag', 'f8'), ('eps_sink', 'f8'),
    ('rho_star', 'f8'), ('rho_dm', 'f8'), ('star_vx', 'f8'), ('star_vy', 'f8'), ('star_vz', 'f8'), ('dm_vx', 'f8'),
    ('dm_vy', 'f8'), ('dm_vz', 'f8'),
    ('low_star', 'f8'), ('low_dm', 'f8'), ('fast_star', 'f8'), ('fast_dm', 'f8')
]

sink_prop_dtype_drag_fornax = [
    ('id', 'i4'), ('n_star', 'i4'), ('n_dm', 'i4'), ('m', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'),
    ('vy', 'f8'), ('vz', 'f8'),
    ('gas_jx', 'f8'), ('gas_jy', 'f8'), ('gas_jz', 'f8'), ('epoch', 'f8'), ('Mdot', 'f8'), ('Medd', 'f8'), ('dM', 'f8'),
    ('d_avgptr', 'f8'), ('c_avgptr', 'f8'), ('v_avgptr', 'f8'), ('Esave', 'f8'),
    ('jx', 'f8'), ('jy', 'f8'), ('jz', 'f8'), ('spinmag', 'f8'), ('eps_sink', 'f8'),
    ('rho_star', 'f8'), ('rho_dm', 'f8'), ('star_vx', 'f8'), ('star_vy', 'f8'), ('star_vz', 'f8'), ('dm_vx', 'f8'),
    ('dm_vy', 'f8'), ('dm_vz', 'f8'),
    ('low_star', 'f8'), ('low_dm', 'f8'), ('fast_star', 'f8'), ('fast_dm', 'f8')
]

sink_prop_dtype_drag_y2 = [
    ('id', 'i4'), ('n_star', 'i4'), ('n_dm', 'i4'), ('m', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'),
    ('vy', 'f8'), ('vz', 'f8'),
    ('gas_jx', 'f8'), ('gas_jy', 'f8'), ('gas_jz', 'f8'), ('epoch', 'f8'), ('Mdot', 'f8'), ('Medd', 'f8'), ('dM', 'f8'),
    ('d_avgptr', 'f8'), ('c_avgptr', 'f8'), ('v_avgptr', 'f8'), ('Esave', 'f8'),
    ('jx', 'f8'), ('jy', 'f8'), ('jz', 'f8'), ('spinmag', 'f8'), ('eps_sink', 'f8'),
    ('rho_star', 'f8'), ('rho_dm', 'f8'), ('star_vx', 'f8'), ('star_vy', 'f8'), ('star_vz', 'f8'), ('dm_vx', 'f8'),
    ('dm_vy', 'f8'), ('dm_vz', 'f8'),
    ('low_star', 'f8'), ('low_dm', 'f8'), ('fast_star', 'f8'), ('fast_dm', 'f8'), ('DF_gas', 'f8'), ('DF_star', 'f8'),
    ('DF_dm', 'f8')
]

sink_prop_dtype = [
    ('id', 'i4'), ('m', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'),
    ('gas_jx', 'f8'), ('gas_jy', 'f8'), ('gas_jz', 'f8'), ('Mdot', 'f8'), ('Medd', 'f8'), ('dM', 'f8'),
    ('d_avgptr', 'f8'), ('c_avgptr', 'f8'), ('v_avgptr', 'f8'), ('Esave', 'f8'),
    ('jx', 'f8'), ('jy', 'f8'), ('jz', 'f8'), ('spinmag', 'f8'), ('eps_sink', 'f8')
]

sink_table_dtype = [('id', 'i8'), ('m', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'),
                    ('vz', 'f8')]

sink_dtype = [('id', 'i4'), ('m', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'),
              ('vz', 'f8'),
              ('tform', 'f8'), ('dM', 'f8'), ('dMBH', 'f8'), ('dMEd', 'f8'), ('Esave', 'f8'),
              ('jx', 'f8'), ('jy', 'f8'), ('jz', 'f8'), ('sx', 'f8'), ('sy', 'f8'), ('sz', 'f8'), ('spinmag', 'f8')]

grafic_header_dtype = [('nx', 'i4'), ('ny', 'i4'), ('nz', 'i4'),
                       ('dx', 'f4'),
                       ('xoff', 'f4'), ('yoff', 'f4'), ('zoff', 'f4'),
                       ('aexp_start', 'f4'),
                       ('Om', 'f4'), ('Ol', 'f4'), ('H0', 'f4')]

iout_avail_dtype = [('iout', 'i4'), ('aexp', 'f8'), ('age', 'f8'), ('icoarse', 'i4'), ('time', 'f8')]

icoarse_avail_dtype = [('icoarse', 'i4'), ('aexp', 'f8'), ('age', 'f8'), ('time', 'f8')]


part_family = {
    'cloud_tracer': -3,
    'star_tracer': -2,
    'gas_tracer': 0,
    'dm': 1,
    'star': 2,
    'cloud': 3,
    'sink': 3,
    'tracer': [-3, -2, -1, 0],
    'init': [-3, -2, -1, 0, 1],
}

dim_keys = ['x', 'y', 'z']
vel_keys = ['vx', 'vy', 'vz']

# some units in cgs
Msol = 2E33  # Msol = 1.9885E33

# parsecs: ramses unit
Gpc = 3.08E27
Mpc = 3.08E24
kpc = 3.08E21
pc = 3.08E18
ly = 9.4607E17
km = 1E5

Gyr = 3.154E16
Myr = 3.154E13
kyr = 3.154E10
yr = 3.154E7

# some constants in cgs
G_const = 6.673E-8  # Gravitational constant in
c_const = 2.99792458E10  # Speed of light
sigma_T = 6.6524E-25  # Thomson cross section
k_B = 1.38062000E-16  # Boltzmann constant (RAMSES:cooling_module.f90)
m_H = 1.6600000E-24  # hydrogen atomic mass (RAMSES:cooling_module.f90)
XH = 0.76 # hydrogen mass fraction (RAMSES:cooling_module.f90)
# others
gamma = 1.6666667

# some preferences
progress_bar_limit = 100

verbose_default = 1
timer = Timer(verbose=verbose_default)
default_box = np.array([[0, 1], [0, 1], [0, 1]])


# custom units used for RAMSES snapshot, conversion from code unit to conventional unit.
def set_custom_units(snap):
    params = snap.params
    l = params['unit_l']
    m = params['unit_m']
    t = params['unit_t']
    d = params['unit_d']
    boxsize_comoving = params['boxsize'] / snap.h

    snap.unit = {
        # Length
        'cm': 1E0 / l,
        'm': 1E2 / l,
        'km': 1E5 / l,
        'pc': pc / l,
        'kpc': kpc / l,
        'Mpc': Mpc / l,
        'Gpc': Gpc / l,
        'ly': ly / l,

        'cpc':  1E-6 / boxsize_comoving,
        'ckpc': 1E-3 / boxsize_comoving,
        'cMpc': 1E0 / boxsize_comoving,
        'cGpc': 1E3 / boxsize_comoving,

        'cpc/h':  1E-6 / boxsize_comoving / snap.h,
        'ckpc/h': 1E-3 / boxsize_comoving / snap.h,
        'cMpc/h': 1E0 / boxsize_comoving / snap.h,

        'pc/h':  pc / l / snap.h,
        'kpc/h': kpc / l / snap.h,
        'Mpc/h': Mpc / l / snap.h,
        'Gpc/h': Gpc / l / snap.h,

        # Mass
        'g': 1 / m,
        'kg': 1E3 / m,
        'Msol': Msol / m,
        'Msun': Msol / m,

        # Time
        'yr': yr / t,
        'kyr': kyr / t,
        'Myr': Myr / t,
        'Gyr': Gyr / t,

        # Density
        'g/cc': 1E0 / d,
        'H/cc': m_H / d / XH,
        'Msol/Mpc3': Msol / Mpc ** 3 / d,
        'Msol/kpc3': Msol / kpc ** 3 / d,

        # Velocity
        'km/s': 1E5 * t / l,
        'cm/s': t / l,

        # Temperature
        'K': t ** 2 / l ** 2 * k_B / m_H,

        # Column density
        'Msol/Mpc2': Msol / Mpc ** 2 / m * l ** 2,
        'Msol/kpc2': Msol / kpc ** 2 / m * l ** 2,
        'Msol/pc2': Msol / pc ** 2 / m * l ** 2,
        'H/cm2': m_H / m * l ** 2,

        # Pressure
        'Ba': t ** 2 * l / m,

        # Mass Flux
        'Msol/yr': Msol / yr / m * t,
        'g/s': t / m,

        # Energy
        'erg': t ** 2 / l ** 2 / m,

        # Energy Density
        'erg/cc': t ** 2 * l / m,

        None: 1E0
    }


# some extra quantities that can be used as key of particle / cell data
def custom_extra_fields(snap, type='common'):
    extra_fields = {
        'pos': lambda table: get_vector(table),  # position vector
        'vel': lambda table: get_vector(table, 'v'),  # velocity vector
        #        'FeH': lambda table: 1.024*np.log10(table['metal']) + 1.739
    }
    if type == 'common':
        return extra_fields

    elif type == 'cell':
        # cell extra keys
        mfactor = 1 if(snap.unitmode == 'code') else snap.unit['Msol']
        extra_fields.update({
            'T': lambda table: table['P'] / table['rho'],  # temperature
            'vol': lambda table: table['dx'] ** 3,  # cell volume
            'm': lambda table: table['vol'] * table['rho']/mfactor,  # cell mass
            'cs': lambda table: np.sqrt(gamma * table['P'] / table['rho']),  # sound speed
            'mach': lambda table: rss(table['vel']) / np.sqrt(gamma * table['P'] / table['rho']),  # mach number
            'e': lambda table: table['P'] / (gamma - 1) + 0.5 * table['rho'] * ss(table['vel']),  # total energy density
            'dx': lambda table: snap.boxlen / snap.unitfactor / 2. ** table['level'],  # spatial resolution
        })

    elif type == 'particle':
        # particle extra keys
        extra_fields.update({
            'age': lambda table: (snap.age - snap.epoch_to_age(table['epoch'])) * snap.unit['Gyr'],  # stellar age
            'aform': lambda table: snap.epoch_to_aexp(table['epoch']),  # formation epoch
            'zform': lambda table: 1. / table['aform'] - 1,  # formation epoch
            'dx': lambda table: snap.boxlen / snap.unitfactor / 2. ** table['level'],  # spatial resolution
        })

    elif type == 'halo':
        # halo extra keys
        extra_fields.update({
        })

    elif type == 'smbh':
        # halo extra keys
        extra_fields.update({
        })

    return extra_fields
