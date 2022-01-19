from rur.utool import get_vector, rotate_vector, rss, ss, Timer, get_box, expand_shape, Table
import numpy as np

def type_of_script():
    """
    Detects and returns the type of python kernel
    :return: string 'jupyter' or 'ipython' or 'terminal'
    """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'

if(type_of_script() == 'jupyter'):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

if(type_of_script() == 'jupyter'):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class alias_dict(dict):
    def __missing__(self, key):
        return key

# avaiable modes: none, ng, nh
output_format = 'output_{snap.iout:05d}'
output_regex = r'output_(?P<iout>\d{5})'
output_glob = 'output_[0-9][0-9][0-9][0-9][0-9]'
sinkprop_glob = 'sink_[0-9][0-9][0-9][0-9][0-9].dat'

info_format = {
    'ng': 'info.txt',
}
info_format.update(dict.fromkeys(['nh', 'nh_dm_only', 'none', 'yzics', 'yzics_dm_only', 'iap', 'gem', 'fornax', 'y2', 'y3', 'y4'], 'info_{snap.iout:05d}.txt'))

data_format = {
    'ng': '{{type}}.out{{icpu:05d}}',
}

sinkprop_format = 'sink_{icoarse:05d}.dat'

data_format.update(dict.fromkeys(['nh', 'nh_dm_only', 'none', 'yzics', 'yzics_dm_only', 'iap', 'gem', 'fornax', 'y2', 'y3', 'y4'], '{{type}}_{snap.iout:05d}.out{{icpu:05d}}'))

default = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'), ('m', 'f8')]

# columns for particle table, see readr.f90
part_dtype = {
    'yzics': default + [('epoch', 'f8'), ('metal', 'f8'), ('id', 'i4'), ('level', 'u1'), ('cpu', 'i4')],
    'yzics_dm_only': default + [('id', 'i4'), ('level', 'u1'), ('cpu', 'i4')],

    'nh': default + [('epoch', 'f8'), ('metal', 'f8'), ('id', 'i4'), ('level', 'u1'), ('cpu', 'i4')],
    'nh_dm_only' : default + [('id', 'i4'), ('level', 'u1'), ('cpu', 'i4')],

    'none': default + [('epoch', 'f8'), ('id', 'i4'), ('level', 'u1'), ('cpu', 'i4'), ('family', 'i1'), ('tag', 'i1')],
    'iap': default + [('epoch', 'f8'), ('metal', 'f8'), ('id', 'i4'), ('level', 'u1'), ('cpu', 'i4'), ('family', 'i1'), ('tag', 'i1')],
    'gem': default + [('epoch', 'f8'), ('metal', 'f8'), ('id', 'i4'), ('level', 'u1'), ('cpu', 'i4'), ('family', 'i1'), ('tag', 'i1')],
    'fornax': default + [('epoch', 'f8'), ('metal', 'f8'), ('id', 'i4'), ('level', 'u1'), ('cpu', 'i4'), ('family', 'i1'), ('tag', 'i1')],
    'y2': default + [('epoch', 'f8'), ('metal', 'f8'), ('m0', 'f8'),
                     ('H', 'f8'), ('O', 'f8'), ('Fe', 'f8'), ('Mg', 'f8'),
                     ('C', 'f8'), ('N', 'f8'), ('Si', 'f8'), ('S', 'f8'),
                     ('id', 'i4'), ('level', 'u1'), ('cpu', 'i4'), ('partp', 'i4'),
                     ('family', 'i1'), ('tag', 'i1')],
    'y3': default + [('epoch', 'f8'), ('metal', 'f8'), ('m0', 'f8'),
                     ('H', 'f8'), ('O', 'f8'), ('Fe', 'f8'), ('Mg', 'f8'),
                     ('C', 'f8'), ('N', 'f8'), ('Si', 'f8'), ('S', 'f8'),
                     ('rho0', 'f8'),
                     ('id', 'i4'), ('level', 'u1'), ('cpu', 'i4'), ('partp', 'i4'),
                     ('family', 'i1'), ('tag', 'i1')],
    'y4': default + [('epoch', 'f8'), ('metal', 'f8'), ('m0', 'f8'),
                     ('H', 'f8'), ('O', 'f8'), ('Fe', 'f8'), ('Mg', 'f8'),
                     ('C', 'f8'), ('N', 'f8'), ('Si', 'f8'), ('S', 'f8'), ('D', 'f8'),
                     ('rho0', 'f8'),
                     ('id', 'i4'), ('level', 'u1'), ('cpu', 'i4'), ('partp', 'i4'),
                     ('family', 'i1'), ('tag', 'i1')],

    'gem_longint': default + [('epoch', 'f8'), ('metal', 'f8'), ('id', 'i8'), ('level', 'u1'), ('cpu', 'i4'), ('family', 'i1'), ('tag', 'i1')],

    'ng': default + [('id', 'i4'), ('level', 'u1'), ('cpu', 'i4')],
}

sink_prop_dtype_drag = [
    ('id', 'i4'), ('n_star', 'i4'), ('n_dm', 'i4'), ('m', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'),
    ('gas_jx', 'f8'), ('gas_jy', 'f8'), ('gas_jz', 'f8'), ('Mdot', 'f8'), ('Medd', 'f8'), ('dM', 'f8'),
    ('d_avgptr', 'f8'), ('c_avgptr', 'f8'), ('v_avgptr', 'f8'), ('Esave', 'f8'),
    ('jx', 'f8'), ('jy', 'f8'), ('jz', 'f8'), ('spinmag', 'f8'), ('eps_sink', 'f8'),
    ('rho_star', 'f8'), ('rho_dm', 'f8'), ('star_vx', 'f8'), ('star_vy', 'f8'), ('star_vz', 'f8'), ('dm_vx', 'f8'), ('dm_vy', 'f8'), ('dm_vz', 'f8'),
    ('low_star', 'f8'), ('low_dm', 'f8'), ('fast_star', 'f8'), ('fast_dm', 'f8')
]

sink_prop_dtype_drag_fornax = [
    ('id', 'i4'), ('n_star', 'i4'), ('n_dm', 'i4'), ('m', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'),
    ('gas_jx', 'f8'), ('gas_jy', 'f8'), ('gas_jz', 'f8'), ('epoch', 'f8'), ('Mdot', 'f8'), ('Medd', 'f8'), ('dM', 'f8'),
    ('d_avgptr', 'f8'), ('c_avgptr', 'f8'), ('v_avgptr', 'f8'), ('Esave', 'f8'),
    ('jx', 'f8'), ('jy', 'f8'), ('jz', 'f8'), ('spinmag', 'f8'), ('eps_sink', 'f8'),
    ('rho_star', 'f8'), ('rho_dm', 'f8'), ('star_vx', 'f8'), ('star_vy', 'f8'), ('star_vz', 'f8'), ('dm_vx', 'f8'), ('dm_vy', 'f8'), ('dm_vz', 'f8'),
    ('low_star', 'f8'), ('low_dm', 'f8'), ('fast_star', 'f8'), ('fast_dm', 'f8')
]

sink_prop_dtype_drag_y2 = [
    ('id', 'i4'), ('n_star', 'i4'), ('n_dm', 'i4'), ('m', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'),
    ('gas_jx', 'f8'), ('gas_jy', 'f8'), ('gas_jz', 'f8'), ('epoch', 'f8'), ('Mdot', 'f8'), ('Medd', 'f8'), ('dM', 'f8'),
    ('d_avgptr', 'f8'), ('c_avgptr', 'f8'), ('v_avgptr', 'f8'), ('Esave', 'f8'),
    ('jx', 'f8'), ('jy', 'f8'), ('jz', 'f8'), ('spinmag', 'f8'), ('eps_sink', 'f8'),
    ('rho_star', 'f8'), ('rho_dm', 'f8'), ('star_vx', 'f8'), ('star_vy', 'f8'), ('star_vz', 'f8'), ('dm_vx', 'f8'), ('dm_vy', 'f8'), ('dm_vz', 'f8'),
    ('low_star', 'f8'), ('low_dm', 'f8'), ('fast_star', 'f8'), ('fast_dm', 'f8'), ('DF_gas', 'f8'), ('DF_star', 'f8'), ('DF_dm', 'f8')
]


sink_prop_dtype = [
    ('id', 'i4'), ('m', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'),
    ('gas_jx', 'f8'), ('gas_jy', 'f8'), ('gas_jz', 'f8'), ('Mdot', 'f8'), ('Medd', 'f8'), ('dM', 'f8'),
    ('d_avgptr', 'f8'), ('c_avgptr', 'f8'), ('v_avgptr', 'f8'), ('Esave', 'f8'),
    ('jx', 'f8'), ('jy', 'f8'), ('jz', 'f8'), ('spinmag', 'f8'), ('eps_sink', 'f8')
]

sink_table_dtype = [('id', 'i8'), ('m', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8')]

grafic_header_dtype = [('nx', 'i4'), ('ny', 'i4'), ('nz', 'i4'),
             ('dx', 'f4'),
             ('xoff', 'f4'), ('yoff', 'f4'), ('zoff', 'f4'),
             ('aexp_start', 'f4'),
             ('Om', 'f4'), ('Ol', 'f4'), ('H0', 'f4')]

# columns for hydro quantity table, all float64, see readr.f90
hydro_names = {
    'nh': ['rho', 'vx', 'vy', 'vz', 'P', 'metal', 'refmask'],
    'nh_dm_only': ['rho', 'vx', 'vy', 'vz', 'P', 'metal', 'refmask'],
    'yzics': ['rho', 'vx', 'vy', 'vz', 'P', 'metal'],
    'yzics_dm_only': ['rho', 'vx', 'vy', 'vz', 'P', 'metal'],
    'none': ['rho', 'vx', 'vy', 'vz', 'P'],
    'iap': ['rho', 'vx', 'vy', 'vz', 'P', 'metal', 'refmask'],
    'gem': ['rho', 'vx', 'vy', 'vz', 'P', 'metal', 'dust', 'refmask'],
    'fornax': ['rho', 'vx', 'vy', 'vz', 'P', 'metal', 'dust', 'refmask'],
    'y2': ['rho', 'vx', 'vy', 'vz', 'P', 'metal', 'H', 'O', 'Fe', 'Mg', 'C', 'N', 'Si', 'S', 'dust', 'refmask'],
    'y3': ['rho', 'vx', 'vy', 'vz', 'P', 'metal', 'H', 'O', 'Fe', 'Mg', 'C', 'N', 'Si', 'S', 'refmask'],
    'y4': ['rho', 'vx', 'vy', 'vz', 'P', 'metal', 'H', 'O', 'Fe', 'Mg', 'C', 'N', 'Si', 'S', 'D', 'refmask'],
    'ng': ['rho', 'vx', 'vy', 'vz', 'P'],
}

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
Msol = 2E33 # Msol = 1.9885E33

# parsecs: ramses unit
Mpc = 3.08E24
kpc = 3.08E21
pc = 3.08E18
ly = 9.4607E17
km = 1E5

Gyr = 3.154E16
Myr = 3.154E13
yr = 3.154E7

# some constants in cgs
k_B = 1.38064852E-16 # Boltzmann constant
m_H = 1.6737236E-24 # hydrogen atomic mass

# others
gamma = 1.6666667

# some preferences
progress_bar_limit = 100

verbose_default = 1
timer = Timer(verbose=verbose_default)
default_box = np.array([[0, 1], [0, 1], [0, 1]])

# custom units used for RAMSES snapshot, conversion from code unit to conventional unit.
def custom_units(snap):
    params = snap.params
    l = params['unit_l']
    m = params['unit_m']
    t = params['unit_t']
    d = params['unit_d']

    snap.unit = {
        # Length
        'cm'  : 1E0 / l,
        'm'   : 1E2 / l,
        'km'  : 1E5 / l,
        'pc'  : pc / l,
        'kpc' : kpc / l,
        'Mpc' : Mpc / l,
        'ly'  : ly / l,

        'ckpc/h': 1E-3 / snap.boxsize / snap.h,
        'cMpc/h': 1E0 / snap.boxsize / snap.h,

        'kpc/h': kpc / l / snap.h,
        'Mpc/h': Mpc / l / snap.h,

        # Mass
        'g'  : 1 / m,
        'kg'  : 1E3 / m,
        'Msol': Msol / m,
        'Msun': Msol / m,

        # Time
        'yr'  : yr / t,
        'Myr' : Myr / t,
        'Gyr' : Gyr / t,

        # Density
        'g/cc': 1E0 / d,
        'H/cc': m_H / d,
        'Msol/Mpc3': Msol / Mpc ** 3 / d,
        'Msol/kpc3': Msol / kpc ** 3 / d,

        # Velocity
        'km/s': 1E5 * t / l,
        'cm/s': t / l,

        # Temperature
        'K'   : t ** 2 / l ** 2 * k_B / m_H,

        # Column density
        'Msol/Mpc2': Msol / Mpc ** 2 / m * l ** 2,
        'Msol/kpc2': Msol / kpc ** 2 / m * l ** 2,
        'Msol/pc2': Msol / pc ** 2 / m * l ** 2,
        'H/cm2': m_H / m * l ** 2,

        # Pressure
        'Ba'  : t**2 * l / m,

        # Mass Flux
        'Msol/yr': Msol / yr / m * t,
        'g/s': t / m,

        # Energy
        'erg': t**2 / l**2 / m,

        # Energy Density
        'erg/cc': t ** 2 * l / m,

        None  : 1
    }

# some extra quantities that can be used as key of particle / cell data
def custom_extra_fields(snap):
    common_extra = {
        'pos': lambda table: get_vector(table),  # position vector
        'vel': lambda table: get_vector(table, 'v'),  # velocity vector
        'dx': lambda table: 0.5 ** table['level'], # spatial resolution
        'FeH': lambda table: 1.024*np.log10(table['metal']) + 1.739
    }

    # cell extra keys
    cell_extra = {
        'T': lambda table: table['P'] / table['rho'], # temperature
        'vol': lambda table: table['dx'] ** 3, # cell volume
        'm': lambda table: table['vol'] * table['rho'], # cell mass
        'cs' : lambda table: np.sqrt(gamma * table['P'] / table['rho']), # sound speed
        'mach': lambda table: rss(table['vel']) / np.sqrt(gamma * table['P'] / table['rho']), # mach number
        'e': lambda table: table['P'] / (gamma - 1) + 0.5 * table['rho'] * ss(table['vel']), # total energy density
    }

    # particle extra keys
    part_extra = {
        'age': lambda table: snap.epoch_to_age(table['epoch']), # stellar age
        'aform': lambda table: snap.epoch_to_aexp(table['epoch']),  # formation epoch
        'zform': lambda table: 1./table['aform']-1,  # formation epoch
    }

    cell_extra.update(common_extra)
    part_extra.update(common_extra)
    return cell_extra, part_extra

grafic_header_dtype = np.dtype([('nx', 'i4'), ('ny', 'i4'), ('nz', 'i4'),
                                ('dx', 'f4'),
                                ('xoff', 'f4'), ('yoff', 'f4'), ('zoff', 'f4'),
                                ('aexp_start', 'f4'),
                                ('Om', 'f4'), ('Ol', 'f4'), ('H0', 'f4')])
