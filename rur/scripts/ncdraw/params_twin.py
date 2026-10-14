from rur import drawer
import numpy as np

######## PARAMS ########
# Snapshots
ZLIST = [3, 1.5, 0.70] # Add reference redshifts
# ZLIST = [] # Only final snapshot
ADD_FINAL = True # Add final (post-processed) snapshot to ZLIST
MULTI = len(ZLIST) > 1


# Drawing
SHOW_SCATTER = not MULTI
# reddish
COLOR1A = 'orangered'    # For good galaxies
COLOR1B = 'mistyrose' # For bad galaxies
COLOR1C = 'firebrick'    # For median points
CMAP_NC = drawer.make_cmap([(1.0,0.8,0.7),(0.9,0.4,0.4),(0.8,0.0,0.0)])
# bluish
COLOR2A = 'dodgerblue'    # For good galaxies
COLOR2B = 'lightblue' # For bad galaxies
COLOR2C = 'mediumblue'    # For median points
CMAP_NH = drawer.make_cmap([(0.7,0.8,1.0),(0.4,0.4,0.9),(0.0,0.0,0.8)])

# "Good" Galaxy Masking
LEVELCUT = 1
NPARTSCUT = 10000
AGECUT = 1.5 # Gyr
SHOW_OTHER = True

# Data Range
XMIN = 10**6    # Galaxy stellar mass minimum
XMAX = 10**13   # Galaxy stellar mass maximum
HXMIN = 10**10      # Halo virial mass minimum
HXMAX = 10**14.5    # Halo virial mass maximum
YMIN = None 
YMAX = None
NBIN = 7
PERCENTILE = [16, 84]

# Detailed
DETAIL={
    'SFMS':{
        'ymin':None,
        'ymax':None,
        'xmin':None,
        'xmax':None,
        'key':'SFR' # 'SFR', 'SFR_r50', 'SFR_r90', 'SFR10', 'SFR10_r50', 'SFR10_r90'
    },
    'SIZE':{
        'ymin':3e-2,
        'ymax':1e1,
        'xmin':None,
        'xmax':None,
        'key':'r50', # 'r50', 'r90', 'r50u', 'r50g', 'r50r', 'r50i', 'r50z', 'r90u', 'r90g', 'r90r', 'r90i', 'r90z'
        'unit':'kpc',
    },
    'Cold':{
        'ymin':1e-2,
        'ymax':None,
        'xmin':None,
        'xmax':None,
        'key':'Mcold_gas_r90' # 'Mcold_gas', 'Mcold_gas_r50', 'Mcold_gas_r90', 'Mdense_gas', 'Mdense_gas_r50', 'Mdense_gas_r90'
    },
    'Metal':{
        'ymin':None,
        'ymax':None,
        'xmin':None,
        'xmax':None,
        'key':'metal' # 'metal', 'metal_gas', 'O/H'
    },
    'OH':{
        'ymin':6.5, # For O/H
        'ymax':9.5, # For O/H
        'xmin':None,
        'xmax':None,
        'key':'O/H'
    },
    'Vsig':{
        'ymin':0,
        'ymax':2,
        'xmin':None,
        'xmax':None,
        'key':'vsig' # 'vsig', 'vsig_r50', 'vsig_r90', 'vsig_gas', 'vsig_gas_r50', 'vsig_gas_r90'
    },
    'MBH':{
        'ymin':1e5,
        'ymax':1e11,
        'xmin':None,
        'xmax':None,
        'key':'MBH',
        'Mseed':1e5, # Below this, not used for fitting
        'dBH':8*68, # pc
    },
    'SHMR':{
        'Hmin':HXMIN,
        'ymin':None,
        'ymax':None,
        'xmin':None,
        'xmax':None,
        'Msep':1e13, # Above this, shown separately
        'Matchlvl':1, # 0: Strict (gal near center), 1: gal inside hal, 2: Loose (gal-hal touched)
    },
    'DTM':{
        'ymin':None,
        'ymax':None,
        'xmin':None,
        'xmax':None,
        'key':{'CDustSmall_gas':1, 'CDustLarge_gas':1, 'SiDustLarge_gas':1/0.163, 'SiDustSmall_gas':1/0.163}
    },
    'CMD':{
        'ymin':-0.6,
        'ymax': 0.9,
        'xmin':-26,
        'xmax':-9.5,
        'colorkey':'g-r', # 'u-g', 'g-r', 'r-i', 'i-z'
        'band':'r' # 'u', 'g', 'r', 'i', 'z'
    },
    'SB':{
        'ymin':14,
        'ymax':32,
        'xmin':-26,
        'xmax':-9.5,
        'band':'r' # 'u', 'g', 'r', 'i', 'z'
    },
    'alpha':{
        'ymin':-0.4,
        'ymax': 1.2,
        'xmin': -3,
        'xmax':  0.8,
    },
}