# coding=utf-8

#!!! Ground 
repo = '/home/jangjk/SKIRT_DIR/rur_dir/testing/' # The path where you want to save the input files
repo_output = 'result/' # If exe_skirt = True, the output files will be saved in repo+repo_output.

opart_name = 'part_old.txt' # old population stellar input will be saved with this given file name 
ypart_name = 'part_young.txt' # younger population stellar input will be saved with this given file name 
gas_name = 'gas_cell.txt' # gaseous input will be saved with this given file name 

ski_name = 'nh_ext.ski' # ski file (the parameter file that SKIRT read) will be saved with this given file name 

exe_skirt = True # If you want to execute SKIRT after saving the input files 
del_input = False # If you want to delete the input particle/gaseous files after running SKIRT

skirt_dir = '/home/jangjk/SKIRT_DIR/SKIRT9/release/SKIRT/main/' # the path where the executable skirt object reside
N_thread = 40 # the number of the thread you want to use
N_phot = 2e7 # The number of the photon packets

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!    Load & pre-set the raw sim data    !!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

#!!! if you want to change the spatial unit to co-moving scale
co_moving = False

#!!! if you want to rotate the galaxy w.r.t the net ang-momentum direction of star particles
rotating_gal = True

#@@@ Sampling criteria
R_low = 2 # Lower bound of the 3d radius when you sample the star particles to calculate net ang-mom 
R_upp = 8 # Upper bound of the 3d radius "
          # Unit: [kpc]

A_upp = 8 # Upper bound of the Age(not a look-back time) "
          # Unit: [Gyr]

mass_weight = True # Apply mass weight when you calculate the angular momentum 

par_mode = 'median' # Simple strategy for calculating net angular momentum from the sample 
                  # Possible option: 'mean', 'median'

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!    Make an input data with SKIRT-readable format    !!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

dx_s = 50  # This is a smoothing length of each stellar particle
           # When we put the data into SKIRT, we treated the stellar particle as an smoothed particle data  
           # Unit: [parsec]

import_velocity = 'false' # If you want to include the velocity info (w.r.t the galactic center) of star particle
                       # set this parameter to 'true'. 

import_dispersion = 'false' # If you want to assume the velocity dispersion inside each star particle,
                          # set this parameter  to 'true'. 
sig_s = 10 # Assumed velocity dispersion of stellar particles
           # Unit: [km/s]

add_young = True # If you want to separate the young stellar particles from the old population
               # If not True, then all the stellar particles will be 
Age_Cut = 0.01 # Demarcation cut for the separation
               # Unit: [Gyr]


# When we dealing with the gaseous cells,
# We have two ways to make an input data
  # 1. make an input data as it is (AMR structure) with a morton order
  # 2. treat each gaseous cells as a smoothed particles with a different smoothing length
# Of course we can use both of the options, but I didn't include Option 1 for now.

fact_dx = 1.5 # So we will treat each cell as a particle with a smoothing length
              # but 'dx', determined from the refinement level, is not a enough for matching the smoothing length
              # so this is an artificial factor for matching 
    
import_velocity_gas = 'false' # If you want to include the velocity info (w.r.t the galactic center) of gaseous particle
                           # set this parameter to 'true'. 
    
import_dispersion_gas = 'false' # If you want to assume the velocity dispersion inside each geseous particle,
                              # set this parameter  to 'true'. 
sig_c = 5 # Assumed velocity dispersion of gaseous particles
          # Unit: [km/s]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!    Basic Setting for running SKIRT    !!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

rnd_seed = 12345 # seed number for the random number generator 
               # not an important parameter 

mode = 'extinction' # Simulation mode 
                  # Possible options: 'extinction'
                  # transparent & secondary radiation options will be updated soon. 

# Output styles for both the wavelength(or frequency) and flux
wavelengthOutputStyle = 'Wavelength' # Possible options: 'Wavelength', 'Frequency'
fluxOutputStyle = 'Wavelength' # Possible options: 'Wavelength' (F_lambda), 'Frequency' (F_nu), and 'Neutral' (lambda*F_lambda or nu*F_nu)

# Cosmology setting
z_red = 0 # assumed redshift that the source reside 
H_0 = 0.675 # assumed Hubble constant
Om_m = 0.31 # assumed matter density

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@@@@@@@@@@@ setting for the source system @@@@@@@@@@@@# 
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

source_unit = 'micron' # unit of the radiated source SED
wv_source_min = 0.091 # minimum wavelength of the source SED
wv_source_max = 1 # maximum "
wv_source_del = 0.001 # Delta(Wvl) 

#------------------------------------------------------------#
#!!@@ you don't have to care about these parameters @@!!#
source_bias = 0.5 
bias_type = 'log'
bias_unit = 'micron' 
wv_bias_min = 0.091 
wv_bias_max = 1 
source_weight = 1 
bias_deg = 0.5 
#------------------------------------------------------------#

#@@@ old (or overall) population 
smoothing_kernel_o = 'cubic' # smoothing kernel for the particle
                             # Possible options: 'cubic', 'gaussian', 'uniform'
SED_type_o = 'BC03' # choice of SED for the old pop
                    # Possible options: 'BC03', 'Blackbody', 'CastelliKrucz', 'Maraston', 
                    #                   'Starburst99', 'FSPS', 'Bpass', 
                    #                   'LyaGaussian', and LyaDoublePeak

SED_IMF_o = 'Chabrier' # assumed IMF, if chosen SED has an option
SED_res_o = 'High' # only applicable for BC03. low-res. has 1200 wvl pts, and high-res has 6900 pts 

#@@@ young population
smoothing_kernel_y = 'cubic' # same with the old population
SED_type_y = 'mappings' # available option is 'BC03' and 'mappings',
                        # but probably I'll add the rest of the SED families.
SED_IMF_y = 'Chabrier'
SED_res_y = 'High'


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@@@@@@@@@@@ setting for the medium system @@@@@@@@@@@@# 
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

#------------------------------------------------------------#
#!!@@ you don't have to care about these parameters @@!!#
forceScattering = 'true' # With or without forced scattering. 
                         # Forced scattering tends to reduce noise for simulations with low to limited optical depths, 
                         # such as for most dust models on galaxy-wide scales
minWeightReduction = 1e4
minScattEvents = 0
pathLengthBias = 0.5

storeRadiationField = 'false' # storeRadiationField can be important in some cases. 
                            # However, if you don't use secondary emission, dont' mind.
radfield_type = 'log'
radfield_unit = 'micron'
radfield_min_wv = 0.1 # micron
radfield_max_wv = 1000 # micron
radfield_N_wv = 500 
#------------------------------------------------------------#

med_type = 'particle' # For now, 'particle' is only available, 
                    # but I'll add 'AMR' option for sure.

smoothing_kernel_med = 'cubic' # you can check the possible options above.
    
med_Mfrac = 0.4 # Dust-to-Metal ratio

importMetallicity = 'true' # Of course
importTemperature = 'true' # Of course
maxTemperature = 30000 # Assumption: the maximum temperture that dust species can exist 
                       # Unit: [K]
    
importMagneticField = 'false' # sadly, we don't have this info.
                              # But if we want to deal with the radio emission from the bipolar jet, 
                              # maybe we can assume some model for this. 
                              # but not now. 
importVariableMixParams = 'false' # This can be used for NC 


Dust_Type = 'themis' # Dust model for the gaseous medium
                     # Possible options: 'mrn', 'zubko', 'drain_li', 'weingartner_draine', 'trust', 'themis'
N_Si = 15 # number of the size population of each dust comp; this can slow down your calculation speed 
N_C = 15 # number of the size population of each dust comp
N_PAH = 15 # number of the size population of each dust comp

#------------------------------------------------------------#
#!!@@ you don't have to care about these parameters @@!!#
numDensitySamples = 100
numPropertySamples = 1
aggregateVelocity = 'Average'
#------------------------------------------------------------#

#!!! Griding Strategy
# SKIRT try to make spatial grid for calculation.
Tree_Type = 'oct' # OctTree

grid_unit = 'kpc' # The unit of the grid setting 
grid_minX = -40
grid_maxX = +40
grid_minY = -40
grid_maxY = +40
grid_minZ = -40
grid_maxZ = +40


minLevel = 3 # minimum level for griding
maxLevel = 11# maximum level for griding

#------------------------------------------------------------#
#!!@@ you don't have to care about these parameters @@!!#
maxDustFraction = 1e-6 
maxDustOpticalDepth = 0
grid_wv_test_unit = 'micron'
grid_wv_test = 0.55
maxDustDensityDispersion = 0
maxElectronFraction = 1e-6
maxGasFraction = 1e-6
#------------------------------------------------------------#


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@@@@@@@@@@@ setting for the instrument system @@@@@@@@@@@@# 
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

#@@@ default wvlength grid
def_wv_grid = 'pre-defined'
includeGALEX = 'true' # if you want to get 2D GALEX filter image
includeSDSS = 'true' # if you want to get 2D SDSS filter image
include2MASS = 'true' # if you want to get 2D 2MASS filter image
includeWISE = 'false' # if you want to get 2D WISE filter image
includeHERSCHEL = 'false' # if you want to get 2D HERSCHEL filter image


inst_dist_unit = 'Mpc' 
inst_dist = 1000 # the distance between the observer and the object
                 # If your redshift setting is >0 and dist==0, then the distance will be calculated with the given redshift
                 # but if this value is above 0, than the flux will be calculated with this distance

inc_default = 0
azm_default = 0
roll_default = 0

#!!! Settings for the observer's POV 
inc_min = 0 # minimum (observing) inclination 
inc_max = 90 # maximum inclination
inc_del = 10 # delta inclination

azm_min = 0 # minimum (observing) azimuthal angle 
azm_max = 0 # maximum azimuthal angle 
azm_del = 60 # delta azimuthal angle 

recordComponents = 'true' # if you set this value 'true',
                        # the transparent, primarydirect, primaryscattered light will be 
                        # additionally saved in your repo
recordPolarization = 'false' # :(
recordStatistics = 'false' # uhh... maybe you don't have to get this statistics.
numScatteringLevels = 0 # you don't have to care this 

#!!! Field-of-View option 
fov_unit = 'kpc'
fov_X = 60 # FoV for X-direction. 
fov_Y = 60 # FoV for Y-direction. 

pscale_X = 0.05 # pixel scale for X-direction. 
                # unit is same with FoV's 
pscale_Y = 0.05 # pixel scale for Y-direction. 

centre_X = 0 # img's centre position
centre_Y = 0

#!!! If you want to get 2d images with a given wavelength grid,
#!!! not with a pre-defined band-pass filters
inst_2d_sed_on = False 
inst_2d_sed_unit = 'micron' # unit for the wavelength grid
inst_2d_sed_type = 'log' # wavelength grid type
                         # Possible options: 'lin', 'log', 'nestlog', and 'file'
                         # 'file' option only works when you already made your own external wavelength grid file 
        
inst_2d_sed_min_wv = 0.09
inst_2d_sed_max_wv = 1000
inst_2d_sed_N_wv = 100 # number of the wavelength grid points between min & max val.

#+++ only activated when you choose 'nestlog' for inst_2d_sed_type
inst_2d_sed_min_wv_sub = 0.4
inst_2d_sed_max_wv_sub = 0.7
inst_2d_sed_N_wv_sub = 200

#+++ only activated when you choose 'file' for inst_2d_sed_type
inst_2d_sed_repo = './'
inst_2d_sed_fname = 'inst_2d_sed.txt'
inst_2d_sed_relHW = 0

#!!! If you want to get the 1-d SED for each inclination and azimuthal angle.
#!!! (integrated over given aperture size)
save_1d_sed = True
# I hope that you now can understand what these parameters mean ^^
aperture_unit = 'kpc' 
aperture_min = 5
aperture_max = 20 
aper_del = 5

inst_sed_grid_type = 'log' # Possible options: 'lin', 'log', 'nestlog'
inst_sed_grid_unit = 'micron'
inst_sed_min_wv = 0.1
inst_sed_max_wv = 1000
inst_sed_N_wv = 5000

#+++ these parameters only activated when you choose 'nestlog' for the inst_sed_grid_type
inst_sed_min_wv_sub = 0.4
inst_sed_max_wv_sub = 0.7
inst_sed_N_wv_sub = 3001


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
indent = '    ' # don't touch






