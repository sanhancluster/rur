import numpy as np

import os
import subprocess

from rur.rskirt.utils import make_INPUT
from rur.rskirt.utils import make_SKI

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def execute(repo,
            repo_output,
            ski_name,
            skirt_dir,
            N_thread):
    
    os.chdir(repo+repo_output)
    process = subprocess.Popen("%s/skirt -t %s ../%s"%(skirt_dir,N_thread,ski_name),
                               shell=True,
                               universal_newlines=True,
                               stdout=subprocess.PIPE)
    for line in process.stdout:
        print(line, end='')

    return 


def delete_indat(repo,
                 opart_name,
                 ypart_name,
                 gas_name):

    return 



def make_INSKI(
           snap,
           star,
           cell,
           pos_ctr,
           vel_ctr,
           co_moving=False,
           rotating_gal=True,
           R_upp=8,
           R_low=2,
           A_upp=8,
           mass_weight=True,
           par_mode='median',
           dx_s = 50, #[pc]
           sig_s = 10, #[km/s]
           Age_Cut = 0.01, #[Myr]
           fact_dx = 1.5,
           sig_c = 5, #[km/s]

           import_velocity_gas='true',
           import_dispersion_gas='false',
    
           mode='extinction', # 'transparent', 'extinction', 're-rad'
           #!!! Basic Settings
           N_phot=5e7,
           rnd_seed=12345,

           wavelengthOutputStyle='Wavelength',
           fluxOutputStyle='Wavelength',

           #!!! Cosmology
           z_red=0,
           H_0=0.675,
           Om_m=0.31,

           #!!! Source System
           #@@@ general setting 
           source_unit='micron',
           wv_source_min=0.0091, # micron
           wv_source_max=1000, # micron
           wv_source_del=0.01, # micron
           source_bias=0.5, 

           bias_type='log',
           bias_unit='micron',
           wv_bias_min=0.0091,
           wv_bias_max=1000,

           import_velocity='true',
           import_dispersion='false',
           source_weight=1,
           bias_deg=0.5,

           #@@@ old particles
           smoothing_kernel_o='cubic',
           SED_type_o='BC03',
           SED_IMF_o='Chabrier',
           SED_res_o='High',

           #@@@ young particles
           add_young=True,
           smoothing_kernel_y='cubic',
           SED_type_y='mappings',
           SED_IMF_y='Chabrier',
           SED_res_y='High',


           #!!! Medium System
           forceScattering='true',
           minWeightReduction=1e4,
           minScattEvents=0,
           pathLengthBias=0.5,

           storeRadiationField='true',
           radfield_type='log',
           radfield_unit='micron',
           radfield_min_wv=0.1, # micron
           radfield_max_wv=1000, # micron
           radfield_N_wv=500, 

           med_type='particle',
           med_Mfrac=0.4, #directly related to Dust-to-Metal ratio
           importMetallicity='true',
           importTemperature='true',
           maxTemperature=30000, #[K]
           importMagneticField='false',
           importVariableMixParams='false',

           smoothing_kernel_med='cubic', 

           Dust_Type='themis',
           N_Si=15,
           N_C=15,
           N_PAH=15,

           #!!! Sampling options
           numDensitySamples=100,
           numPropertySamples=1,
           aggregateVelocity='Average',

           #!!! Griding Strategy
           grid_unit='kpc',
           grid_minX=-40,
           grid_maxX=+40,
           grid_minY=-40,
           grid_maxY=+40,
           grid_minZ=-40,
           grid_maxZ=+40,

           Tree_Type='oct',

           minLevel=3,
           maxLevel=11,
           maxDustFraction=1e-6,
           maxDustOpticalDepth=0,
           grid_wv_test_unit='micron',
           grid_wv_test=0.55,
           maxDustDensityDispersion=0,
           maxElectronFraction=1e-6,
           maxGasFraction=1e-6,

           #!!! Instruments
           #@@@ default wvlength grid
           def_wv_grid='pre-defined',
           includeGALEX='true',
           includeSDSS='true',
           include2MASS='true',
           includeWISE='false',
           includeHERSCHEL='false',

           #@@@ instrument sys
           #@@@!!! general setting
           inst_dist_unit='Mpc',
           inst_dist=1000,

           inc_default=0,
           azm_default=0,
           roll_default=0,

           inc_min=0,
           inc_max=90,
           inc_del=10,

           azm_min=0,
           azm_max=0,
           azm_del=30,

           recordComponents='true',
           recordPolarization='false',
           recordStatistics='false',
           numScatteringLevels=0,

           #@@@!!! fullInstrument
           fov_unit='kpc',
           fov_X=60,
           fov_Y=60,

           pscale_X=0.05, #kpc
           pscale_Y=0.05,

           centre_X=0,
           centre_Y=0,
    
           inst_2d_sed_on=False,
           inst_2d_sed_unit='micron',
           inst_2d_sed_type = 'log',
           inst_2d_sed_min_wv = 0.09,
           inst_2d_sed_max_wv = 1000,
           inst_2d_sed_N_wv = 100,
    
           inst_2d_sed_repo='./',
           inst_2d_sed_fname='inst_2d_sed.txt',
           inst_2d_sed_relHW=0,

           ### only activated when you choose 'nestedlog' for inst_2d_sed_type
           inst_2d_sed_min_wv_sub = 0.4,
           inst_2d_sed_max_wv_sub = 0.7,
           inst_2d_sed_N_wv_sub = 200,
    
           #@@@!!! SEDInstrument
           save_1d_sed=True,
           aperture_unit='kpc', 
           aperture_min=5, 
           aperture_max=20,
           aper_del=5,

           inst_sed_grid_type='log',
           inst_sed_grid_unit='micron',
           inst_sed_min_wv=0.1,
           inst_sed_max_wv=1000,
           inst_sed_N_wv=5000,
           inst_sed_min_wv_sub = 0.4,
           inst_sed_max_wv_sub = 0.7,
           inst_sed_N_wv_sub = 3001,

           #!!! Basic Info
           repo='/home/jangjk/SKIRT_DIR/rur_dir/testing/',
           repo_output='result/',
           opart_name='part_old.txt',
           ypart_name='part_young.txt',
           gas_name='gas_cell.txt',
           ski_name='nh_ext.ski',
    
           exe_skirt=True,
           del_input=False,
           skirt_dir='/home/jangjk/SKIRT_DIR/SKIRT9/release/SKIRT/main/',
           N_thread=10,

           indent='    '
            ):
    
    os.makedirs(repo+repo_output,exist_ok=True)
    
    make_INPUT.make_input_rur(
                               snap=snap,
                               star=star,
                               cell=cell,
                               pos_ctr=pos_ctr,
                               vel_ctr=vel_ctr,
                               co_moving=co_moving,
                               rotating_gal=rotating_gal,
                               R_upp=R_upp,
                               R_low=R_low,
                               A_upp=A_upp,
                               mass_weight=mass_weight,
                               par_mode=par_mode,
                               dx_s = dx_s, 
                               sig_s = sig_s, 
                               import_velocity=import_velocity,
                               import_dispersion=import_dispersion,
                               old_fbase = repo,
                               old_fname = opart_name,
                               add_young = add_young,
                               Age_Cut = Age_Cut,
                               young_fbase = repo,
                               young_fname = ypart_name,
                               SED_type_y = SED_type_y,
                               gas_fbase = repo,
                               gas_fname = gas_name,
                               fact_dx = fact_dx,
                               sig_c = sig_c,
                               import_velocity_gas=import_velocity_gas,
                               import_dispersion_gas=import_dispersion_gas
                               )

    
    make_SKI.writing_ski(
                   mode=mode,
                   N_phot=N_phot,
                   rnd_seed=rnd_seed,
                   wavelengthOutputStyle=wavelengthOutputStyle,
                   fluxOutputStyle=fluxOutputStyle,
                   z_red=z_red,
                   H_0=H_0,
                   Om_m=Om_m,
                   source_unit=source_unit,
                   wv_source_min=wv_source_min, 
                   wv_source_max=wv_source_max, 
                   wv_source_del=wv_source_del,
                   source_bias=source_bias, 
                   bias_type=bias_type,
                   bias_unit=bias_unit,
                   wv_bias_min=wv_bias_min,
                   wv_bias_max=wv_bias_max,
                   import_velocity=import_velocity,
                   import_dispersion=import_dispersion,
                   source_weight=source_weight,
                   bias_deg=bias_deg,
                   smoothing_kernel_o=smoothing_kernel_o,
                   SED_type_o=SED_type_o,
                   SED_IMF_o=SED_IMF_o,
                   SED_res_o=SED_res_o,
                   add_young=add_young,
                   smoothing_kernel_y=smoothing_kernel_y,
                   SED_type_y=SED_type_y,
                   SED_IMF_y=SED_IMF_y,
                   SED_res_y=SED_res_y,
                   forceScattering=forceScattering,
                   minWeightReduction=minWeightReduction,
                   minScattEvents=minScattEvents,
                   pathLengthBias=pathLengthBias,
                   storeRadiationField=storeRadiationField,
                   radfield_type=radfield_type,
                   radfield_unit=radfield_unit,
                   radfield_min_wv=radfield_min_wv, 
                   radfield_max_wv=radfield_max_wv, 
                   radfield_N_wv=radfield_N_wv, 
                   med_type=med_type,
                   med_Mfrac=med_Mfrac,
                   importMetallicity=importMetallicity,
                   importTemperature=importTemperature,
                   maxTemperature=maxTemperature, 
                   importVelocity_med=import_velocity_gas,
                   importMagneticField=importMagneticField,
                   importVariableMixParams=importVariableMixParams,
                   smoothing_kernel_med=smoothing_kernel_med, 
                   Dust_Type=Dust_Type,
                   N_Si=N_Si,
                   N_C=N_C,
                   N_PAH=N_PAH,
                   numDensitySamples=numDensitySamples,
                   numPropertySamples=numPropertySamples,
                   aggregateVelocity=aggregateVelocity,
                   grid_unit=grid_unit,
                   grid_minX=grid_minX,
                   grid_maxX=grid_maxX,
                   grid_minY=grid_minY,
                   grid_maxY=grid_maxY,
                   grid_minZ=grid_minZ,
                   grid_maxZ=grid_maxZ,
                   Tree_Type=Tree_Type,
                   minLevel=minLevel,
                   maxLevel=maxLevel,
                   maxDustFraction=maxDustFraction,
                   maxDustOpticalDepth=maxDustOpticalDepth,
                   grid_wv_test_unit=grid_wv_test_unit,
                   grid_wv_test=grid_wv_test,
                   maxDustDensityDispersion=maxDustDensityDispersion,
                   maxElectronFraction=maxElectronFraction,
                   maxGasFraction=maxGasFraction,
                   def_wv_grid=def_wv_grid,
                   includeGALEX=includeGALEX,
                   includeSDSS=includeSDSS,
                   include2MASS=include2MASS,
                   includeWISE=includeWISE,
                   includeHERSCHEL=includeHERSCHEL,
                   inst_dist_unit=inst_dist_unit,
                   inst_dist=inst_dist,
                   inc_default=inc_default,
                   azm_default=azm_default,
                   roll_default=roll_default,
                   inc_min=inc_min,
                   inc_max=inc_max,
                   inc_del=inc_del,
                   azm_min=azm_min,
                   azm_max=azm_max,
                   azm_del=azm_del,
                   recordComponents=recordComponents,
                   recordPolarization=recordPolarization,
                   recordStatistics=recordStatistics,
                   numScatteringLevels=numScatteringLevels,
                   fov_unit=fov_unit,
                   fov_X=fov_X,
                   fov_Y=fov_Y,
                   pscale_X=pscale_X, 
                   pscale_Y=pscale_Y,
                   centre_X=centre_X,
                   centre_Y=centre_Y,
                   inst_2d_sed_on=inst_2d_sed_on,
                   inst_2d_sed_unit=inst_2d_sed_unit,
                   inst_2d_sed_type = inst_2d_sed_type,
                   inst_2d_sed_min_wv = inst_2d_sed_min_wv,
                   inst_2d_sed_max_wv = inst_2d_sed_max_wv,
                   inst_2d_sed_N_wv = inst_2d_sed_N_wv,
                   inst_2d_sed_min_wv_sub = inst_2d_sed_min_wv_sub,
                   inst_2d_sed_max_wv_sub = inst_2d_sed_max_wv_sub,
                   inst_2d_sed_N_wv_sub = inst_2d_sed_N_wv_sub,
                   inst_2d_sed_repo=inst_2d_sed_repo,
                   inst_2d_sed_fname=inst_2d_sed_fname,
                   inst_2d_sed_relHW=inst_2d_sed_relHW,
                   save_1d_sed=save_1d_sed,
                   aperture_unit=aperture_unit, 
                   aperture_min=aperture_min, 
                   aperture_max=aperture_max,
                   aper_del=aper_del,
                   inst_sed_grid_type=inst_sed_grid_type,
                   inst_sed_grid_unit=inst_sed_grid_unit,
                   inst_sed_min_wv=inst_sed_min_wv,
                   inst_sed_max_wv=inst_sed_max_wv,
                   inst_sed_N_wv=inst_sed_N_wv,
                   inst_sed_min_wv_sub=inst_sed_min_wv_sub,
                   inst_sed_max_wv_sub=inst_sed_max_wv_sub,
                   inst_sed_N_wv_sub=inst_sed_N_wv_sub,
                   repo=repo,
                   opart_name=opart_name,
                   ypart_name=ypart_name,
                   gas_name=gas_name,
                   ski_name=ski_name,
                   indent=indent
               )

    
    if exe_skirt == True:
        execute(repo=repo,
                repo_output=repo_output,
                ski_name=ski_name,
                skirt_dir=skirt_dir,
                N_thread=N_thread)
    
    return 

















