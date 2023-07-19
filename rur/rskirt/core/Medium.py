import numpy as np

from rur.rskirt.core import Base
from rur.rskirt.core import MaterialMix
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Medium System
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def Medium_System(wfile,
                  
                  forceScattering,
                  minWeightReduction,
                  minScattEvents,
                  pathLengthBias,
                  
                  storeRadiationField,
                  radfield_type,
                  radfield_unit,
                  radfield_min_wv,
                  radfield_max_wv,
                  radfield_N_wv,
                  
                  gas_fbase,
                  gas_fname,
                  
                  med_type,
                  med_Mfrac,
                  importMetallicity,
                  importTemperature,
                  maxTemperature,
                  importVelocity,
                  importMagneticField,
                  importVariableMixParams,
                  smoothing_kernel,
                  
                  Dust_Type,
                  N_Si,
                  N_C,
                  N_PAH,
                  
                  numDensitySamples,
                  numPropertySamples,
                  aggregateVelocity,

                  grid_unit,
                  grid_minX,
                  grid_maxX,
                  grid_minY,
                  grid_maxY,
                  grid_minZ,
                  grid_maxZ,
               
                  Tree_Type,
                  minLevel,
                  maxLevel,
                  maxDustFraction,
                  maxDustOpticalDepth,
                  grid_wv_test_unit,
                  grid_wv_test,
                  maxDustDensityDispersion,
                  maxElectronFraction,
                  maxGasFraction,

                  indent,
                  indent_base
                  ):
    
    N_idt = indent_base
    
    print((N_idt)*indent+'<mediumSystem type="MediumSystem">',file=wfile)
    print((N_idt+1)*indent+'<MediumSystem>',file=wfile)
    print((N_idt+2)*indent+'<photonPacketOptions type="PhotonPacketOptions">',file=wfile)
    print((N_idt+3)*indent+'<PhotonPacketOptions forceScattering="%s" minWeightReduction="%s" minScattEvents="%s" pathLengthBias="%s"/>'%(forceScattering,minWeightReduction,minScattEvents,pathLengthBias),file=wfile)
    print((N_idt+2)*indent+'</photonPacketOptions>',file=wfile)
    print((N_idt+2)*indent+'<radiationFieldOptions type="RadiationFieldOptions">',file=wfile)
    print((N_idt+3)*indent+'<RadiationFieldOptions storeRadiationField="%s">'%(storeRadiationField),file=wfile)
    
    if storeRadiationField == 'true':
        print((N_idt+4)*indent+'<radiationFieldWLG type="DisjointWavelengthGrid">',file=wfile)
        
        if radfield_type =='log':
            print((N_idt+5)*indent+'<LogWavelengthGrid minWavelength="%s %s" maxWavelength="%s %s" numWavelengths="%s"/>'%(radfield_min_wv,radfield_unit,radfield_max_wv,radfield_unit,radfield_N_wv),file=wfile)
        elif radfield_type =='lin':
            print((N_idt+5)*indent+'<LinWavelengthGrid minWavelength="%s %s" maxWavelength="%s %s" numWavelengths="%s"/>'%(radfield_min_wv,radfield_unit,radfield_max_wv,radfield_unit,radfield_N_wv),file=wfile)
        
        print((N_idt+4)*indent+'</radiationFieldWLG>',file=wfile)
            
    print((N_idt+3)*indent+'</RadiationFieldOptions>',file=wfile)
    print((N_idt+2)*indent+'</radiationFieldOptions>',file=wfile)    
    
    print((N_idt+2)*indent+'<media type="Medium">',file=wfile)
    
    if med_type == 'particle':
        print((N_idt+3)*indent+'<ParticleMedium filename="%s/%s" massFraction="%s" importMetallicity="%s" importTemperature="%s" maxTemperature="%s K" importVelocity="%s" importMagneticField="%s" importVariableMixParams="%s" useColumns="">'%(gas_fbase,gas_fname,med_Mfrac,importMetallicity,importTemperature,maxTemperature,importVelocity,importMagneticField,importVariableMixParams),file=wfile)
        
    dummy = Base.Smoothing_Kernel(wfile=wfile,
                                  smoothing_type=smoothing_kernel,
                                  indent=indent,
                                  indent_base=N_idt+4)
    
    dummy = MaterialMix.MatMix(wfile=wfile,
                               Dust_Type=Dust_Type,
                               N_Si=N_Si,
                               N_C=N_C,
                               N_PAH=N_PAH,
                               indent=indent,
                               indent_base=N_idt+4)

    print((N_idt+3)*indent+'</ParticleMedium>',file=wfile)
    print((N_idt+2)*indent+'</media>',file=wfile)

    if med_type == 'particle':
        print((N_idt+2)*indent+'<samplingOptions type="SamplingOptions">',file=wfile)
        print((N_idt+3)*indent+'<SamplingOptions numDensitySamples="%s" numPropertySamples="%s" aggregateVelocity="%s"/>'%(numDensitySamples,numPropertySamples,aggregateVelocity),file=wfile)
        print((N_idt+2)*indent+'</samplingOptions>',file=wfile)
        
        print((N_idt+2)*indent+'<grid type="SpatialGrid">',file=wfile)
        if Tree_Type == 'oct':
            print((N_idt+3)*indent+'<PolicyTreeSpatialGrid minX="%s %s" maxX="%s %s" minY="%s %s" maxY="%s %s" minZ="%s %s" maxZ="%s %s" treeType="OctTree">'%(grid_minX, grid_unit, grid_maxX, grid_unit, grid_minY, grid_unit, grid_maxY, grid_unit, grid_minZ, grid_unit, grid_maxZ, grid_unit),file=wfile)

            print((N_idt+4)*indent+'<policy type="TreePolicy">',file=wfile)
            print((N_idt+5)*indent+'<DensityTreePolicy minLevel="%s" maxLevel="%s" maxDustFraction="%s" maxDustOpticalDepth="%s" wavelength="%s %s" maxDustDensityDispersion="%s" maxElectronFraction="%s" maxGasFraction="%s"/>'%(minLevel,maxLevel,maxDustFraction,maxDustOpticalDepth,grid_wv_test,grid_wv_test_unit,maxDustDensityDispersion,maxElectronFraction,maxGasFraction),file=wfile)
            print((N_idt+4)*indent+'</policy>',file=wfile)
            print((N_idt+3)*indent+'</PolicyTreeSpatialGrid>',file=wfile)
            
            
        print((N_idt+2)*indent+'</grid>',file=wfile)
        
        
    print((N_idt+1)*indent+'</MediumSystem>',file=wfile)
    print((N_idt)*indent+'</mediumSystem>',file=wfile)   
    
    return N_idt
        
        
        
        
        
