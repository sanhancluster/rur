import numpy as np

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def Start(wfile,
          N_phot,
          rnd_seed,
          wavelengthOutputStyle,
          fluxOutputStyle,
          z_red,
          H_0,
          Om_m,
          indent,
          indent_base
          ):
    
    N_idt = indent_base

    print((N_idt)*indent+"<?xml version='1.0' encoding='UTF-8'?>",file=wfile)
    print((N_idt)*indent+"<!-- A SKIRT parameter file © Astronomical Observatory, Ghent University -->",file=wfile)
    print((N_idt)*indent+'<skirt-simulation-hierarchy type="MonteCarloSimulation" format="9" producer="Python toolkit for SKIRT (SkiFile class)" time="2020-05-21T15:04:51">',file=wfile)
    
    print((N_idt+1)*indent+'<MonteCarloSimulation userLevel="Expert" simulationMode="ExtinctionOnly" numPackets="%s">'%(N_phot),file=wfile)
    print((N_idt+2)*indent+'<random type="Random">',file=wfile)
    print((N_idt+3)*indent+'<Random seed="%s"/>'%(rnd_seed),file=wfile)        
    print((N_idt+2)*indent+'</random>',file=wfile)
    print((N_idt+2)*indent+'<units type="Units">',file=wfile)
    print((N_idt+3)*indent+'<ExtragalacticUnits wavelengthOutputStyle="%s" fluxOutputStyle="%s"/>'%(wavelengthOutputStyle,fluxOutputStyle),file=wfile)
    print((N_idt+2)*indent+'</units>',file=wfile)
    print((N_idt+2)*indent+'<cosmology type="Cosmology">',file=wfile)
    print((N_idt+3)*indent+'<FlatUniverseCosmology redshift="%s" reducedHubbleConstant="%s" matterDensityFraction="%s"/>'%(z_red,H_0,Om_m),file=wfile)
    print((N_idt+2)*indent+'</cosmology>',file=wfile)
    
    return N_idt+2


def End(
        wfile,
        indent,
        indent_base
        ):
    
    N_idt = indent_base

    print((N_idt-1)*indent+"</MonteCarloSimulation>",file=wfile)
    print((N_idt-2)*indent+"</skirt-simulation-hierarchy>",file=wfile)
    
    return









#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Smoothing Kernel
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def Smoothing_Kernel(wfile,
                     smoothing_type,
                     indent,
                     indent_base
                     ):

    N_idt = indent_base
    
    print((N_idt)*indent+'<smoothingKernel type="SmoothingKernel">',file=wfile)
    
    #!!! To Be Updated
    if smoothing_type == 'cubic':
        print((N_idt+1)*indent+'<CubicSplineSmoothingKernel/>',file=wfile)
        
    print((N_idt)*indent+'</smoothingKernel>',file=wfile)

    return N_idt

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Smoothing Kernel
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def Wave_Bias(wfile,
              bias_type,
              bias_unit,
              wv_min,
              wv_max,
              indent,
              indent_base
              ):
    
    N_idt = indent_base
    
    print((N_idt)*indent+'<wavelengthBiasDistribution type="WavelengthDistribution">',file=wfile)
    
    if bias_type == 'lin':
        print((N_idt+1)*indent+'<LinWavelengthDistribution minWavelength="%s %s" maxWavelength="%s %s"/>'%(wv_min,bias_unit,wv_max,bias_unit),file=wfile)
    elif bias_type == 'log':
        print((N_idt+1)*indent+'<LogWavelengthDistribution minWavelength="%s %s" maxWavelength="%s %s"/>'%(wv_min,bias_unit,wv_max,bias_unit),file=wfile)
    
    print((N_idt)*indent+'</wavelengthBiasDistribution>',file=wfile)

    return N_idt


