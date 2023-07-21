import numpy as np

from rur.rskirt.core import Base
from rur.rskirt.core import SEDFamily
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def Source_System(wfile,
                  
                  source_unit,
                  wv_source_min,
                  wv_source_max,
                  wv_source_del,
                  source_bias,
                  
                  bias_type,
                  bias_unit,
                  wv_bias_min,
                  wv_bias_max,
                  
                  import_velocity,
                  import_dispersion,
                  source_weight,
                  bias_deg,
                  
                  old_fbase,
                  old_fname,
                  kernel_old,
                  sed_type_old,
                  sed_imf_old,
                  sed_res_old,
                  
                  add_young,
                  young_fbase,
                  young_fname,
                  kernel_young,
                  sed_type_young,
                  sed_imf_young,
                  sed_res_young,

                  indent,
                  indent_base
                  ):
                  
    N_idt = indent_base

    print((N_idt)*indent+'<sourceSystem type="SourceSystem">',file=wfile)
    print((N_idt+1)*indent+'<SourceSystem minWavelength="%s %s" maxWavelength="%s %s" wavelengths="%s %s" sourceBias="%s">'%(wv_source_min,source_unit,wv_source_max,source_unit,wv_source_del,source_unit,source_bias),file=wfile)
    print((N_idt+2)*indent+'<sources type="Source">',file=wfile)
    print((N_idt+3)*indent+'<ParticleSource filename="%s/%s" importVelocity="%s" importVelocityDispersion="%s" useColumns="" sourceWeight="%s" wavelengthBias="%s">'%(old_fbase,old_fname,import_velocity,import_dispersion,source_weight,bias_deg),file=wfile)     

    dummy = Base.Smoothing_Kernel(wfile=wfile,
                                  smoothing_type=kernel_old,
                                  indent=indent,
                                  indent_base=N_idt+4)
    
    if sed_type_old == 'BC03' or sed_type_old == 'bc03' or \
       sed_type_old == 'bC03' or sed_type_old == 'Bc03':
        dummy = SEDFamily.BC03(wfile=wfile,
                                  imf=sed_imf_old,
                                  resolution=sed_res_old,
                                  indent=indent,
                                  indent_base=N_idt+4)
    if sed_type_old == 'Blackbody' or sed_type_old == 'BLACKBODY' or \
       sed_type_old == 'BlackBody' or sed_type_old == 'BB' or sed_type_old == 'blackbody':
        dummy = SEDFamily.Blackbody(wfile=wfile,
                                    indent=indent,
                                    indent_base=N_idt+4)    
    if sed_type_old == 'CastelliKrucz' or sed_type_old == 'castellikrucz' or \
       sed_type_old == 'CASTELLIKRUCZ'or sed_type_old == 'CK' or sed_type_old == 'ck':
        dummy = SEDFamily.CastelliKrucz(wfile=wfile,
                                        indent=indent,
                                        indent_base=N_idt+4)    
    if sed_type_old == 'Maraston' or sed_type_old == 'MARASTON':
        dummy = SEDFamily.Maraston(wfile=wfile,
                                   imf=sed_imf_old,
                                   indent=indent,
                                   indent_base=N_idt+4)    
    if sed_type_old == 'Starburst99' or sed_type_old == 'STARBURST99' or \
       sed_type_old == 'SB99' or sed_type_old == 'sb99' or \
       sed_type_old == 'Starburst' or sed_type_old == 'starburst' or sed_type_old == 'STARBURST':
        dummy = SEDFamily.Starburst99(wfile=wfile,
                                      indent=indent,
                                      indent_base=N_idt+4)    
    if sed_type_old == 'FSPS' or sed_type_old == 'fsps':
        dummy = SEDFamily.FSPS(wfile=wfile,
                               imf=sed_imf_old,
                               indent=indent,
                               indent_base=N_idt+4)    
    if sed_type_old == 'Bpass' or sed_type_old == 'BPASS':
        dummy = SEDFamily.Bpass(wfile=wfile,
                                indent=indent,
                                indent_base=N_idt+4)    
    if sed_type_old == 'LyaGaussian' or sed_type_old == 'LyAGaussian':
        dummy = SEDFamily.LyaGaussian(wfile=wfile,
                                      indent=indent,
                                      indent_base=N_idt+4)    
    if sed_type_old == 'LyaDoublePeak' or sed_type_old == 'LyADoublePeak':
        dummy = SEDFamily.LyaDoublePeak(wfile=wfile,
                                        indent=indent,
                                        indent_base=N_idt+4)    

    dummy = Base.Wave_Bias(wfile=wfile,
                           bias_type=bias_type,
                           bias_unit=bias_unit,
                           wv_min=wv_bias_min,
                           wv_max=wv_bias_max,
                           indent=indent,
                           indent_base=N_idt+4)

    print((N_idt+3)*indent+'</ParticleSource>',file=wfile)
    
    if add_young == True:
        print((N_idt+3)*indent+'<ParticleSource filename="%s/%s" importVelocity="%s" importVelocityDispersion="%s" useColumns="" sourceWeight="%s" wavelengthBias="%s">'%(young_fbase,young_fname,import_velocity,import_dispersion,source_weight,bias_deg),file=wfile)     

        dummy = Base.Smoothing_Kernel(wfile=wfile,
                                      smoothing_type=kernel_young,
                                      indent=indent,
                                      indent_base=N_idt+4)

        if sed_type_young == 'BC03':
            dummy = SEDFamily.BC03(wfile=wfile,
                                      imf=sed_imf_young,
                                      resolution=sed_res_young,
                                      indent=indent,
                                      indent_base=N_idt+4)
        elif sed_type_young == 'mappings' or sed_type_young == 'MAPPINGS':
            dummy = SEDFamily.MappingsIII(wfile=wfile,
                                          indent=indent,
                                          indent_base=N_idt+4)

        dummy = Base.Wave_Bias(wfile=wfile,
                               bias_type=bias_type,
                               bias_unit=bias_unit,
                               wv_min=wv_bias_min,
                               wv_max=wv_bias_max,
                               indent=indent,
                               indent_base=N_idt+4)
        print((N_idt+3)*indent+'</ParticleSource>',file=wfile)
    
    print((N_idt+2)*indent+'</sources>',file=wfile)
    print((N_idt+1)*indent+'</SourceSystem>',file=wfile)
    print((N_idt)*indent+'</sourceSystem>',file=wfile)
    
    return N_idt



