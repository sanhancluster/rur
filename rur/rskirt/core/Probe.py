import numpy as np
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def Probe_System(
                wfile,
                indent,
                indent_base
                ):
    
    #!!! Probe system will be updated soon
    
    N_idt = indent_base
    
    print((N_idt)*indent+'<probeSystem type="ProbeSystem">',file=wfile)
    print((N_idt+1)*indent+'<ProbeSystem/>',file=wfile)
    print((N_idt)*indent+'</probeSystem>',file=wfile)
    
    return N_idt
    
    