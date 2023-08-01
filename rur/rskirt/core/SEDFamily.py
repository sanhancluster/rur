import numpy as np
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# SED Family
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def BC03(wfile,
         imf,
         resolution,
         indent,
         indent_base
         ):
    
    N_idt = indent_base

    print((N_idt)*indent+'<sedFamily type="SEDFamily">',file=wfile)
    print((N_idt+1)*indent+'<BruzualCharlotSEDFamily imf="%s" resolution="%s"/>'%(imf,resolution),file=wfile)
    print((N_idt)*indent+'</sedFamily>',file=wfile)
    
    return N_idt


def MappingsIII(wfile,
                indent,
                indent_base
                ):
    
    N_idt = indent_base

    print((N_idt)*indent+'<sedFamily type="SEDFamily">',file=wfile)
    print((N_idt+1)*indent+'<MappingsSEDFamily/>',file=wfile)
    print((N_idt)*indent+'</sedFamily>',file=wfile)
    
    return N_idt


def Starburst99(wfile,
                indent,
                indent_base
                ):
    
    N_idt = indent_base

    print((N_idt)*indent+'<sedFamily type="SEDFamily">',file=wfile)
    print((N_idt+1)*indent+'<Starburst99SEDFamily/>',file=wfile)
    print((N_idt)*indent+'</sedFamily>',file=wfile)
    
    return N_idt