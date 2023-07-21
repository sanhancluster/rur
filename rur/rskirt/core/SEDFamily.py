import numpy as np
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# SED Family
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def BlackBody(wfile,
              indent,
              indent_base
              ):
    
    N_idt = indent_base

    print((N_idt)*indent+'<sedFamily type="SEDFamily">',file=wfile)
    print((N_idt+1)*indent+'<BlackBodySEDFamily/>',file=wfile)
    print((N_idt)*indent+'</sedFamily>',file=wfile)
    
    return N_idt


def CastelliKrucz(wfile,
                  indent,
                  indent_base
                  ):

    N_idt = indent_base

    print((N_idt)*indent+'<sedFamily type="SEDFamily">',file=wfile)
    print((N_idt+1)*indent+'<CastelliKuruczSEDFamily/>',file=wfile)
    print((N_idt)*indent+'</sedFamily>',file=wfile)
    
    return N_idt


def Maraston(wfile,
             imf,
             indent,
             indent_base
             ):

    N_idt = indent_base
    
    if imf == 'kroupa':
        imf = 'Kroupa'
    elif imf == 'salpeter':
        imf = 'Salpeter'

    print((N_idt)*indent+'<sedFamily type="SEDFamily">',file=wfile)
    print((N_idt+1)*indent+'<MarastonSEDFamily imf="%s"/>'%(imf),file=wfile)
    print((N_idt)*indent+'</sedFamily>',file=wfile)
    
    return N_idt


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


def Starburst99(wfile,
                indent,
                indent_base
                ):
    
    N_idt = indent_base

    print((N_idt)*indent+'<sedFamily type="SEDFamily">',file=wfile)
    print((N_idt+1)*indent+'<Starburst99SEDFamily/>',file=wfile)
    print((N_idt)*indent+'</sedFamily>',file=wfile)
    
    return N_idt


def FSPS(wfile,
         imf,
         indent,
         indent_base
         ):

    N_idt = indent_base
    
    if imf == 'kroupa':
        imf = 'Kroupa'
    elif imf == 'salpeter':
        imf = 'Salpeter'
    elif imf == 'chabrier':
        imf = 'Chabrier'

    print((N_idt)*indent+'<sedFamily type="SEDFamily">',file=wfile)
    print((N_idt+1)*indent+'<FSPSSEDFamily imf="%s"/>'%(imf),file=wfile)
    print((N_idt)*indent+'</sedFamily>',file=wfile)
    
    return N_idt


def Bpass(wfile,
          indent,
          indent_base
          ):
    
    N_idt = indent_base

    print((N_idt)*indent+'<sedFamily type="SEDFamily">',file=wfile)
    print((N_idt+1)*indent+'<BpassSEDFamily/>',file=wfile)
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


def LyaGaussian(wfile,
                  indent,
                  indent_base
                  ):

    N_idt = indent_base

    print((N_idt)*indent+'<sedFamily type="SEDFamily">',file=wfile)
    print((N_idt+1)*indent+'<LyaGaussianSEDFamily/>',file=wfile)
    print((N_idt)*indent+'</sedFamily>',file=wfile)
    
    return N_idt


def LyaDoublePeak(wfile,
                  indent,
                  indent_base
                  ):

    N_idt = indent_base

    print((N_idt)*indent+'<sedFamily type="SEDFamily">',file=wfile)
    print((N_idt+1)*indent+'<LyaDoublePeakedSEDFamily/>',file=wfile)
    print((N_idt)*indent+'</sedFamily>',file=wfile)
    
    return N_idt





