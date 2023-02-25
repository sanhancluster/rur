import numpy as np
import h5py
import os
import os.path
import pickle5 as pickle
import pkg_resources

from scipy import interpolate
from scipy.io import FortranFile
import scipy.integrate as integrate

from rur.vr.fortran.find_domain_py import find_domain_py
from rur.vr.fortran.get_ptcl_py import get_ptcl_py
from rur.vr.fortran.get_flux_py import get_flux_py
from rur.vr.fortran.jsamr2cell_totnum_py import jsamr2cell_totnum_py
from rur.vr.fortran.jsamr2cell_py import jsamr2cell_py
from rur.vr.fortran.js_gasmap_py import js_gasmap_py

class vr_load:

    def __init__(self, simtype, cname=False, num_thread=1):
        ##-----
        ## General settings
        ##-----
        self.num_thread = int(num_thread)   # The number of cores
        self.simtype    = simtype
        self.dir_table  = pkg_resources.resource_filename('rur', 'vr/table/')
        self.ssp_type   = 'chab'

        ##-----
        ## Specify configurations based on the simulation type
        ##-----
        if(simtype == 'NH'):
            # Path related
            self.dir_raw        = '/storage6/NewHorizon/snapshots/'
            self.dir_catalog    = '/storage5/NewHorizon/VELOCIraptor/'
            self.dir_sink       = '/storage6/NewHorizon/snapshots/sinkdum/'

            # Ramses related
            self.rtype_llint    = False     # Whether particle IDs are 64 byte integer
            self.rtype_family   = False     # Whether part_out contains family
            self.rtype_neff     = int(4096) # Effective resolution of the zoom region

            # VR output related
            self.vr_columnlist  = ['ID', 'ID_mbp', 'hostHaloID', 'numSubStruct', 'Structuretype', 'Mvir', 'Mass_tot', 'Mass_FOF',
                       'Mass_200mean', 'Efrac', 'Mass_200crit', 'Rvir', 'R_size', 'R_200mean', 'R_200crit',
                       'R_HalfMass', 'R_HalfMass_200mean', 'R_HalfMass_200crit', 'Rmax', 'Xc', 'Yc', 'Zc', 'VXc',
                       'VYc', 'VZc', 'Lx', 'Ly', 'Lz', 'sigV', 'Vmax', 'npart']
                                                                # Catalog output
            self.vr_galprop     = ['SFR', 'ABmag']              # Bulk properties computed in the post-processing
            self.vr_fluxlist    = ['u', 'g', 'r', 'i', 'z']     # flux list of Abmag
            self.vr_fluxzp      = np.double(np.array([895.5*1e-11, 466.9*1e-11, 278.0*1e-11, 185.2*1e-11, 131.5*1e-11]))
                                                                # flux zero points
        elif(simtype == 'NH2'):
            # Path related
            self.dir_raw        = '/storage7/NH2/snapshots/'
            self.dir_catalog    = '/storage7/NH2/VELOCIraptor/'
            self.dir_sink       = '/storage7/NH2/SINKPROPS/'

            # Ramses related
            self.rtype_llint    = False     # Whether particle IDs are 64 byte integer
            self.rtype_family   = True     # Whether part_out contains family
            self.rtype_neff     = int(4096) # Effective resolution of the zoom region

            # VR output related
            self.vr_columnlist  = ['ID', 'ID_mbp', 'hostHaloID', 'numSubStruct', 'Structuretype', 'Mvir', 'Mass_tot', 'Mass_FOF',
                       'Mass_200mean', 'Efrac', 'Mass_200crit', 'Rvir', 'R_size', 'R_200mean', 'R_200crit',
                       'R_HalfMass', 'R_HalfMass_200mean', 'R_HalfMass_200crit', 'Rmax', 'Xc', 'Yc', 'Zc', 'VXc',
                       'VYc', 'VZc', 'Lx', 'Ly', 'Lz', 'sigV', 'Vmax', 'npart']
                                                                # Catalog output
            self.vr_galprop     = ['SFR', 'ABmag']              # Bulk properties computed in the post-processing
            self.vr_fluxlist    = ['u', 'g', 'r', 'i', 'z']     # flux list of Abmag
            self.vr_fluxzp      = np.double(np.array([895.5*1e-11, 466.9*1e-11, 278.0*1e-11, 185.2*1e-11, 131.5*1e-11]))
                                                                # flux zero points

        elif(simtype == 'FORNAX' or simtype == 'FN'):
            # Path related
            self.dir_raw        = '/storage5/FORNAX/KISTI_OUTPUT/l10006/'
            self.dir_catalog    = '/storage5/FORNAX/VELOCIraptor/l10006/'
            self.dir_sink       = '/storage5/FORNAX/KISTI_OUTPUT/l10006/SINKPROPS/'

            # Ramses related
            self.rtype_llint    = False     # Whether particle IDs are 64 byte integer
            self.rtype_family   = True     # Whether part_out contains family
            self.rtype_neff     = int(2048) # Effective resolution of the zoom region

            # VR output related
            self.vr_columnlist  = ['ID', 'ID_mbp', 'hostHaloID', 'numSubStruct', 'Structuretype', 'Mvir', 'Mass_tot', 'Mass_FOF',
                       'Mass_200mean', 'Efrac', 'Mass_200crit', 'Rvir', 'R_size', 'R_200mean', 'R_200crit',
                       'R_HalfMass', 'R_HalfMass_200mean', 'R_HalfMass_200crit', 'Rmax', 'Xc', 'Yc', 'Zc', 'VXc',
                       'VYc', 'VZc', 'Lx', 'Ly', 'Lz', 'sigV', 'Vmax', 'npart']
                                                                # Catalog output
            self.vr_galprop     = ['SFR', 'ABmag']              # Bulk properties computed in the post-processing
            self.vr_fluxlist    = ['u', 'g', 'r', 'i', 'z']     # flux list of Abmag
            self.vr_fluxzp      = np.double(np.array([895.5*1e-11, 466.9*1e-11, 278.0*1e-11, 185.2*1e-11, 131.5*1e-11]))
                                                                # flux zero points
        elif(simtype == 'YZiCS'):
            if(cname == False):
                print('%-----')
                print(' YZiCS load requires the name of cluster (ex: cname=01605)')
                print('     Default cluster has been chosen (29172)')
                print('%-----')
                cname   = 29172

            cname_str   = '%0.5d'%cname
            # Path related
            self.dir_raw        = '/storage3/Clusters/%0.5d'%cname + '/snapshots/' 
            self.dir_catalog    = '/storage3/Clusters/VELOCIraptor/c%0.5d'%cname + '/' 

            # Ramses related
            self.rtype_llint    = False     # Whether particle IDs are 64 byte integer
            self.rtype_family   = False     # Whether part_out contains family
            self.rtype_neff     = int(2048) # Effective resolution of the zoom region

            # VR output related
            self.vr_columnlist  = ['ID', 'ID_mbp', 'hostHaloID', 'numSubStruct', 'Structuretype', 'Mvir', 'Mass_tot', 'Mass_FOF',
                       'Mass_200mean', 'Efrac', 'Mass_200crit', 'Rvir', 'R_size', 'R_200mean', 'R_200crit',
                       'R_HalfMass', 'R_HalfMass_200mean', 'R_HalfMass_200crit', 'Rmax', 'Xc', 'Yc', 'Zc', 'VXc',
                       'VYc', 'VZc', 'Lx', 'Ly', 'Lz', 'sigV', 'Vmax', 'npart']
                                                                # Catalog output
            self.vr_galprop     = ['SFR', 'ABmag']              # Bulk properties computed in the post-processing
            self.vr_fluxlist    = ['u', 'g', 'r', 'i', 'z']     # flux list of Abmag
            self.vr_fluxzp      = np.double(np.array([895.5*1e-11, 466.9*1e-11, 278.0*1e-11, 185.2*1e-11, 131.5*1e-11]))
                                                                # flux zero points
        else:
            print('%-----')
            print(' Wrong argument for the simtype')
            print('     Simtype list: NH, NH2, FORNAX(or FN), NC(not yet)')
            print('%-----')

        ##----- Simulation Parameter load
        infofile    = self.dir_raw + 'output_00001/info_00001.txt'

        self.H0             = np.double(np.loadtxt(infofile, dtype=object, skiprows=10, max_rows=1)[2])
        self.omega_M        = np.double(np.loadtxt(infofile, dtype=object, skiprows=11, max_rows=1)[2])
        self.omega_L        = np.double(np.loadtxt(infofile, dtype=object, skiprows=12, max_rows=1)[2])
        self.rtype_ndomain  = np.int32( np.loadtxt(infofile, dtype=object, skiprows=0, max_rows=1)[2] )
        self.rtype_ndim     = np.int32( np.loadtxt(infofile, dtype=object, skiprows=1, max_rows=1)[2] )
        self.rtype_levmin         = np.int32( np.loadtxt(infofile, dtype=object, skiprows=2, max_rows=1)[2] )
        self.rtype_levmax         = np.int32( np.loadtxt(infofile, dtype=object, skiprows=3, max_rows=1)[2] )


    ##-----
    ## Load Galaxy
    ##  To do list
    ##      1) do not use imglist.txt
    ##      2) More efficienct way of reading multiple hdf5 files?
    ##-----
    def f_rdgal(self, n_snap, id0, horg='g'):

        # Path setting
        directory   = self.dir_catalog
        if(horg=='h'): directory += 'Halo/VR_Halo/snap_%0.4d'%n_snap + '/'
        elif(horg=='g'): directory += 'Galaxy/VR_Galaxy/snap_%0.4d'%n_snap+'/'
        else:
            print('%-----')
            print(' Wrong argument for the horg')
            print('     horg = "g" (for galaxy) or "h" (for halo)')
            print('%-----')

        # Get file list
        if(id0>=0): flist=[directory + 'GAL_%0.6d'%id0 + '.hdf5']
        else:
            flist=os.system('ls '+directory+'GAL_*.hdf5 > imglist.txt')
                              #flist=os.system('find -type f -name "'+proj_images_dir+'*.pickle" -print0 | xargs -0 -n 10 ls > '+proj_images_dir+'imglist.dat')
            flist=np.loadtxt("imglist.txt", dtype=str)
            ## find a more simple way

        # Set column list
        dtype=[]
        for name in self.vr_columnlist:
            if(name=='SFR' and horg=='g'): dtype=dtype+[(name, 'object')]
            elif(name=='ABmag' and horg=='g'): dtype=dtype+[(name, 'object')]
            elif(name=='ID' or name=='hostHaloID' or name=='numSubStruct' or name=='Structuretype' or name=='npart'): dtype=dtype+[(name, np.int32)]
            elif(name=='ID_mbp'): dtype=dtype+[(name, np.int64)]
            else: dtype=dtype+[(name, '<f8')]

        if(horg=='g'):
            column_list_additional=['Domain_List', 'Flux_List', 'MAG_R', 'SFR_R', 'SFR_T', 'ConFrac', 'CONF_R',
                                   'isclump', 'rate', 'Aexp', 'snapnum']
            for name in self.vr_galprop:
                column_list_additional=column_list_additional+[name]
        else:
            column_list_additional=['']###['Domain_List', 'ConFrac', 'CONF_R', 'rate', 'Aexp', 'snapnum']


        if(horg=='g'):
            for name in column_list_additional:
                if(name=='isclump'): dtype=dtype+[(name, np.int32)]
                elif(name=='rate'): dtype=dtype+[(name, '<f8')]
                elif(name=='Aexp'): dtype=dtype+[(name, '<f8')]
                elif(name=='snapnum'): dtype=dtype+[(name, np.int32)]
                else: dtype=dtype+[(name, 'object')]

        galdata=np.zeros(len(flist), dtype=dtype)

        for i, fn in enumerate(flist):
            dat= h5py.File(fn, 'r')
            for name in self.vr_columnlist:
                if(horg=='g'):
                    xdata=dat.get("G_Prop/G_"+name)
                    galdata[name][i]=np.array(xdata)
                else:
                    if(name!='SFR' and name!='ABmag'):
                        xdata=dat.get("G_Prop/G_"+name)
                        galdata[name][i]=np.array(xdata)

            if(horg=='g'):
                for name in column_list_additional:
                    if(name=='ConFrac'): xdata=dat.get("/G_Prop/G_ConFrac")
                    elif(name=='CONF_R'): xdata=dat.get("/CONF_R")
                    elif(name=='isclump'): xdata=dat.get("/isclump")
                    elif(name=='rate'): xdata=dat.get("/rate")
                    elif(name=='Aexp'): xdata=dat.get("/Aexp")
                    elif(name=='Domain_List'): xdata=dat.get("/Domain_List")
                    elif(name=='Flux_List'): xdata=np.array(dat.get("/Flux_List"),dtype=np.str)
                    elif(name=='MAG_R'): xdata=dat.get("/MAG_R")
                    elif(name=='SFR_R'): xdata=dat.get("/SFR_R")
                    elif(name=='SFR_T'): xdata=dat.get("/SFR_T")
                    elif(name=='snapnum'): xdata=np.int32(n_snap)
                    elif(name=='SFR' or name=='ABmag'): xdata=dat.get("G_Prop/G_" + name)
                    ##    break
                    ##else: xdata=dat.get(name)
                    galdata[name][i]=np.array(xdata)
        return galdata

    ##-----
    ## Load Particle of a galaxy
    ##  To do list
    ##      *) Halo member load is not implemented
    ##-----
    def f_rdptcl(self, n_snap, id0, horg='g', p_gyr=False, p_sfactor=False, p_mass=True, p_flux=False,
            p_metal=False, p_id=False, raw=False, boxrange=50., domlist=[0], num_thread=None, sink=False):

        # Get funtions
        gf  = vr_getftns(self)

        # Initial settings
        if(p_gyr==False and p_flux==True): p_gyr=True
        if(num_thread==None): num_thread=self.num_thread

        unit_l  = np.double(np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=15, max_rows=1)[2])
        unit_t  = np.double(np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=17, max_rows=1)[2])
        kms     = np.double(unit_l / unit_t / 1e5)
        unit_d  = np.double(np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=16, max_rows=1)[2])
        unit_m  = unit_d * unit_l**3
        levmax  = np.int32( np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=3, max_rows=1)[2])
        hindex  = np.double(np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=21)[:,1:])
        omega_M = np.double(np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=11, max_rows=1)[2])
        omega_B = np.double(np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=14, max_rows=1)[2])

        dmp_mass    = 1.0/(self.rtype_neff*self.rtype_neff*self.rtype_neff)*(omega_M - omega_B)/omega_M

        if(sink==True):
            raw =True
            p_id=True

        # READ PTCL ID & Domain List (Might be skipped when raw==True)
        if(raw==False):
            if(horg=='h'): fname = self.dir_catalog + 'Halo/VR_Halo/snap_%0.4d'%n_snap+"/"
            elif(horg=='g'): fname = self.dir_catalog + 'Galaxy/VR_Galaxy/snap_%0.4d'%n_snap+"/"
            fname   += 'GAL_%0.6d'%id0+'.hdf5'

            dat     = h5py.File(fname, 'r')
            idlist  = np.array(dat.get("P_Prop/P_ID"))
            domlist = np.array(dat.get("Domain_List"))
        else:
            idlist  = np.zeros(1, dtype=np.int64)
            domlist = np.zeros(self.rtype_ndomain, dtype=np.int32) - 1

            #----- Find Domain
            galtmp  = self.f_rdgal(n_snap, id0, horg=horg)

            xc  = galtmp['Xc']/unit_l * 3.086e21
            yc  = galtmp['Yc']/unit_l * 3.086e21
            zc  = galtmp['Zc']/unit_l * 3.086e21
            rr  = galtmp['R_HalfMass']/unit_l * 3.086e21
            larr    = np.zeros(20, dtype=np.int32)
            darr    = np.zeros(20, dtype='<f8')

            larr[0] = np.int32(len(xc))
            larr[1] = np.int32(len(domlist))
            larr[2] = np.int32(num_thread)
            larr[3] = np.int32(levmax)

            darr[0] = 50.
            if(boxrange!=None): darr[0] = boxrange / (rr * unit_l / 3.086e21)

            find_domain_py.find_domain(xc, yc, zc, rr, hindex, larr, darr)
            domlist     = find_domain_py.dom_list
            domlist     = domlist[0][:]

        domlist = np.int32(np.array(np.where(domlist > 0))[0] + 1)
        idlist  = np.int64(idlist)

        #----- LOAD PARTICLE INFO
        if(raw==False):
            larr    = np.zeros(20, dtype=np.int32)
            darr    = np.zeros(20, dtype='<f8')

            larr[0]     = np.int32(len(idlist))
            larr[1]     = np.int32(len(domlist))
            larr[2]     = np.int32(n_snap)
            larr[3]     = np.int32(num_thread)
            larr[10]    = np.int32(len(self.dir_raw))
            larr[17]    = 0

            if(horg=='g'): larr[11] = 1 # STAR
            else: larr[11] = 2 # DM

            if(self.rtype_family==True): larr[18] = 100
            else: larr[18] = 0
            if(self.rtype_llint==True): larr[19] = 100
            else: larr[19] = 0

            if(horg=='h'): darr[11] = dmp_mass

            get_ptcl_py.get_ptcl(self.dir_raw, idlist, domlist, larr, darr)
            pinfo   = get_ptcl_py.ptcl
        else:
            larr    = np.zeros(20,dtype=np.int32)
            darr    = np.zeros(20,dtype='<f8')

            larr[1] = np.int32(len(domlist))
            larr[2] = np.int32(n_snap)
            larr[3] = np.int32(num_thread)
            larr[10]= np.int32(len(self.dir_raw))
            larr[17]= 100

            if(horg=='g'): larr[11] = 1 # STAR
            else: larr[11] = 2 # DM
            if(sink==True): larr[11] = 3 # SINK
            
            if(self.rtype_family==True): larr[18] = 100
            else: larr[18] = 0
            if(self.rtype_llint==True): larr[19] = 100
            else: larr[19] = 0

            if(horg=='h'): darr[11] = dmp_mass

            get_ptcl_py.get_ptcl(self.dir_raw, idlist, domlist, larr, darr)
            pinfo   = get_ptcl_py.ptcl
            idlist  = get_ptcl_py.id_raw

        #----- EXTRACT
        n_old       = len(pinfo)*1.

        pinfo       = pinfo[np.where(pinfo[:,0]>-1e7)]
        n_new       = len(pinfo)*1.
        rate        = n_new / n_old

        pinfo[:,0]    *= unit_l / 3.086e21
        pinfo[:,1]    *= unit_l / 3.086e21
        pinfo[:,2]    *= unit_l / 3.086e21
        pinfo[:,3]    *= kms
        pinfo[:,4]    *= kms
        pinfo[:,5]    *= kms
        pinfo[:,6]    *= unit_m / 1.98892e33

        #----- OUTPUT ARRAY
        dtype   = [('xx', '<f8'), ('yy', '<f8'), ('zz', '<f8'),
            ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
            ('mass', '<f8'), ('sfact', '<f8'), ('gyr', '<f8'), ('metal', '<f8')]
        for name in self.vr_fluxlist:
            dtype   += [('f_' + name, '<f8')]

        ptcl    = np.zeros(np.int32(n_new), dtype=dtype)

        ptcl['xx'][:]   = pinfo[:,0]
        ptcl['yy'][:]   = pinfo[:,1]
        ptcl['zz'][:]   = pinfo[:,2]

        ptcl['vx'][:]   = pinfo[:,3]
        ptcl['vy'][:]   = pinfo[:,4]
        ptcl['vz'][:]   = pinfo[:,5]

        ptcl['mass'][:] = pinfo[:,6]
        ptcl['metal'][:]= pinfo[:,8]

        ##----- COMPUTE GYR
        if(p_gyr==True):
            gyr = gf.g_gyr(n_snap, pinfo[:,7])
            ptcl['gyr'][:]  = gyr['gyr'][:]
            ptcl['sfact'][:]= gyr['sfact'][:]

        ##----- COMPUTE FLUX
        if(p_flux==True):
            for name in self.vr_fluxlist:
                ptcl['f_' + name][:] = gf.g_flux(ptcl['mass'][:], ptcl['metal'][:], ptcl['gyr'][:],name)[name]

        return ptcl, rate, domlist, idlist


    ##-----
    ## LOAD CELL DATA around galaxies
    ##-----
    def f_rdamr(self, n_snap, id0, boxrange=None, raw=False, raw_xc=None, raw_yc=None, raw_zc=None):

        ##----- Settings
        if(boxrange==None): boxrange = 50.

        ##----- Box Size Settings
        galtmp  = self.f_rdgal(n_snap, id0, horg='g')
        xr  = np.array([-1, 1.],dtype='<f8') * boxrange + galtmp['Xc']
        yr  = np.array([-1, 1.],dtype='<f8') * boxrange + galtmp['Yc']
        zr  = np.array([-1, 1.],dtype='<f8') * boxrange + galtmp['Zc']

        ##------ Unit Load
        infofile    = self.dir_raw + 'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt"
        unit_l  = np.double(np.loadtxt(infofile, dtype=object, skiprows=15, max_rows=1)[2])
        unit_t  = np.double(np.loadtxt(infofile, dtype=object, skiprows=17, max_rows=1)[2])
        unit_d  = np.double(np.loadtxt(infofile, dtype=object, skiprows=16, max_rows=1)[2])
        hindex  = np.double(np.loadtxt(infofile, dtype=object, skiprows=21)[:,1:])
        kms     = np.double(unit_l / unit_t / 1e5)
        unit_m  = unit_d * unit_l**3

        unit_T2 = np.double(1.6600000e-24) / np.double(1.3806200e-16) * np.double(unit_l / unit_t)**2
        nH  = np.double(0.76) / np.double(1.6600000e-24) * unit_d

        ##----- Find domain
        domlist = np.zeros(self.rtype_ndomain, dtype=np.int32) - 1
        xc  = galtmp['Xc']/unit_l * 3.086e21
        yc  = galtmp['Yc']/unit_l * 3.086e21
        zc  = galtmp['Zc']/unit_l * 3.086e21

        #----- Reset Center when loading cells around the arbitrary center
        if(raw==True):
            if(raw_xc!=None):xc = np.array(raw_xc/unit_l * 3.086e21)
            if(raw_yc!=None):yc = np.array(raw_yc/unit_l * 3.086e21)
            if(raw_zc!=None):zc = np.array(raw_zc/unit_l * 3.086e21)

        rr  = galtmp['R_HalfMass']/unit_l * 3.086e21
        larr    = np.zeros(20, dtype=np.int32)
        darr    = np.zeros(20, dtype='<f8')

        larr[0] = np.int32(len(xc))
        larr[1] = np.int32(len(domlist))
        larr[2] = np.int32(self.num_thread)
        larr[3] = np.int32(self.rtype_levmax)

        darr[0] = 50.
        if(boxrange!=None): darr[0] = boxrange / (rr * unit_l / 3.086e21)

        find_domain_py.find_domain(xc, yc, zc, rr, hindex, larr, darr)
        domlist     = find_domain_py.dom_list
        domlist     = domlist[0][:]
        domlist = np.int32(np.array(np.where(domlist > 0))[0] + 1)

        #----- READ AMR (Get Total number of leaf cells)
        larr    = np.zeros(20, dtype=np.int32)
        darr    = np.zeros(20, dtype='<f8')

        file_a  = self.dir_raw + 'output_%0.5d'%n_snap + '/amr_%0.5d'%n_snap + '.out'
        file_h  = self.dir_raw + 'output_%0.5d'%n_snap + '/hydro_%0.5d'%n_snap + '.out'
        file_i  = self.dir_raw + 'output_%0.5d'%n_snap + '/info_%0.5d'%n_snap + '.txt'

        larr[0] = np.int32(len(domlist))
        larr[2] = np.int32(1)#np.int32(num_thread)
        larr[3] = np.int32(len(file_a))
        larr[4] = np.int32(len(file_h))
        larr[5] = np.int32(self.rtype_ndomain)
        larr[6] = np.int32(self.rtype_ndim)
        larr[7] = np.int32(self.rtype_levmin)
        larr[8] = np.int32(self.rtype_levmax)

        jsamr2cell_totnum_py.jsamr2cell_totnum(larr, darr, file_a, file_h, domlist)
        ntot    = jsamr2cell_totnum_py.ntot
        nvarh   = jsamr2cell_totnum_py.nvarh
        mg_ind  = jsamr2cell_totnum_py.mg_ind

        #----- READ AMR (ALLOCATE)
        data    = np.zeros(ntot, dtype=[('xx','<f8'), ('yy','<f8'), ('zz','<f8'),
            ('vx','<f8'), ('vy','<f8'), ('vz','<f8'), ('dx','<f8'), ('mass', '<f8'),
            ('den','<f8'), ('temp','<f8'), ('P_thermal','<f8'), ('metal','<f8'), ('level','int32')])

        ##----- READ AMR
        larr    = np.zeros(20, dtype=np.int32)
        darr    = np.zeros(20, dtype='<f8')

        larr[0] = np.int32(len(domlist))
        larr[2] = np.int32(self.num_thread)
        larr[3] = np.int32(len(file_a))
        larr[4] = np.int32(len(file_h))
        larr[5] = np.int32(len(file_i))
        larr[6] = np.int32(self.rtype_ndomain)
        larr[7] = np.int32(self.rtype_ndim)
        larr[8] = np.int32(self.rtype_levmin)
        larr[9] = np.int32(self.rtype_levmax)
        larr[10]= np.int32(ntot)
        larr[11]= np.int32(nvarh)

        jsamr2cell_py.jsamr2cell(larr, darr, file_a, file_h, file_i, mg_ind, domlist)
        xgdum   = np.array(jsamr2cell_py.mesh_xg,dtype='<f8')
        hvdum   = np.array(jsamr2cell_py.mesh_hd,dtype='<f8')
        dxdum   = np.array(jsamr2cell_py.mesh_dx,dtype='<f8')
        lvdum   = np.array(jsamr2cell_py.mesh_lv,dtype='int32')

        data['xx'][:]   = xgdum[:,0] * unit_l / np.double(3.086e21)
        data['yy'][:]   = xgdum[:,1] * unit_l / np.double(3.086e21)
        data['zz'][:]   = xgdum[:,2] * unit_l / np.double(3.086e21)
        data['vx'][:]   = hvdum[:,1] * kms
        data['vy'][:]   = hvdum[:,2] * kms
        data['vz'][:]   = hvdum[:,3] * kms
        data['den'][:]  = hvdum[:,0]
        data['temp'][:] = hvdum[:,4]
        data['metal'][:]= hvdum[:,5]
        data['dx'][:]   = dxdum[:] * unit_l / np.double(3.086e21)
        data['level'][:]= lvdum[:]
        #data['mass'][:] = np.double(10.**(np.log10(hvdum[:,0]) + np.log10(unit_d) + np.double(3.0) * (np.log10(dxdum[:]) + np.log10(unit_l)) - np.log10(1.98892e33)))
        data['mass'][:] = hvdum[:,0] *unit_d * (dxdum[:] * unit_l)**3 / np.double(1.98892e33)

        data    = data[np.where(lvdum >= 0)]
        dumA = data['temp'][:]
        dumB = data['den'][:]

        data['temp'][:] = data['temp'][:] / data['den'][:] * unit_T2    # [K/mu]
        data['den'][:]  *= nH                                           # [atom/cc]
        data['P_thermal'][:]    = dumA * unit_m / unit_l / unit_t**2 / np.double(1.3806200e-16)

        units   = {'unit_l':unit_l, 'unit_m':unit_m, 'unit_t':unit_t, 'kms':kms, 'unit_nH':nH, 'unit_T2':unit_T2}
        return data, boxrange, units

##-----
## Get Tree Data
##-----
    def f_gettree_readdat(self, filename):
        with open(filename,'rb') as f:
            longtype    = np.dtype(np.int32)
            bdata       = np.fromfile(f,dtype=longtype)

            ## READ
            n_branch    = bdata[0]
            b_startind  = np.zeros(n_branch, dtype='int32')

            ind0    = np.int32(1)
            ind = np.array(range(n_branch),dtype='int32')
            for i in ind:
                #if(ind0>59775170):print(i)
                b_startind[i]   = ind0
                ind0    += np.int32(bdata[ind0] * 2 + 1)
                ind0    += np.int32(bdata[ind0] * 4 + 1)


            tree_key    = bdata[ind0+1:-1]
            return bdata, b_startind, tree_key
    def f_gettree(self, n_snap, id0, horg='g'):


        directory   = self.dir_catalog
        ## Initialize
        if(horg=='g'):
            dir_tree    = directory + 'Galaxy/tree/'
        elif(horg=='h'):
            dir_tree    = directory + 'Halo/tree/'

        ## Is pickle?
        fname   = dir_tree + 'ctree.pkl'
        isfile = os.path.isfile(fname)

        if(isfile==True):
            with open(fname, 'rb') as f:
                data = pickle.load(f)
        else:
            fname_bin   = dir_tree + 'ctree.dat'
            data    = self.f_gettree_readdat(fname_bin)
            with open(fname, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        branch  = data[0]
        bind    = data[1]
        key     = data[2]

        keyval  = n_snap + key[0]*id0
        kind    = key[keyval]
        ind0    = bind[kind]

        n_link  = branch[ind0]

        idlist  = branch[ind0+1:ind0+n_link+1]
        snlist  = branch[ind0+n_link+1:ind0+n_link*2+1]

        ind1    = ind0 + n_link*2+1
        n_prog  = branch[ind1]

        m_idlist    = np.zeros(1, dtype=np.int32) - 1
        m_snaplist  = np.zeros(1, dtype=np.int32) - 1
        m_merit     = np.zeros(1, dtype='<f8') - 1.
        m_bid       = np.zeros(1, dtype=np.int32) - 1
        if(n_prog>0):
            m_idlist    = branch[ind1+1:ind1+n_prog+1]
            m_snaplist  = branch[ind1+n_prog+1:ind1+n_prog*2+1]
            m_merit     = np.double(branch[ind1+n_prog*2+1:ind1+n_prog*3+1])/1e10
            m_bid       = branch[ind1+n_prog*3+1:ind1+n_prog*4+1]

        return idlist, snlist, m_idlist, m_snaplist, m_merit, m_bid
##-----
## Get Merger Tree
##-----
    def f_getevol(self, n_snap, id0, horg='g'):#, gprop=gal_properties, directory=dir_catalog):

        # Get funtions
        gf  = vr_getftns(self)

        ## Get tree of this galaxy
        tree    = self.f_gettree(n_snap, id0, horg)

        idlist  = np.array(tree[0],dtype='int32')
        snlist  = np.array(tree[1],dtype='int32')
        n_link  = len(idlist)

        ## First read the galaxy
        g0  = self.f_rdgal(n_snap, id0, horg=horg)

        ## ALLOCATE
        gal = np.zeros(n_link, dtype=g0.dtype)

        ## READ
        ind = np.array(range(n_link),dtype='int32')
        for i in ind:
            gal[i]  = self.f_rdgal(snlist[i], idlist[i], horg)

        return gal

##-----
## Some basic drawing routines
##-----
class vr_draw:
    def __init__(self, vrobj):
        self.vrobj  = vrobj

    ##-----
    ## Draw Gas map
    ##-----
    def d_gasmap(self, n_snap, id0, cell2, amrtype=None, wtype=None, xr=None, yr=None, zr=None, n_pix=None, minlev=None, maxlev=None, proj=None):


        ##----- Settings
        if(n_pix==None): n_pix      = 1000
        if(amrtype==None): amrtype  = 'D'
        if(wtype==None): wtype      = 'MW'
        if(minlev==None): minlev    = self.vrobj.rtype_levmin
        if(maxlev==None): maxlev    = self.vrobj.rtype_levmax
        if(proj==None): proj        = 'xy'

        cell    = cell2[0]
        boxrange= cell2[1]
        units   = cell2[2]

        ##----- Boxsize
        galtmp  = self.vrobj.f_rdgal(n_snap, id0, horg='g')
        if(xr==None): xr  = np.array([-1, 1.],dtype='<f8') * boxrange + galtmp['Xc']
        if(yr==None): yr  = np.array([-1, 1.],dtype='<f8') * boxrange + galtmp['Yc']
        if(zr==None): zr  = np.array([-1, 1.],dtype='<f8') * boxrange + galtmp['Zc']

        ##----- Allocate
        gasmap  = np.zeros((n_pix,n_pix),dtype='<f8')
        gasmap_t= np.zeros((n_pix,n_pix),dtype='<f8')

        levind  = np.array(range(maxlev-minlev+1),dtype='int32') + minlev

        ##----- Get cell data from each level
        for lev in levind:
            ind = np.where(cell['level'][:]==lev)
            if(np.size(ind)==0): continue

            if(proj=='xy'):
                xx  = cell['xx'][ind]
                yy  = cell['yy'][ind]
                xr0 = xr
                yr0 = yr
            elif(proj=='xz'):
                xx  = cell['xx'][ind]
                yy  = cell['zz'][ind]
                xr0 = xr
                yr0 = zr

            bw  = np.array([1.,1.], dtype='<f8')*cell['dx'][ind[0][0]]


            ####----- Extract
            var = np.zeros((np.size(ind),2),dtype='<f8')
            var[:,1]    = cell['den'][ind]

            if(amrtype=='D'): var[:,0]      = cell['den'][ind]  ## [/cc]
            elif(amrtype=='T'): var[:,0]    = cell['temp'][ind] ## [K/mu]
            elif(amrtype=='PT'): var[:,0]   = cell['P_thermal'][ind]
            elif(amrtype=='PR'):
                tvx = cell['vx'][ind] - galtmp['VXc']
                tvy = cell['vy'][ind] - galtmp['VYc']
                tvz = cell['vz'][ind] - galtmp['VZc']

                txx = cell['xx'][ind] - galtmp['Xc']
                tyy = cell['yy'][ind] - galtmp['Yc']
                tzz = cell['zz'][ind] - galtmp['Zc']
                vv2 = tvx**2 + tvy**2 + tvz**2

                var[:,0]    = (cell['den'][ind]/units.unit_nH) * (vv2 / units.kms**2) * units.unit_m / units.unit_l / units.unit_t**2 / np.double(1.3806200e-16)
                    ## [K/cm^-3]

                ## Inward RP Only
                ind2    = np.where(txx*tvx + tyy*tvy + tzz*tvz > 0)
                if(np.size(ind)>0): var[ind2,0] = 0.

            elif(amrtype=='Z'): var[:,0]    = cell['metal'][ind]


            ####----- Collect map
            larr    = np.zeros(20, dtype=np.int32)
            darr    = np.zeros(20, dtype='<f8')

            larr[0] = np.int32(np.size(ind))
            larr[1] = np.int32(n_pix)
            larr[2] = np.int32(self.vrobj.num_thread)

            if(wtype=='MW'): larr[10] = 1
            if(wtype=='VW'): larr[10] = 2
            if(wtype=='MAX'):larr[10] = 3

            js_gasmap_py.js_gasmap(larr, darr, xx, yy, var, bw, xr0, yr0)
            #mapdum  = js_gasmap_py.map

        mapdum  = js_gasmap_py.map

        gasmap  = np.zeros((n_pix,n_pix),dtype='<f8')
        gasmap_t= np.zeros((n_pix,n_pix),dtype='<f8')

        gasmap      = mapdum[:,:,0]
        gasmap_t    = mapdum[:,:,1]

        if(wtype == 'MW'):
            cut = np.where(gasmap_t > 0)
            if(np.size(cut)>0): gasmap[cut] /= gasmap_t[cut]

            cut = np.where(gasmap_t == 0)
            if(np.size(cut)>0): gasmap[cut] = 0.

        return gasmap, xr, yr

##-----
## Call FTNs
##-----
class vr_getftns:
    def __init__(self, vrobj):
        self.vrobj      = vrobj

    ##-----
    ## Get img
    ##-----
    def g_img(self, gasmap, zmin=None, zmax=None, scale=None):

        if(zmin==None): zmin = np.min(gasmap)
        if(zmax==None): zmax = np.max(gasmap)

        ind    = np.where(gasmap < zmin)
        if(np.size(ind)>0): gasmap[ind] = 0

        ind     = np.where(gasmap > 0)
        gasmap[ind] -= zmin

        ind = np.where(gasmap > (zmax - zmin))
        if(np.size(ind)>0): gasmap[ind] = zmax - zmin

        gasmap  /= (zmax - zmin)

        if(scale=='log' or scale==None):
            gasmap  = np.log(gasmap*1000. + 1.) / np.log(1000.)
            #gasmap  = np.int32(gasmap * 255.)
        return gasmap

    ##-----
    ## Compute Gyr from conformal time
    ##-----
    def g_gyr(self, n_snap, t_conf):

        # Initial Settings
        aexp    = np.double(np.loadtxt(self.vrobj.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=9, max_rows=1)[2])
        H0      = self.vrobj.H0
        omega_M = self.vrobj.omega_M
        omega_L = self.vrobj.omega_L

        #----- Allocate
        data    = np.zeros(len(t_conf), dtype=[('sfact','<f8'), ('gyr','<f8')])

        #----- Get Confalmal T - Sfact Table
        c_table = self.g_cfttable()

        #----- Get Sfactor by interpolation
        lint = interpolate.interp1d(c_table['conft'],c_table['sfact'],kind = 'quadratic')
        data['sfact'][:]   = lint(t_conf)

        #----- Get Gyr from Sfactor
        g_table = self.g_gyrtable()
        lint = interpolate.interp1d(g_table['redsh'],g_table['gyr'],kind = 'quadratic')
        t0  = lint( 1./aexp - 1.)

        data['gyr'][:]      = lint( 1./data['sfact'][:] - 1.) - t0
        return data

    ##-----
    ## Generate or Load Confal-Gyr Table
    ##-----
    def g_cfttable_ftn(self, X, oM, oL):
        return 1./(X**3 * np.sqrt(oM/X**3 + oL))

    def g_cfttable(self):

        H0     = self.vrobj.H0
        oM     = self.vrobj.omega_M
        oL     = self.vrobj.omega_L

        # reference path is correct?
        fname   = self.vrobj.dir_table + 'cft_%0.5d'%(H0*1000.) + '_%0.5d'%(oM*100000.) + '_%0.5d'%(oL*100000.) + '.pkl'
        isfile = os.path.isfile(fname)
        if(isfile==True):
            with open(fname, 'rb') as f:
                data = pickle.load(f)
        else:
            n_table = np.int32(10000)
            data    = np.zeros(n_table, dtype=[('sfact','<f8'), ('conft','<f8')])
            data['sfact'][:]    = np.array(range(n_table),dtype='<f8')/(n_table - 1.) * 0.98 + 0.02

            ind     = np.array(range(n_table),dtype='int32')
            for i in ind:
                data['conft'][i]  = integrate.quad(self.g_cfttable_ftn,data['sfact'][i],1.,args=(oM,oL))[0] * (-1.)

            with open(fname, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        return data

    ##-----
    ## Generate or Load Sfactor-Gyr Table
    ##-----
    def g_gyrtable_ftn(self, X, oM, oL):
        return 1./(1.+X)/np.sqrt(oM*(1.+X)**3 + oL)

    def g_gyrtable(self):

        H0     = self.vrobj.H0
        oM     = self.vrobj.omega_M
        oL     = self.vrobj.omega_L

        fname   = self.vrobj.dir_table + 'gyr_%0.5d'%(H0*1000.) + '_%0.5d'%(oM*100000.) + '_%0.5d'%(oL*100000.) + '.pkl'
        isfile = os.path.isfile(fname)

        if(isfile==True):
            with open(fname, 'rb') as f:
                data = pickle.load(f)
        else:
            n_table = np.int32(10000)
            data    = np.zeros(n_table, dtype=[('redsh','<f8'),('gyr','<f8')])
            data['redsh'][:]    = 1./(np.array(range(n_table),dtype='<f8')/(n_table - 1.) * 0.98 + 0.02) - 1.
            data['gyr'][0]  = 0.

            ind     = np.array(range(n_table),dtype='int32')
            for i in ind:
                data['gyr'][i]  = integrate.quad(self.g_gyrtable_ftn,0.,data['redsh'][i], args=(oM, oL))[0]
                data['gyr'][i]  *= (1./H0 * np.double(3.08568025e19) / np.double(3.1536000e16))

            with open(fname, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        return data

    ##-----
    ## Compute flux
    ##-----
    def g_flux_ssptable(self):

        ssp_type    = self.vrobj.ssp_type
        fname       = self.vrobj.dir_table + 'ssp_' + ssp_type + '.pkl'
        isfile      = os.path.isfile(fname)

        if(isfile==True):
            with open(fname, 'rb') as f:
                data = pickle.load(f)
        else:
            dname   = self.vrobj.dir_table + 'ssp_' + ssp_type

            metal   = np.loadtxt(dname + '/metal.txt', dtype='<f8')
            age     = np.loadtxt(dname + '/age.txt', dtype='<f8')
            lambd   = np.loadtxt(dname + '/lambda.txt', dtype='<f8')

            tr_curve    = []
            for name in flux_list:
                fname2  = dname + '/' + name + '_tr.txt'
                tr_curve.append(np.loadtxt(fname2, dtype='<f8'))

            flux    = np.zeros((len(metal), len(lambd), len(age)), dtype='<f8')
            ind     = np.array(range(len(metal)),dtype='int32')
            for i in ind:
                fname2  = dname + '/flux_%0.1d'%i + '.txt'
                dum     = np.array(np.loadtxt(fname2, dtype='<f8'))
                dum     = np.reshape(dum, (len(age),len(lambd)))
                dum     = np.transpose(dum)
                flux[i,:,:] = dum

            data    = {"metal":metal, "age":age, "lambda":lambd, "tr_curve":tr_curve, "flux":flux}

            with open(fname, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        return data

    def g_flux(self, mass, metal, age, fl_name):
        #----- ALLOCATE
        dtype   = ['']
        for name in self.vrobj.vr_fluxlist:
            dtype   += [(name, '<f8')]

        dtype   = dtype[1:]
        data    = np.zeros(len(mass), dtype=dtype)

        #----- LOAD SSP TABLE
        ssp = self.g_flux_ssptable()

        #----- COMPUTE FLUX
        larr    = np.zeros(20, dtype=np.int32)
        darr    = np.zeros(20, dtype='<f8')
        larr[0]     = np.int32(len(mass))
        larr[1]     = np.int32(len(ssp['age']))
        larr[2]     = np.int32(len(ssp['metal']))
        larr[3]     = np.int32(len(ssp['lambda']))
        larr[10]    = self.vrobj.num_thread

        ind     = np.array(range(len(self.vrobj.vr_fluxlist)),dtype='int32')
        for i in ind:
            if(self.vrobj.vr_fluxlist[i]!=fl_name): continue
            larr[4]     = np.int32(len(ssp['tr_curve'][:][i]))
            get_flux_py.get_flux(age, metal, mass, ssp['age'], ssp['metal'], ssp['lambda'], ssp['flux'], ssp['tr_curve'][i][:,0], ssp['tr_curve'][i][:,1], larr, darr)

            flux_tmp    = get_flux_py.flux * np.double(3.826e33) / (4. * np.pi * (10.0 * 3.08567758128e18)**2)

            dlambda = ssp['tr_curve'][i][1:,0] - ssp['tr_curve'][i][:-1,0]
            clambda = (ssp['tr_curve'][i][1:,0] + ssp['tr_curve'][i][:-1,0])/2.
            trcurve = (ssp['tr_curve'][i][1:,1] + ssp['tr_curve'][i][:-1,1])/2.

            flux0   = np.sum(dlambda * clambda * trcurve * self.vrobj.vr_fluxzp[i])

            data[self.vrobj.vr_fluxlist[i]]    = flux_tmp / flux0


        return data
"""
    Currently aborted because some snapshots do not have the corresponding sinkprops.dat
    Last updated 22.09.01 Jinsu


    def g_sink(self, snap, xr=None, yr=None, zr=None):
        #----- GET COARSE STEP & SIM units
        ncoarse    = np.int64(np.loadtxt(self.vrobj.dir_raw+'output_%0.5d'%snap+"/info_%0.5d"%snap+".txt", dtype=object, skiprows=5, max_rows=1)[1])

        #----- READ sink_*.dat
        fname   = self.vrobj.dir_sink + 'sink_%0.5d'%ncoarse+'.dat'
        f       = FortranFile(fname, 'r')

        nsink   = f.read_ints(np.int32)
        ndim    = f.read_ints(np.int32)
        aexp    = f.read_reals(np.double)
        unit_l  = f.read_reals(np.double)
        unit_d  = f.read_reals(np.double)
        unit_t  = f.read_reals(np.double)

        f.close()

        return nsink, ndim, aexp, unit_l, unit_d, unit_t
"""
