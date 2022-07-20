import numpy as np
import h5py
import os
import os.path

from rur.vr.fortran.find_domain_py import find_domain_py
from rur.vr.fortran.get_ptcl_py import get_ptcl_py

class vr_load:

    def __init__(self, simtype, num_thread=1):
        ##-----
        ## General settings
        ##-----
        self.num_thread = int(num_thread)   # The number of cores
        self.simtype    = simtype

        ##-----
        ## Specify configurations based on the simulation type
        ##-----
        if(simtype == 'NH'):
            # Path related
            self.dir_raw        = '/storage6/NewHorizon/snapshots/'
            self.dir_catalog    = '/storage5/NewHorizon/VELOCIraptor/'

            # Ramses related
            self.rtype_llint    = False     # Whether particle IDs are 64 byte integer
            self.rtype_family   = False     # Whether part_out contains family
            self.rtype_neff     = int(4096) # Effective resolution of the zoom region
            self.rtype_ndomain  = int(4800) # The number of MPI domains

            # VR output related
            self.vr_columnlist  = ['ID', 'ID_mbp', 'hostHaloID', 'numSubStruct', 'Structuretype', 'Mvir', 'Mass_tot', 'Mass_FOF',
                       'Mass_200mean', 'Efrac', 'Mass_200crit', 'Rvir', 'R_size', 'R_200mean', 'R_200crit',
                       'R_HalfMass', 'R_HalfMass_200mean', 'R_HalfMass_200crit', 'Rmax', 'Xc', 'Yc', 'Zc', 'VXc',
                       'VYc', 'VZc', 'Lx', 'Ly', 'Lz', 'sigV', 'Vmax', 'npart']
                                                                # Catalog output
            self.vr_galprop     = ['SFR', 'ABmag']              # Bulk properties computed in the post-processing
            self.vr_fluxlist    = ['u', 'g', 'r', 'i', 'z']     # flux list of Abmag
            self.vr_fluxzp      = np.double(np.array([895.5*1e-11, 466.9*1e-11, 278.0*1e-11, 185.2*1e-11, 131.5*1e-11]))
            self.vr_treefile    = '/storage5/NewHorizon/VELOCIraptor/Galaxy/tree/l1/ctree.dat'
                                                                # flux zero points
        elif(simtype == 'NH2'):
            # Path related
            self.dir_raw        = '/storage7/NH2/snapshots/'
            self.dir_catalog    = '/storage7/NH2/VELOCIraptor/'

            # Ramses related
            self.rtype_llint    = False     # Whether particle IDs are 64 byte integer
            self.rtype_family   = True     # Whether part_out contains family
            self.rtype_neff     = int(4096) # Effective resolution of the zoom region
            self.rtype_ndomain  = int(480) # The number of MPI domains

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
            self.vr_treefile    = 'void'

        elif(simtype == 'FORNAX' or simtype == 'FN'):
            # Path related
            self.dir_raw        = '/storage5/FORNAX/KISTI_OUTPUT/l10006/'
            self.dir_catalog    = '/storage5/FORNAX/VELOCIraptor/l10006/'

            # Ramses related
            self.rtype_llint    = False     # Whether particle IDs are 64 byte integer
            self.rtype_family   = True     # Whether part_out contains family
            self.rtype_neff     = int(2048) # Effective resolution of the zoom region
            self.rtype_ndomain  = int(480) # The number of MPI domains

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
            self.vr_treefile    = '/storage5/FORNAX/VELOCIraptor/Galaxy/tree/l3/ctree.dat'
        else:
            print('%-----')
            print(' Wrong argument for the simtype')
            print('     Simtype list: NH, NH2, FORNAX(or FN), NC(not yet)')
            print('%-----')


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
    ##      *) INCLUDE FLUX & TIME COMPUTATION PARTS
    ##      *) Halo member load is not implemented
    ##-----
    def f_rdptcl(self, n_snap, id0, horg='g', p_gyr=False, p_sfactor=False, p_mass=True, p_flux=False,
            p_metal=False, p_id=False, raw=False, boxrange=50., domlist=[0], num_thread=None):


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

        # READ PTCL ID & Domain List (Might be skipped when raw==True)
        if(raw==False):
            if(horg=='h'): fname = self.dir_catalog + 'Halo/VR_Halo/snap_%0.4d'%n_snap+"/"
            elif(horg=='g'): fname = self.dir_catalog + 'Galaxy/VR_Galaxy/snap_%0.4d'%n_snap+"/"
            fname   += 'GAL_%0.6d'%id0+'.hdf5'

            dat     = h5py.File(fname, 'r')
            idlist  = np.array(dat.get("P_Prop/P_ID"))
            if(horg=='g'): domlist = np.array(dat.get("Domain_List"))
            else: domlist = np.zeros(1)
        else:
            idlist  = np.zeros(1, dtype=np.int64)
            domlist = np.zeros(self.rtype_ndomain, dtype=np.int32) - 1

            #----- Find Domain
            galtmp  = f_rdgal(n_snap, id0, horg=horg)

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

            if(horg=='g'): larr[11] = 10
            else: larr[11] = -10

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
            larr[10]= np.int32(len(dir_raw))
            larr[17]= 100

            if(horg=='g'): larr[11] = 10
            else: larr[11] = -10

            if(self.rtype_family==True): larr[18] = 100
            else: larr[18] = 0
            if(self.rtype_llint==True): larr[19] = 100
            else: larr[19] = 0

            if(horg=='h'): darr[11] = dmp_mass

            get_ptcl_py.get_ptcl(dir_raw, idlist, domlist, larr, darr)
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
            dtype   += [(name, '<f8')]

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
        #if(p_gyr==True):
        #    gyr = g_gyr(n_snap, pinfo[:,7])
        #    ptcl['gyr'][:]  = gyr['gyr'][:]
        #    ptcl['sfact'][:]= gyr['sfact'][:]

        ##---- COMPUTE FLUX
        #if(p_flux==True):
        #    for name in flux_list:
        #        ptcl[name][:] = g_flux(ptcl['mass'][:], ptcl['metal'][:], ptcl['gyr'][:],name)[name]

        return ptcl, rate, domlist, idlist
