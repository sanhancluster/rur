import numpy as np
import h5py
import os
import os.path

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

