import numpy as np
import h5py
import sys
import os
import copy
import os.path
import pickle as pickle
import pkg_resources
import multiprocessing
from multiprocessing import Process, Queue
from time import sleep

from scipy import interpolate
from scipy.io import FortranFile
import scipy.integrate as integrate

from rur.sci import kinematics
import rur.uri as uri

from rur.vr.fortran.find_domain_py import find_domain_py
from rur.vr.fortran.get_ptcl_py import get_ptcl_py
from rur.vr.fortran.get_flux_py import get_flux_py
from rur.vr.fortran.jsamr2cell_totnum_py import jsamr2cell_totnum_py
from rur.vr.fortran.jsamr2cell_py import jsamr2cell_py
from rur.vr.fortran.js_gasmap_py import js_gasmap_py
from rur.vr.fortran.jsrd_part_totnum_py import jsrd_part_totnum_py
from rur.vr.fortran.jsrd_part_py import jsrd_part_py

class vr_load:

    def __init__(self, simtype, cname=False, num_thread=1):
        ##-----
        ## General settings
        ##-----
        self.num_thread = int(num_thread)   # The number of cores
        self.simtype    = simtype
        self.dir_table  = pkg_resources.resource_filename('rur', 'vr/table/')
        self.ssp_type   = 'chab'
        self.sun_met    = 0.02

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
            self.vr_galprop     = ['SFR', 'ABmag', 'ConFrac_n', 'ConFrac_M']              # Bulk properties computed in the post-processing
            self.vr_general     = ['SFR_R', 'SFR_T', 'MAG_R', 'CONF_R']
            self.vr_galinfo     = ['isclump', 'rate', 'Aexp', 'Domain_List', 'snapnum']
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
            self.vr_galprop     = ['SFR', 'ABmag', 'ConFrac_n', 'ConFrac_M']              # Bulk properties computed in the post-processing
            self.vr_general     = ['SFR_R', 'SFR_T', 'MAG_R', 'CONF_R']
            self.vr_galinfo     = ['isclump', 'rate', 'Aexp', 'Domain_List', 'snapnum']
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
            self.vr_galprop     = ['SFR', 'ABmag', 'ConFrac_n', 'ConFrac_M']              # Bulk properties computed in the post-processing
            self.vr_general     = ['SFR_R', 'SFR_T', 'MAG_R', 'CONF_R']
            self.vr_galinfo     = ['isclump', 'rate', 'Aexp', 'Domain_List', 'snapnum']
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
            self.vr_galprop     = ['SFR', 'ABmag', 'ConFrac_n', 'ConFrac_M']              # Bulk properties computed in the post-processing
            self.vr_general     = ['SFR_R', 'SFR_T', 'MAG_R', 'CONF_R']
            self.vr_galinfo     = ['isclump', 'rate', 'Aexp', 'Domain_List', 'snapnum']
            self.vr_fluxlist    = ['u', 'g', 'r', 'i', 'z']     # flux list of Abmag
            self.vr_fluxzp      = np.double(np.array([895.5*1e-11, 466.9*1e-11, 278.0*1e-11, 185.2*1e-11, 131.5*1e-11]))

        elif(simtype == 'NC'):
            # Path related
            self.dir_raw        = '/storage7/NewCluster2/snapshots/'
            self.dir_catalog    = '/storage7/NewCluster2/VELOCIraptor/'
            self.dir_sink       = '/storage7/NewCluster2/SINKPROPS/'

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
            self.vr_galprop     = ['SFR', 'ABmag', 'ConFrac_n', 'ConFrac_M']              # Bulk properties computed in the post-processing
            self.vr_general     = ['SFR_R', 'SFR_T', 'MAG_R', 'CONF_R']
            self.vr_galinfo     = ['isclump', 'rate', 'Aexp', 'Domain_List', 'snapnum']
            self.vr_fluxlist    = ['u', 'g', 'r', 'i', 'z']     # flux list of Abmag
            self.vr_fluxzp      = np.double(np.array([895.5*1e-11, 466.9*1e-11, 278.0*1e-11, 185.2*1e-11, 131.5*1e-11]))
                                                                # flux zero points                                                                # flux zero points
        else:
            print('%-----')
            print(' Wrong argument for the simtype')
            print('     Simtype list: NH, NH2, FORNAX(or FN), NC')
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

        ##----- Catalog output
        self.dtype_part = [('xx', '<f8'), ('yy', '<f8'), ('zz', '<f8'),
                ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
                ('mass', '<f8'), ('sfact', '<f8'), ('gyr', '<f8'), ('metal', '<f8'), ('family', np.int32) , ('domain', np.int32), 
                ('id', np.int64)]
        for name in self.vr_fluxlist:
            self.dtype_part   += [('f_' + name, '<f8')]

        self.dtype_amr  = [('xx','<f8'), ('yy','<f8'), ('zz','<f8'),
            ('vx','<f8'), ('vy','<f8'), ('vz','<f8'), ('dx','<f8'), ('mass', '<f8'),
            ('type',np.int32), ('PE','<f8'), ('KE','<f8'), ('UE','<f8'), 
            ('den','<f8'), ('temp','<f8'), ('P_thermal','<f8'), ('metal','<f8'), ('level',np.int32)]

        ##----- LOAD TREE DATA
        self.tree_data_g = None
        self.tree_data_h = None
        self.tree_simtype= None

    ##-----
    ## Load Galaxy
    ##  To do list
    ##      1) do not use imglist.txt
    ##      2) More efficienct way of reading multiple hdf5 files?
    ##-----
    class f_rdgal_parallel:
        def __init__(self, vrobj, h5data, gidlist, galdata):
            self.galdata = galdata
            self.gidlist = gidlist
            self.h5data = h5data
            self.vrobj = vrobj

        def f_rdgal_input(self, start, end):
            for i in range(start, end):
                idstr  = 'ID_%0.6d'%self.gidlist[i]

                for name in self.vrobj.vr_columnlist:
                    xdata = self.h5data.get(idstr + "/G_Prop/G_" + name)
                    self.galdata[name][i] = np.array(xdata)

                for name in self.vrobj.vr_galprop:
                    xdata = self.h5data.get(idstr + "/G_Prop/G_" + name)
                    self.galdata[name][i] = np.array(xdata)
    
                for name in self.vrobj.vr_general:
                    xdata = self.h5data.get("/" + name)
                    self.galdata[name][i] = np.array(xdata)
    
                for name in self.vrobj.vr_galinfo:
                    if(name!='snapnum'):
                        xdata = self.h5data.get(idstr + "/" + name)
                        self.galdata[name][i] = np.array(xdata)
        
        def f_rdgal_input_p(self, start, end, q):
            for i in range(start, end):
                idstr  = 'ID_%0.6d'%self.gidlist[i]

                for name in self.vrobj.vr_columnlist:
                    xdata = self.h5data.get(idstr + "/G_Prop/G_" + name)
                    self.galdata[name][i] = np.array(xdata)

                for name in self.vrobj.vr_galprop:
                    xdata = self.h5data.get(idstr + "/G_Prop/G_" + name)
                    self.galdata[name][i] = np.array(xdata)
    
                for name in self.vrobj.vr_general:
                    xdata = self.h5data.get("/" + name)
                    self.galdata[name][i] = np.array(xdata)
    
                for name in self.vrobj.vr_galinfo:
                    if(name!='snapnum'):
                        xdata = self.h5data.get(idstr + "/" + name)
                        self.galdata[name][i] = np.array(xdata)

            q.put((start, end, self.galdata[start:end]))

    def f_rdgal(self, n_snap, id0, mrange=None, horg='g'):

        # Path setting
        if(horg=='h'): fname = self.dir_catalog + 'Halo/VR_Halo/snap_%0.4d'%n_snap + '.hdf5'
        elif(horg=='g'): fname = self.dir_catalog + 'Galaxy/VR_Galaxy/snap_%0.4d'%n_snap+'.hdf5'
        else:
            print('%-----')
            print(' Wrong argument for the horg')
            print('     horg = "g" (for galaxy) or "h" (for halo)')
            print('%-----')


        # Open hdf5
        dat     = h5py.File(fname, 'r')

        # Get ID & Mass First
        mass_tot    = np.array(dat.get('Mass_tot'),dtype='<f8')
        mvir        = np.array(dat.get('Mvir'),dtype='<f8')
        idlist      = np.array(dat.get('ID'),dtype='<i4')

        # Extract
        if id0 > 0:
            idlist = idlist[idlist == id0]
        else:
            if not (mrange is None):
                if horg == 'g': idlist = idlist[(mass_tot >= mrange[0]) * (mass_tot < mrange[1])]
                if horg == 'h': idlist = idlist[(mvir >= mrange[0]) * (mvir < mrange[1])]

        n_gal   = len(idlist)

        # Set column list
        dtype=[]

        ##----- Original catalog
        for name in self.vr_columnlist:
            if(name=='ID' or name=='hostHaloID' or name=='numSubStruct' or name=='Structuretype' or name=='npart'): dtype=dtype+[(name, np.int32)]
            elif(name=='ID_mbp'): dtype=dtype+[(name, np.int64)]
            else: dtype=dtype+[(name, '<f8')]


        ##----- Gal prop with post-processing
        for name in self.vr_galprop:
            dtype=dtype+[(name, 'object')]

        for name in self.vr_general:
            dtype=dtype+[(name, 'object')]

        for name in self.vr_galinfo:
            if(name=='Domain_List'):
                dtype=dtype + [(name, 'object')]
            elif(name=='snapnum'):
                dtype=dtype + [(name, '<i4')]
            else:
                dtype=dtype + [(name, '<f8')]

        galdata=np.zeros(n_gal, dtype=dtype)

        galdata['snapnum'] = n_snap
        #with multiprocessing.Pool(self.num_thread) as pool:
        #    #tasks = [(self.f_rdgal_input, (i, gid, galdata, dat)) for i, gid in enumerate(idlist)]
        #    void = pool.starmap(self.f_rdgal_input, [(i, gid, galdata, dat) for i, gid in enumerate(idlist)])
        #    pool.close()
        #    pool.join()
        #if __name__ == '__main__':
        p_input = self.f_rdgal_parallel(self, dat, idlist, galdata)

        if n_gal < self.num_thread:
            #serially
            p_input.f_rdgal_input(0, n_gal)
            return p_input.galdata
        else:
            #parallelly (motivated by uhmi.py)
            dind  = np.int32(n_gal / self.num_thread)
            ps = []
            q = Queue()

            for th in range(self.num_thread):
                i0 = th * dind
                i1 = (th+1) * dind
                if(th==0): i0 = 0
                if(th==self.num_thread-1): i1 = n_gal
                p = Process(target=p_input.f_rdgal_input_p, args=(i0, i1, q))
                ps.append(p)

                p.start()
                while not q.empty():
                    i0, i1, dumdata = q.get()
                    galdata[i0:i1] = dumdata
            ok = False
            while not ok:
                ok = True
                for idx in np.arange(len(ps)):
                    if (ps[idx].is_alive()):
                        ok = False
                if(not q.empty()):
                    i0, i1, dumdata = q.get()
                    galdata[i0:i1] = dumdata
                else:
                    sleep(0.5)

            return galdata


    ##-----
    ## Load Particle of a galaxy
    ##  To do list
    ##      *) Halo member load is not implemented
    ##-----
    def f_rdptcl(self, n_snap, id0, radius=-1.0, horg='g', p_age=False, p_flux=False, p_sink=False,
            num_thread=None, info=None):

        # Get funtions
        gf  = vr_getftns(self)

        # Initial settings
        if(p_age==False and p_flux==True): p_age=True
        if(num_thread is None): num_thread=self.num_thread
        if(info is None): info=gf.g_info(n_snap)
        if(p_sink==True): p_id=True

        #unit_l  = np.double(np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=15, max_rows=1)[2])
        #unit_t  = np.double(np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=17, max_rows=1)[2])
        #kms     = np.double(unit_l / unit_t / 1e5)
        #unit_d  = np.double(np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=16, max_rows=1)[2])
        #unit_m  = unit_d * unit_l**3
        #levmax  = np.int32( np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=3, max_rows=1)[2])
        #hindex  = np.double(np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=21)[:,1:])
        #omega_M = np.double(np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=11, max_rows=1)[2])
        #omega_B = np.double(np.loadtxt(self.dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=14, max_rows=1)[2])

        dmp_mass    = 1.0/(self.rtype_neff*self.rtype_neff*self.rtype_neff)*(info['oM'] - info['oB'])/info['oM']

        
        if(radius > 0): # READ PTCLS WITHIN THE APERTURE
            galtmp  = self.f_rdgal(n_snap, id0, horg=horg)
            return gf.g_ptcl(n_snap, galtmp['Xc'], galtmp['Yc'], galtmp['Zc'], radius, 
                info=info, num_thread=num_thread, p_age=p_age, p_flux=p_flux, p_sink=p_sink)
        else: # READ MEMBER PTCLS ONLY
            if(horg=='h'): fname = self.dir_catalog + 'Halo/VR_Halo/snap_%0.4d'%n_snap+"/"
            elif(horg=='g'): fname = self.dir_catalog + 'Galaxy/VR_Galaxy/snap_%0.4d'%n_snap+"/"
            fname   += 'GAL_%0.6d'%id0+'.hdf5'

            dat     = h5py.File(fname, 'r')
            idlist  = np.int64(np.array(dat.get("P_Prop/P_ID")))
            domlist = np.int32(np.array(dat.get("Domain_List")))
            domlist = np.int32(np.array(np.where(domlist > 0))[0] + 1)

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
      
            #----- EXTRACT
            n_old       = len(pinfo)*1.

            pinfo       = pinfo[np.where(pinfo[:,0]>-1e7)]
            n_new       = len(pinfo)*1.
            rate        = n_new / n_old

            pinfo[:,0]    *= info['unit_l'] / 3.086e21
            pinfo[:,1]    *= info['unit_l'] / 3.086e21
            pinfo[:,2]    *= info['unit_l'] / 3.086e21
            pinfo[:,3]    *= info['kms']
            pinfo[:,4]    *= info['kms']
            pinfo[:,5]    *= info['kms']
            pinfo[:,6]    *= info['unit_m'] / 1.98892e33

            #----- OUTPUT ARRAY
            #dtype   = [('xx', '<f8'), ('yy', '<f8'), ('zz', '<f8'),
            #    ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
            #    ('mass', '<f8'), ('sfact', '<f8'), ('gyr', '<f8'), ('metal', '<f8'), ('family', np.int32) , ('domain', np.int32), 
            #    ('id', np.int64)]

            #for name in self.vr_fluxlist:
            #    dtype   += [('f_' + name, '<f8')]
            dtype   = self.dtype_part

            ptcl    = np.zeros(np.int32(n_new), dtype=dtype)

            ptcl['xx'][:]   = pinfo[:,0]
            ptcl['yy'][:]   = pinfo[:,1]
            ptcl['zz'][:]   = pinfo[:,2]

            ptcl['vx'][:]   = pinfo[:,3]
            ptcl['vy'][:]   = pinfo[:,4]
            ptcl['vz'][:]   = pinfo[:,5]

            ptcl['mass'][:] = pinfo[:,6]
            ptcl['metal'][:]= pinfo[:,8]

            ptcl['id'][:]   = idlist

            ##----- COMPUTE GYR
            if(p_age==True):
                gyr = gf.g_gyr(n_snap, pinfo[:,7])
                ptcl['gyr'][:]  = gyr['gyr'][:]
                ptcl['sfact'][:]= gyr['sfact'][:]

            ##----- COMPUTE FLUX
            if(p_flux==True):
                for name in self.vr_fluxlist:
                    ptcl['f_' + name][:] = gf.g_flux(ptcl['mass'][:], ptcl['metal'][:], ptcl['gyr'][:],name)[name]


            get_ptcl_py.get_ptcl_free()
            return ptcl


    ##-----
    ## LOAD CELL DATA around galaxies
    ##  IF domlist is given, the shell is negelected
    ##-----
    def f_rdamr(self, n_snap, id0, radius, horg='g', info=None, domlist=None, num_thread=None, amrtype=False):

        ##----- Settings
        gf  = vr_getftns(self)
        if(info is None): info = gf.g_info(n_snap)
        if(num_thread is None): num_thread = self.num_thread

        ##----- Read gal
        galtmp  = self.f_rdgal(n_snap, id0, horg=horg)
           
        ##----- Get Domain
        if(domlist is None):
            if(radius<0): domlist = np.int32(np.arange(1, info['ncpu']+1))
            else:
                domlist = gf.g_domain(n_snap, galtmp['Xc'], galtmp['Yc'], galtmp['Zc'], radius, info=info)
        else:
            if not (isinstance(domlist, np.ndarray)): domlist = np.int32(np.array(domlist))

            if(np.amin(domlist) < 1 or np.amax(domlist) > info['ncpu']):
                print('%-----')
                print(' Wrongly argued')
                print('     out of range of the domain: domlist')
                print('%-----')
                sys.exit(1)

        ##----- Set box
        xc = galtmp['Xc']
        yc = galtmp['Yc']
        zc = galtmp['Zc']
        vxc = galtmp['VXc']
        vyc = galtmp['VYc']
        vzc = galtmp['VZc']

        return gf.g_amr(n_snap, xc, yc, zc, vxc, vyc, vzc, radius, domlist=domlist, num_thread=num_thread, info=info, amrtype=amrtype)


        ##------ Unit Load
        #infofile    = self.dir_raw + 'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt"
        #unit_l  = np.double(np.loadtxt(infofile, dtype=object, skiprows=15, max_rows=1)[2])
        #unit_t  = np.double(np.loadtxt(infofile, dtype=object, skiprows=17, max_rows=1)[2])
        #unit_d  = np.double(np.loadtxt(infofile, dtype=object, skiprows=16, max_rows=1)[2])
        #hindex  = np.double(np.loadtxt(infofile, dtype=object, skiprows=21)[:,1:])
        #kms     = np.double(unit_l / unit_t / 1e5)
        #unit_m  = unit_d * unit_l**3
        #unit_T2 = np.double(1.6600000e-24) / np.double(1.3806200e-16) * np.double(unit_l / unit_t)**2
        #nH  = np.double(0.76) / np.double(1.6600000e-24) * unit_d

        ##----- Find domain
        #domlist = gf.g_domain(n_snap, xc, yc, zc, radius)

        #domlist = np.zeros(self.rtype_ndomain, dtype=np.int32) - 1
        #xc  = galtmp['Xc']/unit_l * 3.086e21
        #yc  = galtmp['Yc']/unit_l * 3.086e21
        #zc  = galtmp['Zc']/unit_l * 3.086e21

        #rr  = galtmp['R_HalfMass']/unit_l * 3.086e21
        #larr    = np.zeros(20, dtype=np.int32)
        #darr    = np.zeros(20, dtype='<f8')

        #larr[0] = np.int32(len(xc))
        #larr[1] = np.int32(len(domlist))
        #larr[2] = np.int32(self.num_thread)
        #larr[3] = np.int32(self.rtype_levmax)

        #darr[0] = 50.
        #if(radius!=None): darr[0] = radius / (rr * unit_l / 3.086e21)

        #find_domain_py.find_domain(xc, yc, zc, rr, hindex, larr, darr)
        #domlist     = find_domain_py.dom_list
        #domlist     = domlist[0][:]
        #domlist = np.int32(np.array(np.where(domlist > 0))[0] + 1)



        """
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
        jsamr2cell_totnum_py.jsamr2cell_totnum_free()

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
        jsamr2cell_py.jsamr2cell_free()

        data['xx'][:]   = xgdum[:,0] * info['unit_l'] / np.double(3.086e21)
        data['yy'][:]   = xgdum[:,1] * info['unit_l'] / np.double(3.086e21)
        data['zz'][:]   = xgdum[:,2] * info['unit_l'] / np.double(3.086e21)
        data['vx'][:]   = hvdum[:,1] * kms
        data['vy'][:]   = hvdum[:,2] * kms
        data['vz'][:]   = hvdum[:,3] * kms
        data['den'][:]  = hvdum[:,0]
        data['temp'][:] = hvdum[:,4]
        data['metal'][:]= hvdum[:,5]
        data['dx'][:]   = dxdum[:] * info['unit_l'] / np.double(3.086e21)
        data['level'][:]= lvdum[:]
        #data['mass'][:] = np.double(10.**(np.log10(hvdum[:,0]) + np.log10(unit_d) + np.double(3.0) * (np.log10(dxdum[:]) + np.log10(info['unit_l'])) - np.log10(1.98892e33)))
        data['mass'][:] = hvdum[:,0] *unit_d * (dxdum[:] * info['unit_l'])**3 / np.double(1.98892e33)

        data    = data[np.where(lvdum >= 0)]
        dumA = data['temp'][:]
        dumB = data['den'][:]

        data['temp'][:] = data['temp'][:] / data['den'][:] * unit_T2    # [K/mu]
        data['den'][:]  *= nH                                           # [atom/cc]
        data['P_thermal'][:]    = dumA * unit_m / info['unit_l'] / unit_t**2 / np.double(1.3806200e-16)

        units   = {'unit_l':info['unit_l'], 'unit_m':unit_m, 'unit_t':unit_t, 'kms':kms, 'unit_nH':nH, 'unit_T2':unit_T2}
        return data, radius, units
        """

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

        # Check whether tree has ever been loaded
        if(self.tree_simtype == None or self.tree_simtype != self.simtype):
            self.tree_simtype = copy.deepcopy(self.simtype)

            directory   = self.dir_catalog

            ## FOR GALAXIES            
            dir_tree_g  = directory + 'Galaxy/tree/'

            ## Is pickle?
            fname_g   = dir_tree_g + 'ctree.pkl'
            isfile = os.path.isfile(fname_g)

            if(isfile==True):
                with open(fname_g, 'rb') as f:
                    data_g = pickle.load(f)
            else:
                fname_bin   = dir_tree_g + 'ctree.dat'
                data_g    = self.f_gettree_readdat(fname_bin)
                with open(fname_g, 'wb') as f:
                    pickle.dump(data_g, f, pickle.HIGHEST_PROTOCOL)

            self.tree_data_g = data_g

            ## FOR HALOS            
            dir_tree_h  = directory + 'Halo/tree/'

            ## Is pickle?
            fname_h   = dir_tree_h + 'ctree.pkl'
            isfile = os.path.isfile(fname_h)

            if(isfile==True):
                with open(fname_h, 'rb') as f:
                    data_h = pickle.load(f)
            else:
                fname_bin   = dir_tree_h + 'ctree.dat'
                data_h    = self.f_gettree_readdat(fname_bin)
                with open(fname_h, 'wb') as f:
                    pickle.dump(data_h, f, pickle.HIGHEST_PROTOCOL)

            self.tree_data_h = data_h


        # LOAD TREE
        if(horg == 'g'): data = self.tree_data_g
        if(horg == 'h'): data = self.tree_data_h

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
    class f_getevol_p:
        def __init__(self, galdata, idlist, slist, horg):
            self.galdata = galdata
            self.idlist = idlist
            self.slist = slist
            self.horg = horg

        def run(self, start, end, q):
            for i in range(start, end):
                self.galdata[i] = self.vrobj.f_rdgal(self.slist[i], self.idlist[i], horg=self.horg)

            q.put((start, end, self.galdata[start:end]))



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

        print("parallelization should be tested")
        if(n_link < self.num_thread):
            for i in ind:
                gal[i]  = self.f_rdgal(snlist[i], idlist[i], horg=horg)
        else:
            prun = self.f_getevol_p(gal, idlist, snlist, horg)
            prun.vrobj = self

            dind = np.int32(n_link / self.num_thread)
            ps = []
            q = Queue()

            for th in range(self.num_thread):
                i0 = th*dind
                i1 = (th+1)*dind
                if(th==0): i0 = 0
                if(th==self.num_thread-1): i1 = n_link
                p = Process(target=prun.run, args=(i0, i1, q))
                #p.Process(target=prun.run, args=(i0, i1, q))
                ps.append(p)

                p.start()
                while not q.empty():
                    i0, i1, dumdata = q.get()
                    gal[i0:i1] = dumdata
            ok = False
            while not ok:
                ok = True
                for idx in np.arange(len(ps)):
                    if (ps[idx].is_alive()):
                        ok = False
                if(not q.empty()):
                    i0, i1, dumdata = q.get()
                    gal[i0:i1] = dumdata
                else:
                    sleep(0.5)
        return gal

    ##-----
    ## Convert to RamsesSnapshot object
    ##-----
    def f_touri_part(self, snapnum, part, gal=None, center=None, radius=None):

        """
        if(part is None) and (cell is None):
            print('%-----')
            print(' either part or cell must be aruged')
            print('%-----')
            sys.exit(1)
        """

        ##-----
        ## initial settings
        ##-----
        gf  = vr_getftns(self)
        info    = gf.g_info(snapnum)
        ##-----
        ## Object
        ##-----
        snap = uri.RamsesSnapshot(self.dir_raw[:-11], iout=snapnum, mode='none')

        dtype_part   = copy.deepcopy(snap.part_dtype)
        for name in self.vr_fluxlist:
            dtype_part   += [('f_' + name, '<f8')]

        ##-----
        ## Input
        ##-----
        snap.longint    = self.rtype_llint


        arr     = np.zeros(len(part), dtype=dtype_part)
        #return part, arr, snap
        arr['x']    = part['xx'] * 3.086e21 / info['unit_l']
        arr['y']    = part['yy'] * 3.086e21 / info['unit_l']
        arr['z']    = part['zz'] * 3.086e21 / info['unit_l']

        arr['vx']   = part['vx'] / info['kms']
        arr['vy']   = part['vy'] / info['kms']
        arr['vz']   = part['vz'] / info['kms']

        arr['m']    = part['mass']
        #arr['epoch']=?
        arr['metal']= part['metal']
        arr['id']   = part['id']
        arr['cpu']  = part['domain']
        arr['family']   = part['family']
        #arr['tag']      = part['tag']

        for name in self.vr_fluxlist:
            arr['f_' + name]   = part['f_' + name]

        snap.part_data  = arr
        snap.part       = snap.Particle(arr, snap) # required?



        if center is None:
            if not (gal is None):
                center  = np.array([gal['Xc'], gal['Yc'], gal['Zc']]) * 3.086e21 / info['unit_l']
            else:
                center  = np.array([np.median(arr['x']), np.median(arr['y']), np.median(arr['z'])])
        
        if radius is None:
            if not (gal is None):
                radius  = gal['R_HalfMass'] * 3.086e21 / info['unit_l'] * 2.5
            else:
                xmax    = np.amax(np.abs(arr['x'] - center[0]))
                ymax    = np.amax(np.abs(arr['y'] - center[1]))
                zmax    = np.amax(np.abs(arr['z'] - center[2]))
                radius  = np.amax(np.array([xmax, ymax, zmax]))


        radius  = np.array([radius, radius, radius])
        snap.box    = np.stack([center-radius, center+radius], axis=-1)
        snap.box.resize(3,2)


        return snap
       

    def f_touri_cell(self, snapnum, cell, gal=None, center=None, radius=None):


        ##-----
        ## initial settings
        ##-----
        gf  = vr_getftns(self)
        info    = gf.g_info(snapnum)
        ##-----
        ## Object
        ##-----
        snap = uri.RamsesSnapshot(self.dir_raw[:-11], iout=snapnum, mode='none')

        dtype_cell   = [('x', '<f8'), ('y', '<f8'), ('z', '<f8')]
        for name in snap.hydro_names:
            dtype_cell  += [(name, '<f8')]
        dtype_cell  += [('level', '<i4'), ('temp', '<f8')]

        ##-----
        ## Input
        ##-----
        arr     = np.zeros(len(cell), dtype=dtype_cell)
        #return part, arr, snap
        arr['x']    = cell['xx'] * 3.086e21 / info['unit_l']
        arr['y']    = cell['yy'] * 3.086e21 / info['unit_l']
        arr['z']    = cell['zz'] * 3.086e21 / info['unit_l']

        arr['vx']   = cell['vx'] / info['kms']
        arr['vy']   = cell['vy'] / info['kms']
        arr['vz']   = cell['vz'] / info['kms']

        arr['rho']  = cell['den']
        arr['P']    = cell['P_thermal']
        arr['level']= cell['level']
        arr['temp']    = cell['temp']
        #arr['epoch']=?
        #arr['tag']      = part['tag']

        snap.cell_data  = arr
        snap.cell       = snap.Cell(arr, snap) # required?

        #if ~(cell is None):

        if center is None:
            if not (gal is None):
                center  = np.array([gal['Xc'], gal['Yc'], gal['Zc']]) * 3.086e21 / info['unit_l']
            else:
                center  = np.array([np.median(arr['x']), np.median(arr['y']), np.median(arr['z'])])
        
        if radius is None:
            if not (gal is None):
                radius  = gal['R_HalfMass'] * 3.086e21 / info['unit_l'] * 2.5
            else:
                xmax    = np.amax(np.abs(arr['x'] - center[0]))
                ymax    = np.amax(np.abs(arr['y'] - center[1]))
                zmax    = np.amax(np.abs(arr['z'] - center[2]))
                radius  = np.amax(np.array([xmax, ymax, zmax]))


        radius  = np.array([radius, radius, radius])
        snap.box    = np.stack([center-radius, center+radius], axis=-1)
        snap.box.resize(3,2)


        return snap

##-----
## Some basic drawing routines
##-----
class vr_draw:
    def __init__(self, vrobj):
        self.vrobj  = vrobj

    ##-----
    ## Draw Gas map
    ##      Currently aborted
    ##-----
    def d_gasmap(self, n_snap, id0, cell2, amrtype=None, wtype=None, xr=None, yr=None, zr=None, n_pix=None, minlev=None, maxlev=None, proj=None):

        """
        ##----- Settings
        if(n_pix is None): n_pix      = 1000
        if(amrtype is None): amrtype  = 'D'
        if(wtype is None): wtype      = 'MW'
        if(minlev is None): minlev    = self.vrobj.rtype_levmin
        if(maxlev is None): maxlev    = self.vrobj.rtype_levmax
        if(proj is None): proj        = 'xy'

        cell    = cell2[0]
        radius= cell2[1]
        units   = cell2[2]

        ##----- Boxsize
        galtmp  = self.vrobj.f_rdgal(n_snap, id0, horg='g')
        if(xr is None): xr  = np.array([-1, 1.],dtype='<f8') * radius + galtmp['Xc']
        if(yr is None): yr  = np.array([-1, 1.],dtype='<f8') * radius + galtmp['Yc']
        if(zr is None): zr  = np.array([-1, 1.],dtype='<f8') * radius + galtmp['Zc']

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
        js_gasmap_py.js_gasmap_free()

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
        """
##-----
## Call FTNs
##-----
class vr_getftns:
    def __init__(self, vrobj):
        self.vrobj      = vrobj
        self.pot_bsize  = np.int32(1024)    # leaf bucket size for potential calculation

    ##-----
    ## Get Sim info
    ##-----
    def g_info(self, snapnum):
        fname   = self.vrobj.dir_raw + "output_%0.5d"%snapnum + "/info_%0.5d"%snapnum + ".txt"
       
        fdata1  = np.loadtxt(fname, dtype=object, max_rows=6, delimiter='=')
        fdata2  = np.loadtxt(fname, dtype=object, skiprows=7, max_rows=11, delimiter='=')

        info    = {'ncpu':0, 'ndim':0, 'levmin':0, 'levmax':0, 'aexp':0, 
                'H0':0, 'oM':0, 'oB':0, 'oL':0, 'unit_l':0, 'unit_d':0, 'unit_T2':0, 'nH':0, 
                'unit_t':0, 'kms':0, 'unit_m':0, 'hindex':0}

        info['ncpu']    = np.int32(fdata1[0][1])
        info['ndim']    = np.int32(fdata1[1][1])
        info['levmin']  = np.int32(fdata1[2][1])
        info['levmax']  = np.int32(fdata1[3][1])
        info['aexp']    = np.double(fdata2[2][1])
        info['H0']      = np.double(fdata2[3][1])
        info['oM']      = np.double(fdata2[4][1])
        info['oL']      = np.double(fdata2[5][1])
        info['oB']      = np.double(fdata2[7][1])
        info['unit_l']  = np.double(fdata2[8][1])
        info['unit_d']  = np.double(fdata2[9][1])
        info['unit_t']  = np.double(fdata2[10][1])
        info['unit_T2'] = np.double(1.66e-24) / np.double(1.3806200e-16) * np.double(info['unit_l'] / info['unit_t'])**2
        info['nH']      = np.double(0.76) / np.double(1.66e-24) * info['unit_d']
        info['kms']     = info['unit_l'] / info['unit_t'] / 1e5
        info['unit_m']  = info['unit_d'] * info['unit_l']**3

        info['hindex']  = np.double(np.loadtxt(fname, dtype=object, skiprows=21)[:,1:])
        return info
    ##-----
    ## Get Domain enclosing the shell
    ##  Domain list enclosing the input sphere
    ##      xc, yc, zc, rr as in kpc unit
    ##  
    ##      xc, yc, zc, rr can be a numpy array
    ##-----
    def g_domain(self, snapnum, xc, yc, zc, rr, info=None):
        if(info is None): info = self.g_info(snapnum)

        xc2  = np.array(xc / info['unit_l'] * 3.086e21, dtype='<f8')
        yc2  = np.array(yc / info['unit_l'] * 3.086e21, dtype='<f8')
        zc2  = np.array(zc / info['unit_l'] * 3.086e21, dtype='<f8')
        rr2  = np.array(rr / info['unit_l'] * 3.086e21, dtype='<f8')
        hindex  = info['hindex']

        larr    = np.zeros(20, dtype=np.int32)
        darr    = np.zeros(20, dtype='<f8')

        larr[0] = np.int32(np.size(xc2))
        larr[1] = np.int32(info['ncpu'])
        larr[2] = np.int32(self.vrobj.num_thread)
        larr[3] = np.int32(info['levmax'])

        darr[0] = np.double(1.0)

        find_domain_py.find_domain(xc2, yc2, zc2, rr2, hindex, larr, darr)
        domlist = find_domain_py.dom_list
        domlist = domlist[0][:]
        domlist = np.int32(np.array(np.where(domlist > 0))[0] + 1)

        find_domain_py.find_domain_free()
        return domlist

    ##-----
    ## Get Particle within the domain enclosing the shell
    ##      1) xc, yc, zc, rr [kpc]
    ##          single value or a numpy array
    ##      2) if 'domlist' argued, the given shell (xc, yc, zc, rr) is negelected
    ##      3) Negative rr gives all ptcls
    ##-----
    def g_ptcl(self, snapnum, xc, yc, zc, rr, domlist=None, num_thread=None, info=None, 
        p_age=False, p_flux=False, p_sink=False):

        # Initial settings
        if(info is None): info = self.g_info(snapnum)
        if(num_thread is None): num_thread = self.vrobj.num_thread

        # Get Domain
        if(domlist is None):
            if(rr<0): domlist = np.int32(np.arange(1, info['ncpu']+1))
            else:
                domlist = self.g_domain(snapnum, xc, yc, zc, rr, info=info)
        else:
            if(~isinstance(domlist, np.ndarray)): domlist = np.int32(np.array(domlist))

            if(np.amin(domlist) < 1 or np.amax(domlist) > info['ncpu']):
                print('%-----')
                print(' Wrongly argued')
                print('     out of range of the domain: domlist')
                print('%-----')
                sys.exit(1)

        fname   = self.vrobj.dir_raw + "output_%0.5d"%snapnum + "/part_%0.5d"%snapnum + ".out"

        # Get total particle numbers and omp index
        larr        = np.zeros(20, dtype=np.int32)
        darr        = np.zeros(20, dtype='<f8')

        larr[0]     = np.int32(len(domlist))
        larr[2]     = np.int32(num_thread)
        larr[3]     = np.int32(len(fname))


        jsrd_part_totnum_py.jsrd_part_totnum(larr, darr, fname, domlist)
        npart_tot   = jsrd_part_totnum_py.npart_tot
        part_ind    = np.array(jsrd_part_totnum_py.part_ind, dtype=np.int32)

        # Allocate
        #dtype   = [('xx', '<f8'), ('yy', '<f8'), ('zz', '<f8'),
        #    ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
        #    ('mass', '<f8'), ('sfact', '<f8'), ('gyr', '<f8'), ('metal', '<f8'), ('family', np.int32) , ('domain', np.int32), 
        #    ('id', np.int64)]

        #for name in self.vrobj.vr_fluxlist:
        #    dtype   += [('f_' + name, '<f8')]
        dtype   = self.vrobj.dtype_part

        ptcl    = np.zeros(np.int32(npart_tot), dtype=dtype)

        # Read Ptcls
        larr        = np.zeros(20, dtype=np.int32)
        darr        = np.zeros(20, dtype='<f8')

        larr[0]     = np.int32(len(domlist))
        larr[2]     = np.int32(num_thread)
        larr[3]     = np.int32(len(fname))
        larr[4]     = np.int32(npart_tot)

        if(self.vrobj.rtype_family == True): larr[18] = 20
        else: larr[18] = 0

        if(self.vrobj.rtype_llint == True): larr[19] = 20
        else: larr[19] = 0

        jsrd_part_py.jsrd_part(larr, darr, fname, part_ind, domlist)

        # Input
        ptcl['xx'][:]   = jsrd_part_py.xx * info['unit_l'] / 3.086e21
        ptcl['yy'][:]   = jsrd_part_py.yy * info['unit_l'] / 3.086e21
        ptcl['zz'][:]   = jsrd_part_py.zz * info['unit_l'] / 3.086e21

        ptcl['vx'][:]   = jsrd_part_py.vx * info['kms']
        ptcl['vy'][:]   = jsrd_part_py.vy * info['kms']
        ptcl['vz'][:]   = jsrd_part_py.vz * info['kms']

        ptcl['mass'][:] = jsrd_part_py.mp * info['unit_m'] / 1.98892e33
        #ptcl['ap'][:]   = jsrd_part_py.ap
        ptcl['metal'][:]= jsrd_part_py.zp

        ptcl['family'][:]   = jsrd_part_py.fam
        ptcl['domain'][:]   = jsrd_part_py.domain

        ptcl['id'][:]       = jsrd_part_py.idvar

        

        ##----- COMPUTE GYR
        if(p_age==True):
            gyr = self.g_gyr(snapnum, ptcl['ap'][:])
            ptcl['gyr'][:]  = gyr['gyr'][:]
            ptcl['sfact'][:]= gyr['sfact'][:]

        ##----- COMPUTE FLUX
        if(p_flux==True):
            for name in self.vrobj.vr_fluxlist:
                ptcl['f_' + name][:] = self.g_flux(ptcl['mass'][:], ptcl['metal'][:], ptcl['gyr'][:],name)[name]

        jsrd_part_totnum_py.jsrd_part_totnum_free()
        jsrd_part_py.jsrd_part_free()
        return ptcl
                
    ##-----
    ## Get AMR within the domain enclosing the shell
    ##      1) xc, yc, zc, rr [kpc]
    ##          single value or a numpy array
    ##      2) if 'domlist' argued, the given shell (xc, yc, zc, rr) is negelected
    ##      3) Negative rr gives all cells
    ##-----
    def g_amr(self, snapnum, xc, yc, zc, vxc, vyc, vzc, rr, domlist=None, num_thread=None, info=None, amrtype=False):

        ##----- Settings
        if(num_thread is None): num_thread = self.vrobj.num_thread
        if(domlist is None):
            if(rr<0): domlist = np.int32(np.arange(1, info['ncpu']+1))
            else:
                domlist = self.g_domain(snapnum, xc, yc, zc, rr, info=info)
        else:
            if not (isinstance(domlist, np.ndarray)): domlist = np.int32(np.array(domlist))

            if(np.amin(domlist) < 1 or np.amax(domlist) > info['ncpu']):
                print('%-----')
                print(' Wrongly argued')
                print('     out of range of the domain: domlist')
                print('%-----')
                sys.exit(1)

        if(info is None):info = self.g_info(snapnum)

        xr  = np.array([-1, 1.],dtype='<f8') * rr + xc
        yr  = np.array([-1, 1.],dtype='<f8') * rr + yc
        zr  = np.array([-1, 1.],dtype='<f8') * rr + zc

        #----- READ AMR (Get Total number of leaf cells)
        larr    = np.zeros(20, dtype=np.int32)
        darr    = np.zeros(20, dtype='<f8')

        file_a  = self.vrobj.dir_raw + 'output_%0.5d'%snapnum + '/amr_%0.5d'%snapnum + '.out'
        file_h  = self.vrobj.dir_raw + 'output_%0.5d'%snapnum + '/hydro_%0.5d'%snapnum + '.out'
        file_i  = self.vrobj.dir_raw + 'output_%0.5d'%snapnum + '/info_%0.5d'%snapnum + '.txt'

        larr[0] = np.int32(len(domlist))
        larr[2] = np.int32(1)#np.int32(num_thread)
        larr[3] = np.int32(len(file_a))
        larr[4] = np.int32(len(file_h))
        larr[5] = np.int32(self.vrobj.rtype_ndomain)
        larr[6] = np.int32(self.vrobj.rtype_ndim)
        larr[7] = np.int32(self.vrobj.rtype_levmin)
        larr[8] = np.int32(self.vrobj.rtype_levmax)


        jsamr2cell_totnum_py.jsamr2cell_totnum(larr, darr, file_a, file_h, domlist)
        ntot    = jsamr2cell_totnum_py.ntot
        nvarh   = jsamr2cell_totnum_py.nvarh
        mg_ind  = jsamr2cell_totnum_py.mg_ind
        

        #----- READ AMR (ALLOCATE)
        data    = np.zeros(ntot, dtype=self.vrobj.dtype_amr)
                
        #        [('xx','<f8'), ('yy','<f8'), ('zz','<f8'),
        #    ('vx','<f8'), ('vy','<f8'), ('vz','<f8'), ('dx','<f8'), ('mass', '<f8'),
        #    ('type',np.int32), ('PE','<f8'), ('KE','<f8'), ('UE','<f8'), 
        #    ('den','<f8'), ('temp','<f8'), ('P_thermal','<f8'), ('metal','<f8'), ('level',np.int32)])

        ##----- READ AMR
        larr    = np.zeros(20, dtype=np.int32)
        darr    = np.zeros(20, dtype='<f8')

        larr[0] = np.int32(len(domlist))
        larr[2] = np.int32(num_thread)
        larr[3] = np.int32(len(file_a))
        larr[4] = np.int32(len(file_h))
        larr[5] = np.int32(len(file_i))
        larr[6] = np.int32(self.vrobj.rtype_ndomain)
        larr[7] = np.int32(self.vrobj.rtype_ndim)
        larr[8] = np.int32(self.vrobj.rtype_levmin)
        larr[9] = np.int32(self.vrobj.rtype_levmax)
        larr[10]= np.int32(ntot)
        larr[11]= np.int32(nvarh)

        jsamr2cell_py.jsamr2cell(larr, darr, file_a, file_h, file_i, mg_ind, domlist)
        xgdum   = np.array(jsamr2cell_py.mesh_xg,dtype='<f8')
        hvdum   = np.array(jsamr2cell_py.mesh_hd,dtype='<f8')
        dxdum   = np.array(jsamr2cell_py.mesh_dx,dtype='<f8')
        lvdum   = np.array(jsamr2cell_py.mesh_lv,dtype='int32')

        data['xx'][:]   = xgdum[:,0] * info['unit_l'] / np.double(3.086e21)
        data['yy'][:]   = xgdum[:,1] * info['unit_l'] / np.double(3.086e21)
        data['zz'][:]   = xgdum[:,2] * info['unit_l'] / np.double(3.086e21)
        data['vx'][:]   = hvdum[:,1] * info['kms']
        data['vy'][:]   = hvdum[:,2] * info['kms']
        data['vz'][:]   = hvdum[:,3] * info['kms']
        data['den'][:]  = hvdum[:,0]
        data['temp'][:] = hvdum[:,4]
        data['metal'][:]= hvdum[:,5]
        data['dx'][:]   = dxdum[:] * info['unit_l'] / np.double(3.086e21)
        data['level'][:]= lvdum[:]
        #data['mass'][:] = np.double(10.**(np.log10(hvdum[:,0]) + np.log10(unit_d) + np.double(3.0) * (np.log10(dxdum[:]) + np.log10(unit_l)) - np.log10(1.98892e33)))
        data['mass'][:] = hvdum[:,0] * info['unit_d'] * (dxdum[:] * info['unit_l'])**3 / np.double(1.98892e33)

        data    = data[np.where(lvdum >= 0)]
        dumA = data['temp'][:]
        dumB = data['den'][:]

        data['temp'][:] = data['temp'][:] / data['den'][:] * info['unit_T2']    # [K/mu]
        data['den'][:]  *= info['nH']                                           # [atom/cc]
        data['P_thermal'][:]    = dumA * info['unit_m'] / info['unit_l'] / info['unit_t']**2 / np.double(1.3806200e-16)

        units   = {'unit_l':info['unit_l'], 'unit_m':info['unit_m'], 'unit_t':info['unit_t'], 'kms':info['kms'], 'unit_nH':info['nH'], 'unit_T2':info['unit_T2']}

        jsamr2cell_totnum_py.jsamr2cell_totnum_free()
        jsamr2cell_py.jsamr2cell_free()

        if(amrtype == True):
            celltype    = self.g_celltype(snapnum, data, xc, yc, zc, vxc, vyc, vzc, rr, domlist=domlist, info=info, num_thread=num_thread)
            data['type']    = celltype[0]
            data['PE']    = celltype[1]
            data['KE']    = celltype[2]
            data['UE']    = celltype[3]
        return data
        
    ##-----
    ## Get AMR related properties
    ##  Compute following using cells within "rr X rfact"
    ## 
    ##      - Particles within rr X 5 are used to compute potential
    ##-----
    def g_celltype(self, snapnum, cell, xc, yc, zc, vxc, vyc, vzc, rr, domlist=None, info=None, num_thread=None, n_shell=None):

        ##----- Settings
        if(num_thread is None): num_thread = self.vrobj.num_thread
        if(domlist is None):
            if(rr<0): domlist = np.int32(np.arange(1, info['ncpu']+1))
            else:
                domlist = self.g_domain(snapnum, xc, yc, zc, rr, info=info)
        else:
            if not (isinstance(domlist, np.ndarray)): domlist = np.int32(np.array(domlist))

            if(np.amin(domlist) < 1 or np.amax(domlist) > info['ncpu']):
                print('%-----')
                print(' Wrongly argued')
                print('     out of range of the domain: domlist')
                print('%-----')
                sys.exit(1)

        if(info is None):info = self.g_info(snapnum)
        if(n_shell is None): n_shell = 100
        d_shell = rr / n_shell

        ##----- Extract Cell properties
        cell_m  = cell['mass']
        cell_met= cell['metal'] / self.vrobj.sun_met

        # Read Particles and Extract star & DM within the sphere
        ptcl    = self.g_ptcl(snapnum, xc, yc, zc, rr, 
            domlist=domlist, num_thread=num_thread, info=info)

        d3d     = np.sqrt((ptcl['xx']-xc)**2 + (ptcl['yy']-yc)**2 + (ptcl['zz']-zc)**2)
        ptcl    = ptcl[np.logical_and( d3d < rr, np.logical_or(ptcl['family']==1, ptcl['family']==2))]
        if len(ptcl) == 0:
                print('%-----')
                print(' Warning (g_celltype)')
                print('     No ptcls within the domain')
                print('%-----')
        
        # Create new arrays for potential calculations
        npart   = len(ptcl) + len(cell)
        #pos     = np.zeros((npart,3),dtype='<f8')
        #pos[:,0]    = np.concatenate((ptcl['xx'],cell['xx']))
        #pos[:,1]    = np.concatenate((ptcl['yy'],cell['yy']))
        #pos[:,2]    = np.concatenate((ptcl['zz'],cell['zz']))

        pos     = np.concatenate((ptcl['xx'], cell['xx'], ptcl['yy'], cell['yy'], ptcl['zz'], cell['zz']), dtype='<f8')
        pos     = np.resize(pos,(3,npart)).T
        pot_m   = np.concatenate((ptcl['mass'], cell['mass']))
        
        # Reset the bucket size
        bsize   = self.pot_bsize
        if(npart < 1e6):bsize = np.int32(512)
        if(npart < 5e5):bsize = np.int32(256)
        if(npart < 1e5):bsize = np.int32(128)
        if(npart < 5e4):bsize = np.int32(64)
        if(npart < 1e4):bsize = np.int32(32)
        if(npart < 5e3):bsize = np.int32(16)
        if(npart < 1e3):bsize = np.int32(8)

        # Potential calculation
        pot     = kinematics.f_getpot(pos, pot_m, num_thread=self.vrobj.num_thread, bsize=bsize)
        pot     = pot[len(ptcl):npart]

        # Kinetic E & Thermal E
        KE      = (cell['vx'] - vxc)**2 + (cell['vy'] - vyc)**2 + (cell['vz'] - vzc)**2
        KE      = KE * 0.5      # [km/s]^2

        UE      = cell['temp'] / (5./3. - 1.) / (1.66e-24) * 1.380649e-23 * 1e-3    # [km/s]^2
        E_tot   = KE + UE + pot

        # Cell type
        #      1  - ISM
        #      0  - CGM
        #      -1 - IGM
        cell_type   = np.zeros(len(cell),dtype=np.int32) - 1
        cell_d3d    = np.sqrt((cell['xx']-xc)**2 + (cell['yy']-yc)**2 + (cell['zz']-zc)**2)

        E_zero      = 0.
        Mcut        = 1.
        minZval     = 0.1 # Lowest metallicity for CGM (in Zsun)

        # ISM (bound gas)
        cell_type[E_tot < E_zero] = 1

        # Compute ISM Metallicity
        ism_met     = np.zeros((n_shell, 2), dtype='<f8')
        ism_met[:,0]= 1e8
        ism_met[:,1]= 0.

        for i in range(0,n_shell):
            r0 = d_shell * i
            r1 = d_shell * (i+1.)

            ind     = (cell_d3d >= r0) * (cell_d3d < r1) * (cell_type == 1)
            if sum(ind) == 0: continue

            ism_met[i,0] = np.average(cell_met[ind], weights=cell_m[ind])
            # for weighted std
            ism_met[i,1] = np.sqrt( np.sum( cell_m[ind] * (cell_met[ind] - ism_met[i,0])**2 )   / np.sum(cell_m[ind]))

        # Extrapolate
        for i in range(1,n_shell):
            if( ism_met[i,0] > 1e7 ):
                ism_met[i,0]    = ism_met[i-1,0]
                ism_met[i,1]    = ism_met[i-1,1]


        # CGM as
        #   1) E_tot > 0
        #   2) Z > Z_ism - dZ_ism
        #   3) Z > Z_minval
        #   4) outflowing ( dot(r,v) > 0 )
        #   5) Hot gas (T > 1e7)
        
        vdot    = (cell['xx'] - xc) * (cell['vx'] - vxc) + (cell['yy'] - yc) * (cell['vy'] - vyc) + (cell['zz'] - zc) * (cell['vz'] - vzc)

        for i in range(0,n_shell):
            r0 = d_shell * i
            r1 = d_shell * (i+1.)

            ind     = (cell_d3d >= r0) * (cell_d3d < r1) * (cell_type == -1)
            if sum(ind) == 0: continue

            met_avg = np.average(cell_met[ind], weights=cell_m[ind])
            lowZval = np.amax([ism_met[i,0]-Mcut*ism_met[i,1], minZval])

            ind2    = ind * np.logical_or(vdot > 0., cell['temp']>1e7) * (cell_met > lowZval)
            if sum(ind2) == 0: continue

            cell_type[ind2] = 0
            

        # The others are IGM

        return cell_type, pot, KE, UE


    ##-----
    ## Get AMR related properties
    ##  Compute following using cells within the shell
    ##-----
    def g_gasall(self, snapnum, xc, yc, zc, vxc, vyc, vzc, radius, info=None, domlist=None, cell=None, num_thread=None):

        
        # initial settings
        if(num_thread is None): num_thread = self.vrobj.num_thread
        xr, yr, zr = np.zeros(2, dtype='<f8'), np.zeros(2, dtype='<f8'), np.zeros(2, dtype='<f8')
        xr[0], xr[1] = xc - radius, xc + radius
        yr[0], yr[1] = yc - radius, yc + radius
        zr[0], zr[1] = zc - radius, zc + radius

        # Get Sim props
        if(info is None): info = self.g_info(snapnum)
        if(domlist is None): domlist = self.g_domain(snapnum, xc, yc, zc, radius, info)
        if(cell is None):
            cell = self.g_amr(snapnum, xc, yc, zc, vxc, vyc, vzc, radius, domlist=domlist, info=info, amrtype=True)
        
        # Compute AMR type (ISM / CGM / IGM)
        #celltype    = (self.g_celltype(snapnum, cell, xc, yc, zc, vxc, vyc, vzc, radius, domlist=domlist, info=info, num_thread=num_thread))[0]

        # FROM HERE 123123
        return cell


    ##-----
    ## Get img
    ##-----
    def g_img(self, gasmap, zmin=None, zmax=None, scale=None):

        if(zmin is None): zmin = np.min(gasmap)
        if(zmax is None): zmax = np.max(gasmap)

        ind    = np.where(gasmap < zmin)
        if(np.size(ind)>0): gasmap[ind] = 0

        ind     = np.where(gasmap > 0)
        gasmap[ind] -= zmin

        ind = np.where(gasmap > (zmax - zmin))
        if(np.size(ind)>0): gasmap[ind] = zmax - zmin

        gasmap  /= (zmax - zmin)

        if(scale=='log' or scale is None):
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

            get_flux_py.get_flux_free()
        return data

    ##-----
    ## HaloMaker matching
    ##-----
    def g_HMmatch(self, snapnum, g_vr, g_hm, info=None):
        if info is None: info = self.g_info(snapnum)

        # convert to code unit
        xc_vr   = g_vr['Xc'] * 3.086e21 / info['unit_l']
        yc_vr   = g_vr['Yc'] * 3.086e21 / info['unit_l']
        zc_vr   = g_vr['Zc'] * 3.086e21 / info['unit_l']

        #distance
        d3d = np.sqrt(
                (xc_vr - g_hm['x'])**2 + (yc_vr - g_hm['y'])**2 + (zc_vr - g_hm['z'])**2)

        return g_hm[d3d == np.amin(d3d)]

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
