import numpy as np

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
        else:
            print('%-----')
            print(' Wrong argument for the simtype')
            print('     Simtype list: NH, NH2, FORNAX(or FN), NC(not yet)')
            print('%-----')



