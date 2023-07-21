import numpy as np

import rur
from rur import uri, drawer, utool, uhmi, sci

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

kpc = 3.0857e21
msun = 1.989e33
Zsun = 0.0134
Lsun = 3.826e33

k_boltzmann = k_b = 1.38064852*1e-23 ## m^2 kg s^-2 K^-1
m_H = 1.6735575*1e-27 ## kg 

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def rotation(array1, array2, theta, type='rad'):
    if type == 'deg':
        theta *= np.pi/ 180
    new_array1 = np.cos(theta) * array1 - np.sin(theta) * array2
    new_array2 = np.sin(theta) * array1 + np.cos(theta) * array2
    return new_array1, new_array2

def Ang_Rot( 
            M,
            X,Y,Z,
            VX,VY,VZ,
            age,
            R_upp,R_low=0,
            A_upp=8,
            mass_weight=True,
            par_mode='median',
            Param=True
            ):
    
    ######### rotation ###########

    if Param == True:
        ang_x = (Y*VZ - Z*VY)
        ang_y = (Z*VX - X*VZ)
        ang_z = (X*VY - Y*VX)
        if mass_weight==True:
            ang_x *= M
            ang_y *= M
            ang_z *= M
        
        rotsort = (R_low<np.sqrt(np.square(X)+np.square(Y)+np.square(Z)))*\
                  (np.sqrt(np.square(X)+np.square(Y)+np.square(Z))<R_upp)*\
                  (age<A_upp)
        
        if par_mode == 'median' or par_mode == 'Median' or par_mode == 'MEDIAN':
            param = [np.median(ang_x[rotsort]),np.median(ang_y[rotsort]),np.median(ang_z[rotsort])]
        elif par_mode == 'mean' or par_mode == 'Mean' or par_mode == 'MEAN':
            param = [np.mean(ang_x[rotsort]),np.mean(ang_y[rotsort]),np.mean(ang_z[rotsort])]
        else:
            print('Please choose "mean" or "median".')
        
    else:
        param = Param

    a, b, c = param[0], param[1], param[2]
    ab = np.sqrt(np.square(a)+np.square(b))
    
    Tht_xy = -1 * np.arctan2(a,b)
    Tht_yz = -1 * np.arctan2(ab,c)
    
    y_mid,x_new = rotation(Y,X,Tht_xy)
    z_new,y_new = rotation(Z,y_mid,Tht_yz)

    vy_mid,vx_new = rotation(VY,VX,Tht_xy)
    vz_new,vy_new = rotation(VZ,vy_mid,Tht_yz)

    return  x_new, y_new, z_new,\
            vx_new,vy_new,vz_new, param


#!!!!!!!

def make_input_rur(
                   snap,
                   star,
                   cell,
                   pos_ctr,
                   vel_ctr,
                   co_moving=False,
                   
                   # gal rotation
                   rotating_gal=True,
                   R_upp=8,
                   R_low=2,
                   A_upp=8,
                   mass_weight=True,
                   par_mode='median',
    
                   # Stellar population
                   dx_s = 50, #[pc]
                   sig_s = 10, #[km/s]
    
                   # Old population
                   old_fbase = './',
                   old_fname = 'part_old.txt',
    
                   # Young star selection criteria
                   add_young=False,
                    
                   young_fbase = './',
                   young_fname = 'part_young.txt',
    
                   Age_Cut = 0.01, #[Myr]
                   SED_type_y='mappings',
                   
                   # Using additional columns?
                   import_velocity='true',
                   import_dispersion='false',

                   # Gas cells 
                   gas_fbase = './',
                   gas_fname = 'gas_cell.txt',
    
                   fact_dx = 1.5,
                   sig_c = 5, #[km/s]
    
                   import_velocity_gas='true',
                   import_dispersion_gas='false'

                   ):
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    boxtokpc = 1 /snap.unit['kpc']
    if co_moving == True:
        boxtokpc /= snap.params['aexp']
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    x_s = (star['x'] -pos_ctr[0]) *boxtokpc
    y_s = (star['y'] -pos_ctr[1]) *boxtokpc
    z_s = (star['z'] -pos_ctr[2]) *boxtokpc
    
    vx_s = 1e4*star['vx'] -vel_ctr[0]
    vy_s = 1e4*star['vy'] -vel_ctr[1]
    vz_s = 1e4*star['vz'] -vel_ctr[2]

    m_s = star['m'] *snap.params['unit_m'] /msun
    age_s = star['age','Gyr']
    metal_s = star['metal']
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    x_c = (cell['x'] -pos_ctr[0]) *boxtokpc
    y_c = (cell['y'] -pos_ctr[1]) *boxtokpc
    z_c = (cell['z'] -pos_ctr[2]) *boxtokpc
    
    vx_c = 1e4*cell['vx'] -vel_ctr[0]
    vy_c = 1e4*cell['vy'] -vel_ctr[1]
    vz_c = 1e4*cell['vz'] -vel_ctr[2]

    dx_c = 1 /(2**cell['level']) *boxtokpc
    m_c = cell['m'] *snap.params['unit_m'] /msun
    T_c = cell['T','K']
    metal_c = cell['metal']
    
    print("done -- basic setting")
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if rotating_gal == True:
        x_s, y_s, z_s, vx_s, vy_s, vz_s, par_rot =  Ang_Rot( 
                                                            M=m_s,
                                                            X=x_s,Y=y_s,Z=z_s,
                                                            VX=vx_s,VY=vy_s,VZ=vz_s,
                                                            age=age_s,
                                                            R_upp=R_upp,R_low=R_low,
                                                            A_upp=A_upp,
                                                            mass_weight=mass_weight,
                                                            par_mode=par_mode,
                                                            Param=True
                                                            )               
        x_c, y_c, z_c, vx_c, vy_c, vz_c, par_rot =  Ang_Rot( 
                                                            M=m_c,
                                                            X=x_c,Y=y_c,Z=z_c,
                                                            VX=vx_c,VY=vy_c,VZ=vz_c,
                                                            age=x_c,
                                                            R_upp=R_upp,R_low=R_low,
                                                            A_upp=A_upp,
                                                            mass_weight=mass_weight,
                                                            par_mode=par_mode,
                                                            Param=par_rot
                                                            )    
        
        print("done -- galaxy rotation")
        
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #!!! For old stellar population
    
    x_s_ind, y_s_ind, z_s_ind = 0, 1, 2
    dx_s_ind = 3 
    m_s_ind, metal_s_ind, age_s_ind = 4, 5, 6
    
    old_arr_len = 7
    if import_velocity == 'true':
        old_arr_len += 3
        vx_s_ind, vy_s_ind, vz_s_ind = 4, 5, 6
        m_s_ind, metal_s_ind, age_s_ind = 7, 8, 9

    if import_dispersion == 'true':
        old_arr_len += 1
        sig_s_ind = 7
        m_s_ind, metal_s_ind, age_s_ind = 8, 9, 10
        
    if add_young == True:
        sort = (Age_Cut<=age_s)
    else:
        sort = (0<age_s)
        
    old_arr = np.empty((len(x_s[sort]), old_arr_len))

    old_arr[:,x_s_ind] = x_s[sort]
    old_arr[:,y_s_ind] = y_s[sort]
    old_arr[:,z_s_ind] = z_s[sort]
    old_arr[:,dx_s_ind] = np.ones(len(x_s[sort]))*dx_s
    if import_velocity == 'true':
        old_arr[:,vx_s_ind] = vx_s[sort]
        old_arr[:,vy_s_ind] = vy_s[sort]
        old_arr[:,vz_s_ind] = vz_s[sort]
    if import_dispersion == 'true':
        old_arr[:,sig_s_ind] = np.ones(len(x_s[sort]))*sig_s
    old_arr[:,m_s_ind] = np.ones(len(x_s[sort]))*12750
    old_arr[:,metal_s_ind] = metal_s[sort]
    old_arr[:,age_s_ind] = age_s[sort]

    header = "Column %s: position x (kpc)\n" %(x_s_ind+1)
    header += "Column %s: position y (kpc)\n" %(y_s_ind+1)
    header += "Column %s: position z (kpc)\n" %(z_s_ind+1)
    header += "Column %s: smoothing length (pc)\n" %(dx_s_ind+1)
    if import_velocity == 'true':
        header += "Column %s: velocity x (km/s)\n" %(vx_s_ind+1)
        header += "Column %s: velocity y (km/s)\n" %(vy_s_ind+1)
        header += "Column %s: velocity z (km/s)\n" %(vz_s_ind+1)
    if import_dispersion == 'true':
        header += "Column %s: velocity dispersion (km/s)\n" %(sig_s_ind+1)
    header += "Column %s: mass (Msun)\n" %(m_s_ind+1)
    header += "Column %s: metallicity (1)\n" %(metal_s_ind+1)
    header += "Column %s: age (Gyr)\n" %(age_s_ind+1)
    
    np.savetxt(
        old_fbase+old_fname,
        old_arr, delimiter=' ', newline='\n', header=header, footer='', comments='# ', encoding=None)
    
    print("done -- writing old population")
    #!!! For young stellar population
    if add_young == True:
        if SED_type_y == 'BC03':
            sort = (Age_Cut>age_s)
            young_arr = np.empty((len(x_s[sort]), old_arr_len))

            young_arr[:,x_s_ind] = x_s[sort]
            young_arr[:,y_s_ind] = y_s[sort]
            young_arr[:,z_s_ind] = z_s[sort]
            young_arr[:,dx_s_ind] = np.ones(len(x_s[sort]))*dx_s
            if import_velocity == 'true':
                young_arr[:,vx_s_ind] = vx_s[sort]
                young_arr[:,vy_s_ind] = vy_s[sort]
                young_arr[:,vz_s_ind] = vz_s[sort]
            if import_dispersion == 'true':
                young_arr[:,sig_s_ind] = np.ones(len(x_s[sort]))*sig_s
            young_arr[:,m_s_ind] = np.ones(len(x_s[sort]))*12750
            young_arr[:,metal_s_ind] = metal_s[sort]
            young_arr[:,age_s_ind] = age_s[sort]

            header = "Column %s: position x (kpc)\n" %(x_s_ind+1)
            header += "Column %s: position y (kpc)\n" %(y_s_ind+1)
            header += "Column %s: position z (kpc)\n" %(z_s_ind+1)
            header += "Column %s: smoothing length (pc)\n" %(dx_s_ind+1)
            if import_velocity == 'true':
                header += "Column %s: velocity x (km/s)\n" %(vx_s_ind+1)
                header += "Column %s: velocity y (km/s)\n" %(vy_s_ind+1)
                header += "Column %s: velocity z (km/s)\n" %(vz_s_ind+1)
            if import_dispersion == 'true':
                header += "Column %s: velocity dispersion (km/s)\n" %(sig_s_ind+1)
            header += "Column %s: mass (Msun)\n" %(m_s_ind+1)
            header += "Column %s: metallicity (1)\n" %(metal_s_ind+1)
            header += "Column %s: age (Gyr)\n" %(age_s_ind+1)

            np.savetxt(
                young_fbase+young_fname,
                young_arr, delimiter=' ', newline='\n', header=header, footer='', comments='# ', encoding=None)
            
            
        elif SED_type_y == 'mappings':
            #!!! linear regression for the parameter space. 
            #!!! If you want to apply your model, try.
            a_prs, b_prs, c_prs = 0.4, 10, -8.86
            a_cpt, b_cpt, c_cpt = 0.25, 10, 6.5
            t_clr = 0.003 #[Gyr]
            
            prs_s = 10**(a_prs*(age_s-b_prs)+c_prs)
            cpt_s = a_cpt*(age_s-b_cpt)+c_cpt
            fpdr_s = np.exp(-age_s/t_clr)
            
            #!!!!!
            
            x_s_ind, y_s_ind, z_s_ind = 0, 1, 2
            dx_s_ind = 3 
            sfr_s_ind, metal_s_ind, cpt_s_ind, prs_s_ind, fpdr_s_ind = 4, 5, 6, 7, 8

            young_arr_len = 9
            if import_velocity == 'true':
                young_arr_len += 3
                vx_s_ind, vy_s_ind, vz_s_ind = 4, 5, 6
                sfr_s_ind, metal_s_ind, cpt_s_ind, prs_s_ind, fpdr_s_ind = 7, 8, 9, 10, 11

            if import_dispersion == 'true':
                young_arr_len += 1
                sig_s_ind = 7
                sfr_s_ind, metal_s_ind, cpt_s_ind, prs_s_ind, fpdr_s_ind = 8, 9, 10, 11, 12
            
            #!!!!!
            
            

            sort = (Age_Cut>age_s)
            young_arr = np.empty((len(x_s[sort]), young_arr_len))

            young_arr[:,x_s_ind] = x_s[sort]
            young_arr[:,y_s_ind] = y_s[sort]
            young_arr[:,z_s_ind] = z_s[sort]
            young_arr[:,dx_s_ind] = np.ones(len(x_s[sort]))*dx_s
            if import_velocity == 'true':
                young_arr[:,vx_s_ind] = vx_s[sort]
                young_arr[:,vy_s_ind] = vy_s[sort]
                young_arr[:,vz_s_ind] = vz_s[sort]
            if import_dispersion == 'true':
                young_arr[:,sig_s_ind] = np.ones(len(x_s[sort]))*sig_s
            young_arr[:,sfr_s_ind] = np.ones(len(x_s[sort]))*12750/(age_s[sort]*1e9)
            young_arr[:,metal_s_ind] = metal_s[sort]
            young_arr[:,cpt_s_ind] = cpt_s[sort]
            young_arr[:,prs_s_ind] = prs_s[sort]
            young_arr[:,fpdr_s_ind] = fpdr_s[sort]

            header = "Column %s: position x (kpc)\n" %(x_s_ind+1)
            header += "Column %s: position y (kpc)\n" %(y_s_ind+1)
            header += "Column %s: position z (kpc)\n" %(z_s_ind+1)
            header += "Column %s: smoothing length (pc)\n" %(dx_s_ind+1)
            if import_velocity == 'true':
                header += "Column %s: velocity x (km/s)\n" %(vx_s_ind+1)
                header += "Column %s: velocity y (km/s)\n" %(vy_s_ind+1)
                header += "Column %s: velocity z (km/s)\n" %(vz_s_ind+1)
            if import_dispersion == 'true':
                header += "Column %s: velocity dispersion (km/s)\n" %(sig_s_ind+1)
            header += "Column %s: star formation rate (Msun/yr)\n" %(sfr_s_ind+1)
            header += "Column %s: metallicity (1)\n" %(metal_s_ind+1)
            header += "Column %s: compactness (1)\n" %(cpt_s_ind+1)
            header += "Column %s: pressure (Pa)\n" %(prs_s_ind+1)
            header += "Column %s: covering factor (1)\n" %(fpdr_s_ind+1)

            np.savetxt(
                young_fbase+young_fname,
                young_arr, delimiter=' ', newline='\n', header=header, footer='', comments='# ', encoding=None)
    
        print("done -- writing young population")
    
    #!!! For gas cells
    x_c_ind, y_c_ind, z_c_ind = 0, 1, 2
    dx_c_ind = 3 
    m_c_ind, metal_c_ind, T_c_ind = 4, 5, 6
    
    gas_arr_len = 7
    if import_velocity_gas == 'true':
        gas_arr_len += 3
        vx_c_ind, vy_c_ind, vz_c_ind = 7, 8, 9

    if import_dispersion_gas == 'true':
        gas_arr_len += 1
        sig_c_ind = 10
        
    gas_arr = np.empty((len(x_c), gas_arr_len))

    gas_arr[:,x_c_ind] = x_c
    gas_arr[:,y_c_ind] = y_c
    gas_arr[:,z_c_ind] = z_c
    gas_arr[:,dx_c_ind] = dx_c *fact_dx
    if import_velocity_gas == 'true':
        gas_arr[:,vx_c_ind] = vx_c
        gas_arr[:,vy_c_ind] = vy_c
        gas_arr[:,vz_c_ind] = vz_c
    if import_dispersion_gas == 'true':
        gas_arr[:,sig_c_ind] = np.ones(len(x_c))*sig_c
    gas_arr[:,m_c_ind] = m_c
    gas_arr[:,metal_c_ind] = metal_c
    gas_arr[:,T_c_ind] = T_c

    header = "Column %s: position x (kpc)\n" %(x_c_ind+1)
    header += "Column %s: position y (kpc)\n" %(y_c_ind+1)
    header += "Column %s: position z (kpc)\n" %(z_c_ind+1)
    header += "Column %s: smoothing length (kpc)\n" %(dx_c_ind+1)
    header += "Column %s: mass (Msun)\n" %(m_c_ind+1)
    header += "Column %s: metallicity (1)\n" %(metal_c_ind+1)
    header += "Column %s: Temperature (K)\n" %(T_c_ind+1)
    if import_velocity_gas == 'true':
        header += "Column %s: velocity x (km/s)\n" %(vx_c_ind+1)
        header += "Column %s: velocity y (km/s)\n" %(vy_c_ind+1)
        header += "Column %s: velocity z (km/s)\n" %(vz_c_ind+1)
    if import_dispersion_gas == 'true':
        header += "Column %s: velocity dispersion (km/s)\n" %(sig_c_ind+1)
    
    np.savetxt(
        gas_fbase+gas_fname,
        gas_arr, delimiter=' ', newline='\n', header=header, footer='', comments='# ', encoding=None)

    print("done -- writing gas cells")
    
    return









