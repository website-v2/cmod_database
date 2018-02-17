# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:26:33 2018

@author: mathewsa
"""

#run 4 efits (33,65,129,257), retrieve the stored value, then plot for single 
#time slice as function of ngrid to compare convergence of these different efits

import MDSplus 
from MDSplus import *
import math
import os
import subprocess 
import numpy as np

#This code takes shot, tstart, dt, and tend as input,
#and will write to trees efit06 or efit07 if they are available


#function to run efit in mode 10
#efit_ is the tree (e.g. efit06 or efit07)
#shot is the shot number, tstart, tend, and dt are in milliseconds
#overwrites efit06 tree only; ensure the tree is free for the shot!****

shot = 1160803022
number = 1 #number of burst (<20)
tstart = 801.0
tend = 1100.0
dt = 1.0
ntimes = 300 
timebase = np.arange(round(tstart,3),round(tend+1.,3),dt)
#this results in finding the time from 1.001s - 1.300s at 1 millisecond resolution

filepath = "efit_input.txt"

def smooth(y,box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y,box,mode='same')
    return y_smooth
    
def array_psiRZ_r(array1):
    i = 0  
    app = []
    while i < len(array1[40]):
        app.append(array1[40][int((len(array1[40]))/2.)][i]) #halfway point in z over all r 
        i = i + 1
    return app
    
def array_psiRZ_z(array2):
    i = 0  
    app = []
    while i < len(array2[40]):
        app.append(array2[40][i][[int((len(array2[40]))/2.)]]) #halfway point in r ove all z
        i = i + 1
    return app
    
def array_r(array):
    i = 0  
    app = []
    while i < len(array):
        app.append(array[i][40]) #40 corresponds to 40th time slice out of 300 time slices
        i = i + 1  
    return app

with open("{}".format(filepath), 'w') as file_handler: 
    tree = Tree('efit06',-1)
    tree.createPulse(shot)
    file_handler.write('10\n') 
    file_handler.write('efit06\n') #overwriting efit06 tree only!
    file_handler.write('\n') 
    file_handler.write("{}, -{}\n".format(shot,number)) #the -{#} appends {#} overlapping bursts 
    file_handler.write("{},{},{}\n".format(tstart,dt,ntimes))  
    file_handler.close() 
    os.system("/usr/local/cmod/codes/efit/bin/fast_efitd129d < efit_input.txt > efit_output.txt") #129x129  
    os.remove("efit_input.txt")
    os.remove("efit_output.txt")
    
    magnetics = MDSplus.Tree('magnetics', shot)
    ip = magnetics.getNode('\ip').data()
    time_ip = magnetics.getNode('\ip').dim_of().data() 
    
    efit = MDSplus.Tree('efit06', shot)
    time_efit129 = (efit.getNode('\efit_aeqdsk:time')).data()
    beta_N129 = (efit.getNode('\efit_aeqdsk:betan')).data(); #normalized beta 
    beta_p129 = (efit.getNode('\efit_aeqdsk:betap')).data(); #beta_poloidal  
    beta_t129 = (efit.getNode('\efit_aeqdsk:betat')).data(); #beta_toroidal 
    kappa129 = (efit.getNode('\efit_aeqdsk:eout')).data(); #elongation - vertical 
    triang_l129 = (efit.getNode('\efit_aeqdsk:doutl')).data(); #lower triangularity of lcfs 
    triang_u129 = (efit.getNode('\efit_aeqdsk:doutu')).data(); #upper triangularity of lcfs 
    triang129 = (triang_u129 + triang_l129)/2. #overall triangularity - horizontal (major radius)
    li129 = (efit.getNode('\efit_aeqdsk:li')).data(); #internal inductance 
    areao129 = (efit.getNode('\efit_aeqdsk:areao')).data(); #area of lcfs 
    vout129 = (efit.getNode('\efit_aeqdsk:vout')).data(); #volume of lcfs 
    aout129 = (efit.getNode('\efit_aeqdsk:aout')).data(); #minor radius of lcfs 
    rout129 = (efit.getNode('\efit_aeqdsk:rout')).data(); #major radius of geometric center 
    zmag129 = (efit.getNode('\efit_aeqdsk:zmagx')).data(); #z of magnetic axis 
    zout129 = (efit.getNode('\efit_aeqdsk:zout')).data(); #z of lcfs (constructed) 
    zseps129 = (efit.getNode('\efit_aeqdsk:zseps')).data(); #z of upper and lower xpts
    zsep_lower129 = zseps129[:,0]
    zsep_upper129 = zseps129[:,1]  
    rGrid129 = (efit.getNode('\efit_g_eqdsk:rGrid')).data()
    zGrid129 = (efit.getNode('\efit_g_eqdsk:zGrid')).data()
    psiRZ129 = (efit.getNode('\efit_g_eqdsk:psiRZ')).data() #psi not normalized
    psiRZ129_r = array_psiRZ_r(psiRZ129)    
    psiRZ129_z = array_psiRZ_z(psiRZ129) 
    qpsi129 = (efit.getNode('\efit_g_eqdsk:qpsi')).data() #array of q, safety factor, on flux surface psi
    qpsi129 = array_r(qpsi129) 
    pres_flux129 = (efit.getNode('\efit_g_eqdsk:pres')).data() #array of pressure on flux surface psi
    pres_flux129 = array_r(pres_flux129)    
    zvsin129 = (efit.getNode('\efit_aeqdsk:zvsin')).data(); #z of inner strike point 
    zvsout129 = (efit.getNode('\efit_aeqdsk:zvsout')).data(); #z of outer strike point 
    upper_gap129 = (efit.getNode('\efit_aeqdsk:otop')).data()/100.; # meters 
    lower_gap129 = (efit.getNode('\efit_aeqdsk:obott')).data()/100.; # meters 
    q0129 = (efit.getNode('\efit_aeqdsk:q0')).data(); #safety factor at center 
    qstar129 = (efit.getNode('\efit_aeqdsk:qstar')).data(); #cylindrical safety factor 
    q95129 = (efit.getNode('\efit_aeqdsk:q95')).data(); #edge safety factor 
    V_loop_efit129 = (efit.getNode('\efit_aeqdsk:vloopt')).data(); #loop voltage 
    V_surf_efit129 = (efit.getNode('\efit_aeqdsk:vsurfa')).data(); #surface voltage 
    Wmhd129 = (efit.getNode('\efit_aeqdsk:wplasm')).data(); #diamagnetic/stored energy, [J] 
    ssep129 = (efit.getNode('\efit_aeqdsk:ssep')).data()/100.; # distance on midplane between 1st and 2nd separatrices [m]
    n_over_ncrit129 = (efit.getNode('\efit_aeqdsk:xnnc')).data(); #vertical stability criterion (EFIT name: xnnc) 
    inductance129 = 4.*np.pi*1.E-7 * 0.68 * li129/2.; # For simplicity, we use R0 = 0.68 m, but we could use \efit_aeqdsk:rmagx
#    dipdt = np.gradient(ip,timebase)
#    dipdt_smoothed = smooth(dipdt,11) #11-point smoothing (moving average box/by convolution)
#    V_inductive129 = inductance129*dipdt_smoothed
#    V_resistive129 = V_loop_efit129 - V_inductive129
#    P_ohm = ip*V_resistive129  
    
with open("{}".format(filepath), 'w') as file_handler: 
    tree = Tree('efit06',-1)
    tree.createPulse(shot)
    file_handler.write('10\n') 
    file_handler.write('efit06\n') #overwriting efit06 tree only!
    file_handler.write('\n') 
    file_handler.write("{}, -{}\n".format(shot,number)) #the -{#} appends {#} overlapping bursts 
    file_handler.write("{},{},{}\n".format(tstart,dt,ntimes))  
    file_handler.close() 
    os.system("/usr/local/cmod/codes/efit/bin/fast_efitdd < efit_input.txt > efit_output.txt") #33x33
    os.remove("efit_input.txt")
    os.remove("efit_output.txt")
    
    magnetics = MDSplus.Tree('magnetics', shot)
    ip = magnetics.getNode('\ip').data()
    time_ip = magnetics.getNode('\ip').dim_of().data() 
    
    efit = MDSplus.Tree('efit06', shot)
    time_efit33 = (efit.getNode('\efit_aeqdsk:time')).data()
    beta_N33 = (efit.getNode('\efit_aeqdsk:betan')).data(); #normalized beta 
    beta_p33 = (efit.getNode('\efit_aeqdsk:betap')).data(); #beta_poloidal  
    beta_t33 = (efit.getNode('\efit_aeqdsk:betat')).data(); #beta_toroidal 
    kappa33 = (efit.getNode('\efit_aeqdsk:eout')).data(); #elongation - vertical 
    triang_l33 = (efit.getNode('\efit_aeqdsk:doutl')).data(); #lower triangularity of lcfs 
    triang_u33 = (efit.getNode('\efit_aeqdsk:doutu')).data(); #upper triangularity of lcfs 
    triang33 = (triang_u33 + triang_l33)/2. #overall triangularity - horizontal (major radius)
    li33 = (efit.getNode('\efit_aeqdsk:li')).data(); #internal inductance 
    areao33 = (efit.getNode('\efit_aeqdsk:areao')).data(); #area of lcfs 
    vout33 = (efit.getNode('\efit_aeqdsk:vout')).data(); #volume of lcfs 
    aout33 = (efit.getNode('\efit_aeqdsk:aout')).data(); #minor radius of lcfs 
    rout33 = (efit.getNode('\efit_aeqdsk:rout')).data(); #major radius of geometric center 
    zmag33 = (efit.getNode('\efit_aeqdsk:zmagx')).data(); #z of magnetic axis 
    zout33 = (efit.getNode('\efit_aeqdsk:zout')).data(); #z of lcfs (constructed) 
    zseps33 = (efit.getNode('\efit_aeqdsk:zseps')).data(); #z of upper and lower xpts
    zsep_lower33 = zseps33[:,0]
    zsep_upper33 = zseps33[:,1]  
    rGrid33 = (efit.getNode('\efit_g_eqdsk:rGrid')).data()
    zGrid33 = (efit.getNode('\efit_g_eqdsk:zGrid')).data()
    psiRZ33 = (efit.getNode('\efit_g_eqdsk:psiRZ')).data() #psi not normalized
    psiRZ33_r = array_psiRZ_r(psiRZ33)  
    psiRZ33_z = array_psiRZ_z(psiRZ33) 
    qpsi33 = (efit.getNode('\efit_g_eqdsk:qpsi')).data() #array of q, safety factor, on flux surface psi
    qpsi33 = array_r(qpsi33) 
    pres_flux33 = (efit.getNode('\efit_g_eqdsk:pres')).data() #array of pressure on flux surface psi
    pres_flux33 = array_r(pres_flux33)
    zvsin33 = (efit.getNode('\efit_aeqdsk:zvsin')).data(); #z of inner strike point 
    zvsout33 = (efit.getNode('\efit_aeqdsk:zvsout')).data(); #z of outer strike point 
    upper_gap33 = (efit.getNode('\efit_aeqdsk:otop')).data()/100.; # meters 
    lower_gap33 = (efit.getNode('\efit_aeqdsk:obott')).data()/100.; # meters 
    q033 = (efit.getNode('\efit_aeqdsk:q0')).data(); #safety factor at center 
    qstar33 = (efit.getNode('\efit_aeqdsk:qstar')).data(); #cylindrical safety factor 
    q9533 = (efit.getNode('\efit_aeqdsk:q95')).data(); #edge safety factor 
    V_loop_efit33 = (efit.getNode('\efit_aeqdsk:vloopt')).data(); #loop voltage 
    V_surf_efit33 = (efit.getNode('\efit_aeqdsk:vsurfa')).data(); #surface voltage 
    Wmhd33 = (efit.getNode('\efit_aeqdsk:wplasm')).data(); #diamagnetic/stored energy, [J] 
    ssep33 = (efit.getNode('\efit_aeqdsk:ssep')).data()/100.; # distance on midplane between 1st and 2nd separatrices [m]
    n_over_ncrit33 = (efit.getNode('\efit_aeqdsk:xnnc')).data(); #vertical stability criterion (EFIT name: xnnc) 
    inductance33 = 4.*np.pi*1.E-7 * 0.68 * li33/2.; # For simplicity, we use R0 = 0.68 m, but we could use \efit_aeqdsk:rmagx
#    dipdt = np.gradient(ip,timebase)
#    dipdt_smoothed = smooth(dipdt,11) #11-point smoothing (moving average box/by convolution)
#    V_inductive33 = inductance33*dipdt_smoothed
#    V_resistive33 = V_loop_efit33 - V_inductive33
#    P_ohm = ip*V_resistive33
    

with open("{}".format(filepath), 'w') as file_handler:
    tree = Tree('efit06',-1)
    tree.createPulse(shot)
    file_handler.write('10\n') 
    file_handler.write('efit06\n') #overwriting efit06 tree only!
    file_handler.write('\n') 
    file_handler.write("{}, -{}\n".format(shot,number)) #the -{#} appends {#} overlapping bursts 
    file_handler.write("{},{},{}\n".format(tstart,dt,ntimes))  
    file_handler.close() 
    os.system("/usr/local/cmod/codes/efit/bin/efitd6565d < efit_input.txt > efit_output.txt") #65x65
    os.remove("efit_input.txt")
    os.remove("efit_output.txt")
    magnetics = MDSplus.Tree('magnetics', shot)
    ip = magnetics.getNode('\ip').data()
    time_ip = magnetics.getNode('\ip').dim_of().data() 
    efit = MDSplus.Tree('efit06', shot)
    time_efit65 = (efit.getNode('\efit_aeqdsk:time')).data()
    beta_N65 = (efit.getNode('\efit_aeqdsk:betan')).data(); #normalized beta 
    beta_p65 = (efit.getNode('\efit_aeqdsk:betap')).data(); #beta_poloidal  
    beta_t65 = (efit.getNode('\efit_aeqdsk:betat')).data(); #beta_toroidal 
    kappa65 = (efit.getNode('\efit_aeqdsk:eout')).data(); #elongation - vertical 
    triang_l65 = (efit.getNode('\efit_aeqdsk:doutl')).data(); #lower triangularity of lcfs 
    triang_u65 = (efit.getNode('\efit_aeqdsk:doutu')).data(); #upper triangularity of lcfs 
    triang65 = (triang_u65 + triang_l65)/2. #overall triangularity - horizontal (major radius)
    li65 = (efit.getNode('\efit_aeqdsk:li')).data(); #internal inductance 
    areao65 = (efit.getNode('\efit_aeqdsk:areao')).data(); #area of lcfs 
    vout65 = (efit.getNode('\efit_aeqdsk:vout')).data(); #volume of lcfs 
    aout65 = (efit.getNode('\efit_aeqdsk:aout')).data(); #minor radius of lcfs 
    rout65 = (efit.getNode('\efit_aeqdsk:rout')).data(); #major radius of geometric center 
    zmag65 = (efit.getNode('\efit_aeqdsk:zmagx')).data(); #z of magnetic axis 
    zout65 = (efit.getNode('\efit_aeqdsk:zout')).data(); #z of lcfs (constructed) 
    zseps65 = (efit.getNode('\efit_aeqdsk:zseps')).data(); #z of upper and lower xpts
    zsep_lower65 = zseps65[:,0]
    zsep_upper65 = zseps65[:,1]   
    rGrid65 = (efit.getNode('\efit_g_eqdsk:rGrid')).data()
    zGrid65 = (efit.getNode('\efit_g_eqdsk:zGrid')).data()
    psiRZ65 = (efit.getNode('\efit_g_eqdsk:psiRZ')).data() #psi not normalized
    psiRZ65_r = array_psiRZ_r(psiRZ65)    
    psiRZ65_z = array_psiRZ_z(psiRZ65) 
    qpsi65 = (efit.getNode('\efit_g_eqdsk:qpsi')).data() #array of q, safety factor, on flux surface psi
    qpsi65 = array_r(qpsi65) 
    pres_flux65 = (efit.getNode('\efit_g_eqdsk:pres')).data() #array of pressure on flux surface psi
    pres_flux65 = array_r(pres_flux65)
    zvsin65 = (efit.getNode('\efit_aeqdsk:zvsin')).data(); #z of inner strike point 
    zvsout65 = (efit.getNode('\efit_aeqdsk:zvsout')).data(); #z of outer strike point 
    upper_gap65 = (efit.getNode('\efit_aeqdsk:otop')).data()/100.; # meters 
    lower_gap65 = (efit.getNode('\efit_aeqdsk:obott')).data()/100.; # meters 
    q065 = (efit.getNode('\efit_aeqdsk:q0')).data(); #safety factor at center 
    qstar65 = (efit.getNode('\efit_aeqdsk:qstar')).data(); #cylindrical safety factor 
    q9565 = (efit.getNode('\efit_aeqdsk:q95')).data(); #edge safety factor 
    V_loop_efit65 = (efit.getNode('\efit_aeqdsk:vloopt')).data(); #loop voltage 
    V_surf_efit65 = (efit.getNode('\efit_aeqdsk:vsurfa')).data(); #surface voltage 
    Wmhd65 = (efit.getNode('\efit_aeqdsk:wplasm')).data(); #diamagnetic/stored energy, [J] 
    ssep65 = (efit.getNode('\efit_aeqdsk:ssep')).data()/100.; # distance on midplane between 1st and 2nd separatrices [m]
    n_over_ncrit65 = (efit.getNode('\efit_aeqdsk:xnnc')).data(); #vertical stability criterion (EFIT name: xnnc) 
    inductance65 = 4.*np.pi*1.E-7 * 0.68 * li65/2.; # For simplicity, we use R0 = 0.68 m, but we could use \efit_aeqdsk:rmagx
#    dipdt = np.gradient(ip,timebase)
#    dipdt_smoothed = smooth(dipdt,11) #11-point smoothing (moving average box/by convolution)
#    V_inductive65 = inductance65*dipdt_smoothed
#    V_resistive65 = V_loop_efit65 - V_inductive65
#    P_ohm = ip*V_resistive65


with open("{}".format(filepath), 'w') as file_handler: 
    tree = Tree('efit06',-1)
    tree.createPulse(shot)
    file_handler.write('10\n') 
    file_handler.write('efit06\n') #overwriting efit06 tree only!
    file_handler.write('\n') 
    file_handler.write("{}, -{}\n".format(shot,number)) #the -{#} appends {#} overlapping bursts 
    file_handler.write("{},{},{}\n".format(tstart,dt,ntimes))  
    file_handler.close() 
    os.system("/usr/local/cmod/codes/efit/bin/efitd257d < efit_input.txt > efit_output.txt") #257x257
    os.remove("efit_input.txt")
    os.remove("efit_output.txt")
    
    magnetics = MDSplus.Tree('magnetics', shot)
    ip = magnetics.getNode('\ip').data()
    time_ip = magnetics.getNode('\ip').dim_of().data() 
    
    efit = MDSplus.Tree('efit06', shot)
    time_efit257 = (efit.getNode('\efit_aeqdsk:time')).data()
    beta_N257 = (efit.getNode('\efit_aeqdsk:betan')).data(); #normalized beta 
    beta_p257 = (efit.getNode('\efit_aeqdsk:betap')).data(); #beta_poloidal  
    beta_t257 = (efit.getNode('\efit_aeqdsk:betat')).data(); #beta_toroidal 
    kappa257 = (efit.getNode('\efit_aeqdsk:eout')).data(); #elongation - vertical 
    triang_l257 = (efit.getNode('\efit_aeqdsk:doutl')).data(); #lower triangularity of lcfs 
    triang_u257 = (efit.getNode('\efit_aeqdsk:doutu')).data(); #upper triangularity of lcfs 
    triang257 = (triang_u257 + triang_l257)/2. #overall triangularity - horizontal (major radius)
    li257 = (efit.getNode('\efit_aeqdsk:li')).data(); #internal inductance 
    areao257 = (efit.getNode('\efit_aeqdsk:areao')).data(); #area of lcfs 
    vout257 = (efit.getNode('\efit_aeqdsk:vout')).data(); #volume of lcfs 
    aout257 = (efit.getNode('\efit_aeqdsk:aout')).data(); #minor radius of lcfs 
    rout257 = (efit.getNode('\efit_aeqdsk:rout')).data(); #major radius of geometric center 
    zmag257 = (efit.getNode('\efit_aeqdsk:zmagx')).data(); #z of magnetic axis 
    zout257 = (efit.getNode('\efit_aeqdsk:zout')).data(); #z of lcfs (constructed) 
    zseps257 = (efit.getNode('\efit_aeqdsk:zseps')).data(); #z of upper and lower xpts
    zsep_lower257 = zseps257[:,0]
    zsep_upper257 = zseps257[:,1]   
    rGrid257 = (efit.getNode('\efit_g_eqdsk:rGrid')).data()
    zGrid257 = (efit.getNode('\efit_g_eqdsk:zGrid')).data()
    psiRZ257 = (efit.getNode('\efit_g_eqdsk:psiRZ')).data() #psi not normalized
    psiRZ257_r = array_psiRZ_r(psiRZ257)    
    psiRZ257_z = array_psiRZ_z(psiRZ257)  
    qpsi257 = (efit.getNode('\efit_g_eqdsk:qpsi')).data() #array of q, safety factor, on flux surface psi
    qpsi257 = array_r(qpsi257) 
    pres_flux257 = (efit.getNode('\efit_g_eqdsk:pres')).data() #array of pressure on flux surface psi
    pres_flux257 = array_r(pres_flux257)
    zvsin257 = (efit.getNode('\efit_aeqdsk:zvsin')).data(); #z of inner strike point 
    zvsout257 = (efit.getNode('\efit_aeqdsk:zvsout')).data(); #z of outer strike point 
    upper_gap257 = (efit.getNode('\efit_aeqdsk:otop')).data()/100.; # meters 
    lower_gap257 = (efit.getNode('\efit_aeqdsk:obott')).data()/100.; # meters 
    q0257 = (efit.getNode('\efit_aeqdsk:q0')).data(); #safety factor at center 
    qstar257 = (efit.getNode('\efit_aeqdsk:qstar')).data(); #cylindrical safety factor 
    q95257 = (efit.getNode('\efit_aeqdsk:q95')).data(); #edge safety factor 
    V_loop_efit257 = (efit.getNode('\efit_aeqdsk:vloopt')).data(); #loop voltage 
    V_surf_efit257 = (efit.getNode('\efit_aeqdsk:vsurfa')).data(); #surface voltage 
    Wmhd257 = (efit.getNode('\efit_aeqdsk:wplasm')).data(); #diamagnetic/stored energy, [J] 
    ssep257 = (efit.getNode('\efit_aeqdsk:ssep')).data()/100.; # distance on midplane between 1st and 2nd separatrices [m]
    n_over_ncrit257 = (efit.getNode('\efit_aeqdsk:xnnc')).data(); #vertical stability criterion (EFIT name: xnnc) 
    inductance257 = 4.*np.pi*1.E-7 * 0.68 * li257/2.; # For simplicity, we use R0 = 0.68 m, but we could use \efit_aeqdsk:rmagx
#    dipdt = np.gradient(ip,timebase)
#    dipdt_smoothed = smooth(dipdt,11) #11-point smoothing (moving average box/by convolution)
#    V_inductive257 = inductance257*dipdt_smoothed
#    V_resistive257 = V_loop_efit257 - V_inductive257
#    P_ohm = ip*V_resistive257
    
#with open("{}".format(filepath), 'w') as file_handler: 
#    tree = Tree('efit06',-1)
#    tree.createPulse(shot)
#    file_handler.write('10\n') 
#    file_handler.write('efit06\n') #overwriting efit06 tree only!
#    file_handler.write('\n') 
#    file_handler.write("{}, -{}\n".format(shot,number)) #the -{#} appends {#} overlapping bursts 
#    file_handler.write("{},{},{}\n".format(tstart,dt,ntimes))  
#    file_handler.close() 
#    os.system("/usr/local/cmod/codes/efit/bin/efitd513d < efit_input.txt > efit_output.txt") #513x513 
#    os.remove("efit_input.txt")
#    os.remove("efit_output.txt")
#    
#    magnetics = MDSplus.Tree('magnetics', shot)
#    ip = magnetics.getNode('\ip').data()
#    time_ip = magnetics.getNode('\ip').dim_of().data() 
#    
#    efit = MDSplus.Tree('efit06', shot)
#    time_efit513 = (efit.getNode('\efit_aeqdsk:time')).data()
#    beta_N513 = (efit.getNode('\efit_aeqdsk:betan')).data(); #normalized beta 
#    beta_p513 = (efit.getNode('\efit_aeqdsk:betap')).data(); #beta_poloidal  
#    beta_t513 = (efit.getNode('\efit_aeqdsk:betat')).data(); #beta_toroidal 
#    kappa513 = (efit.getNode('\efit_aeqdsk:eout')).data(); #elongation - vertical 
#    triang_l513 = (efit.getNode('\efit_aeqdsk:doutl')).data(); #lower triangularity of lcfs 
#    triang_u513 = (efit.getNode('\efit_aeqdsk:doutu')).data(); #upper triangularity of lcfs 
#    triang513 = (triang_u513 + triang_l513)/2. #overall triangularity - horizontal (major radius)
#    li513 = (efit.getNode('\efit_aeqdsk:li')).data(); #internal inductance 
#    areao513 = (efit.getNode('\efit_aeqdsk:areao')).data(); #area of lcfs 
#    vout513 = (efit.getNode('\efit_aeqdsk:vout')).data(); #volume of lcfs 
#    aout513 = (efit.getNode('\efit_aeqdsk:aout')).data(); #minor radius of lcfs 
#    rout513 = (efit.getNode('\efit_aeqdsk:rout')).data(); #major radius of geometric center 
#    zmag513 = (efit.getNode('\efit_aeqdsk:zmagx')).data(); #z of magnetic axis 
#    zout513 = (efit.getNode('\efit_aeqdsk:zout')).data(); #z of lcfs (constructed) 
#    zseps513 = (efit.getNode('\efit_aeqdsk:zseps')).data(); #z of upper and lower xpts
#    zsep_lower513 = zseps513[:,0]
#    zsep_upper513 = zseps513[:,1]   
#    zvsin513 = (efit.getNode('\efit_aeqdsk:zvsin')).data(); #z of inner strike point 
#    zvsout513 = (efit.getNode('\efit_aeqdsk:zvsout')).data(); #z of outer strike point 
#    upper_gap513 = (efit.getNode('\efit_aeqdsk:otop')).data()/100.; # meters 
#    lower_gap513 = (efit.getNode('\efit_aeqdsk:obott')).data()/100.; # meters 
#    q0513 = (efit.getNode('\efit_aeqdsk:q0')).data(); #safety factor at center 
#    qstar513 = (efit.getNode('\efit_aeqdsk:qstar')).data(); #cylindrical safety factor 
#    q95513 = (efit.getNode('\efit_aeqdsk:q95')).data(); #edge safety factor 
#    V_loop_efit513 = (efit.getNode('\efit_aeqdsk:vloopt')).data(); #loop voltage 
#    V_surf_efit513 = (efit.getNode('\efit_aeqdsk:vsurfa')).data(); #surface voltage 
#    Wmhd513 = (efit.getNode('\efit_aeqdsk:wplasm')).data(); #diamagnetic/stored energy, [J] 
#    ssep513 = (efit.getNode('\efit_aeqdsk:ssep')).data()/100.; # distance on midplane between 1st and 2nd separatrices [m]
#    n_over_ncrit513 = (efit.getNode('\efit_aeqdsk:xnnc')).data(); #vertical stability criterion (EFIT name: xnnc) 
#    inductance513 = 4.*np.pi*1.E-7 * 0.68 * li513/2.; # For simplicity, we use R0 = 0.68 m, but we could use \efit_aeqdsk:rmagx
#    dipdt = np.gradient(ip,timebase)
#    dipdt_smoothed = smooth(dipdt,11) #11-point smoothing (moving average box/by convolution)
#    V_inductive513 = inductance513*dipdt_smoothed
#    V_resistive513 = V_loop_efit513 - V_inductive513
##    P_ohm = ip*V_resistive513 
    
import matplotlib.pyplot as plt 

fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(time_efit129,q0129,c='b',marker="^",ls='--',label='129x129',fillstyle='none')
ax.plot(time_efit129,q033,c='g',marker=(8,2,0),ls='--',label='33x33')
ax.plot(time_efit129,q065,c='k',ls='-',label='65x65')
ax.plot(time_efit129,q0257,c='r',marker="v",ls='-',label='257x257')
#ax.plot(time_efit129,x**2-1,c='m',marker="o",ls='--',label='BSwap',fillstyle='none')
#ax.plot(time_efit129,x-1,c='k',marker="+",ls=':',label='MSD')
fig.savefig('q0.png', dpi=fig.dpi)
plt.legend(loc=2)
plt.draw()
plt.close()

fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(time_efit129,beta_N129,c='b',marker="^",ls='--',label='129x129',fillstyle='none')
ax.plot(time_efit129,beta_N33,c='g',marker=(8,2,0),ls='--',label='33x33')
ax.plot(time_efit129,beta_N65,c='k',ls='-',label='65x65')
ax.plot(time_efit129,beta_N257,c='r',marker="v",ls='-',label='257x257') 
fig.savefig('beta_N.png', dpi=fig.dpi)
plt.legend(loc=2)
plt.draw()
plt.close() 

fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(time_efit129,V_loop_efit129,c='b',marker="^",ls='--',label='129x129',fillstyle='none')
ax.plot(time_efit129,V_loop_efit33,c='g',marker=(8,2,0),ls='--',label='33x33')
ax.plot(time_efit129,V_loop_efit65,c='k',ls='-',label='65x65')
ax.plot(time_efit129,V_loop_efit257,c='r',marker="v",ls='-',label='257x257') 
fig.savefig('V_loop_efit.png', dpi=fig.dpi)
plt.legend(loc=2)
plt.draw()
plt.close() 

fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(time_efit129,inductance129,c='b',marker="^",ls='--',label='129x129',fillstyle='none')
ax.plot(time_efit129,inductance33,c='g',marker=(8,2,0),ls='--',label='33x33')
ax.plot(time_efit129,inductance65,c='k',ls='-',label='65x65')
ax.plot(time_efit129,inductance257,c='r',marker="v",ls='-',label='257x257') 
fig.savefig('inductance.png', dpi=fig.dpi)
plt.legend(loc=2)
plt.draw()
plt.close() 

fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(time_efit129,triang129,c='b',marker="^",ls='--',label='129x129',fillstyle='none')
ax.plot(time_efit129,triang33,c='g',marker=(8,2,0),ls='--',label='33x33')
ax.plot(time_efit129,triang65,c='k',ls='-',label='65x65')
ax.plot(time_efit129,triang257,c='r',marker="v",ls='-',label='257x257') 
fig.savefig('triangularity.png', dpi=fig.dpi)
plt.legend(loc=2)
plt.draw()
plt.close() 

fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(time_efit129,q95129,c='b',marker="^",ls='--',label='129x129',fillstyle='none')
ax.plot(time_efit129,q9533,c='g',marker=(8,2,0),ls='--',label='33x33')
ax.plot(time_efit129,q9565,c='k',ls='-',label='65x65')
ax.plot(time_efit129,q95257,c='r',marker="v",ls='-',label='257x257') 
fig.savefig('q95.png', dpi=fig.dpi)
plt.legend(loc=2)
plt.draw()
plt.close() 

fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(time_efit129,ssep129,c='b',marker="^",ls='--',label='129x129',fillstyle='none')
ax.plot(time_efit129,ssep33,c='g',marker=(8,2,0),ls='--',label='33x33')
ax.plot(time_efit129,ssep65,c='k',ls='-',label='65x65')
ax.plot(time_efit129,ssep257,c='r',marker="v",ls='-',label='257x257') 
fig.savefig('ssep.png', dpi=fig.dpi)
plt.legend(loc=2)
plt.draw()
plt.close() 

fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(time_efit129,Wmhd129,c='b',marker="^",ls='--',label='129x129',fillstyle='none')
ax.plot(time_efit129,Wmhd33,c='g',marker=(8,2,0),ls='--',label='33x33')
ax.plot(time_efit129,Wmhd65,c='k',ls='-',label='65x65')
ax.plot(time_efit129,Wmhd257,c='r',marker="v",ls='-',label='257x257') 
fig.savefig('Wmhd.png', dpi=fig.dpi)
plt.legend(loc=2)
plt.draw()
plt.close()   

fig=plt.figure() 
ax=fig.add_subplot(111) 
ax.plot(rGrid129,psiRZ129_r,c='b',marker=(8,2,0),ls='--',label='129x129') 
ax.plot(rGrid33,psiRZ33_r,c='g',marker=(8,2,0),ls='--',label='33x33') 
ax.plot(rGrid65,psiRZ65_r,c='k',marker=(8,2,0),ls='--',label='65x65') 
ax.plot(rGrid257,psiRZ257_r,c='r',marker=(8,2,0),ls='--',label='257x257') 
plt.legend(loc=2)
plt.show()
fig.savefig('psiRZ vs r.png', dpi=fig.dpi)
plt.close()    

fig=plt.figure() 
ax=fig.add_subplot(111) 
ax.plot(zGrid129,psiRZ129_z,c='b',marker=(8,2,0),ls='--',label='129x129') 
ax.plot(zGrid33,psiRZ33_z,c='g',marker=(8,2,0),ls='--',label='33x33') 
ax.plot(zGrid65,psiRZ65_z,c='k',marker=(8,2,0),ls='--',label='65x65') 
ax.plot(zGrid257,psiRZ257_z,c='r',marker=(8,2,0),ls='--',label='257x257') 
plt.legend(loc=2)
plt.show()
fig.savefig('psiRZ vs z.png', dpi=fig.dpi)
plt.close()     

fig=plt.figure() 
ax=fig.add_subplot(111) 
ax.plot(rGrid129,pres_flux129,c='b',marker=(8,2,0),ls='--',label='129x129') 
ax.plot(rGrid33,pres_flux33,c='g',marker=(8,2,0),ls='--',label='33x33') 
ax.plot(rGrid65,pres_flux65,c='k',marker=(8,2,0),ls='--',label='65x65') 
ax.plot(rGrid257,pres_flux257,c='r',marker=(8,2,0),ls='--',label='257x257') 
plt.legend(loc=2)
plt.show()
fig.savefig('pressure_flux vs r.png', dpi=fig.dpi)
plt.close()    

fig=plt.figure() 
ax=fig.add_subplot(111) 
ax.plot(rGrid129,qpsi129,c='b',marker=(8,2,0),ls='--',label='129x129') 
ax.plot(rGrid33,qpsi33,c='g',marker=(8,2,0),ls='--',label='33x33') 
ax.plot(rGrid65,qpsi65,c='k',marker=(8,2,0),ls='--',label='65x65') 
ax.plot(rGrid257,qpsi257,c='r',marker=(8,2,0),ls='--',label='257x257') 
plt.legend(loc=2)
plt.show()
fig.savefig('safety factor vs r.png', dpi=fig.dpi)
plt.close() 