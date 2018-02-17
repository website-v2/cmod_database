# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:47:47 2018

@author: mathewsa
"""

import MDSplus 
from MDSplus import *
import numpy as np   
import os
import sys
from os import getenv 
import matplotlib.pyplot as plt
import idlpy
from idlpy import *

def smooth(y,box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y,box,mode='same')
    return y_smooth
 
shot = 1120824016

magnetics = MDSplus.Tree('magnetics', shot)
ip = magnetics.getNode('\ip').data()
time_ip = magnetics.getNode('\ip').dim_of().data()

tree_efit = 'efit07'

efit = MDSplus.Tree(tree_efit, shot)
IDL.run("good_time_slices = efit_check(shot={},tree='{}')".format(shot,tree_efit)) 
good_time_slices = getattr(IDL,"good_time_slices") #returns indexes for values passing efit_check test
time_efit = (efit.getNode('\efit_aeqdsk:time')).data()[good_time_slices]
beta_N = (efit.getNode('\efit_aeqdsk:betan')).data()[good_time_slices]; #normalized beta 
beta_p = (efit.getNode('\efit_aeqdsk:betap')).data()[good_time_slices]; #beta_poloidal 
beta_t = (efit.getNode('\efit_aeqdsk:betat')).data()[good_time_slices]; #beta_toroidal 
kappa = (efit.getNode('\efit_aeqdsk:eout')).data()[good_time_slices]; #elongation - vertical 
triang_l = (efit.getNode('\efit_aeqdsk:doutl')).data()[good_time_slices]; #lower triangularity  
triang_u = (efit.getNode('\efit_aeqdsk:doutu')).data()[good_time_slices]; #upper triangularity of lcfs 
triang = (triang_u + triang_l)/2. #overall triangularity - horizontal (major radius)
li = (efit.getNode('\efit_aeqdsk:li')).data()[good_time_slices]; #internal inductance 
areao = (efit.getNode('\efit_aeqdsk:areao')).data()[good_time_slices]/(100.*100.); #area of lcfs 
vout = (efit.getNode('\efit_aeqdsk:vout')).data()[good_time_slices]/(100.*100.*100.); #volume of lcfs
aout = (efit.getNode('\efit_aeqdsk:aout')).data()[good_time_slices]/(100.); #minor radius of lcfs
rout = (efit.getNode('\efit_aeqdsk:rout')).data()[good_time_slices]/(100.); #major radius of geometric center
zout = (efit.getNode('\efit_aeqdsk:zout')).data()[good_time_slices]/(100.); #z of lcfs (constructed)
rmag = (efit.getNode('\efit_aeqdsk:rmagx')).data()[good_time_slices]/(100.)
zmag = (efit.getNode('\efit_aeqdsk:zmagx')).data()[good_time_slices]/(100.); #z of magnetic axis
rseps = (efit.getNode('\efit_aeqdsk:rseps')).data()[good_time_slices]; #r of upper and lower xpts
rsep_lower = rseps[:,0]/(100.)
rsep_upper = rseps[:,1]/(100.)    
zseps = (efit.getNode('\efit_aeqdsk:zseps')).data()[good_time_slices]; #z of upper and lower xpts
zsep_lower = zseps[:,0]/(100.)
zsep_upper = zseps[:,1] /(100.)  
rvsin = (efit.getNode('\efit_aeqdsk:rvsin')).data()[good_time_slices]/(100.); #r of inner strike point
zvsin = (efit.getNode('\efit_aeqdsk:zvsin')).data()[good_time_slices]/(100.); #z of inner strike point
rvsout = (efit.getNode('\efit_aeqdsk:rvsout')).data()[good_time_slices]/(100.); #r of outer strike point
zvsout = (efit.getNode('\efit_aeqdsk:zvsout')).data()[good_time_slices]/(100.); #z of outer strike point
upper_gap = (efit.getNode('\efit_aeqdsk:otop')).data()[good_time_slices]/100.; # meters
lower_gap = (efit.getNode('\efit_aeqdsk:obott')).data()[good_time_slices]/100.; # meters
q0 = (efit.getNode('\efit_aeqdsk:q0')).data()[good_time_slices]; #safety factor at center
qstar = (efit.getNode('\efit_aeqdsk:qstar')).data()[good_time_slices]; #cylindrical safety factor
q95 = (efit.getNode('\efit_aeqdsk:q95')).data()[good_time_slices]; #edge safety factor
qout = (efit.getNode('\efit_aeqdsk:qout')).data()[good_time_slices]
cpasma = (efit.getNode('\efit_aeqdsk:cpasma')).data()[good_time_slices] #calculated plasma current
BtVac = (efit.getNode('\efit_aeqdsk:btaxv')).data()[good_time_slices] #on-axis plasma toroidal field 
#BtPlasma = (efit.getNode('\efit_aeqdsk:btaxp')).data()[good_time_slices] #on-axis plasma toroidal field
BpAvg = (efit.getNode('\efit_aeqdsk:bpolav')).data()[good_time_slices] #average poloidal field
V_loop_efit = (efit.getNode('\efit_aeqdsk:vloopt')).data()[good_time_slices]; #loop voltage
V_surf_efit = (efit.getNode('\efit_aeqdsk:vsurfa')).data()[good_time_slices]; #surface voltage
Wmhd = (efit.getNode('\efit_aeqdsk:wplasm')).data()[good_time_slices]; #diamagnetic/stored energy, [J]
ssep = (efit.getNode('\efit_aeqdsk:ssep')).data()[good_time_slices]/100.; # distance on midplane between 1st and 2nd separatrices [m]
n_over_ncrit = (efit.getNode('\efit_aeqdsk:xnnc')).data()[good_time_slices]; #vertical stability criterion (EFIT name: xnnc)
inductance = 4.*np.pi*1.E-7 * 0.68 * li/2.; # For simplicity, we use R0 = 0.68 m, but we could use \efit_aeqdsk:rmagx
#dipdt = np.gradient(ip,time_efit)
#dipdt_smoothed = smooth(dipdt,11) #11-point smoothing (moving average box/by convolution)
#V_inductive = inductance*dipdt_smoothed
#V_resistive = V_loop_efit - V_inductive
#P_ohm = ip*V_resistive

efit_a = MDSplus.Tree('analysis', shot)
IDL.run("good_time_slices = efit_check(shot={},tree='{}')".format(shot,'analysis')) 
good_time_slices = getattr(IDL,"good_time_slices") #returns indexes for values passing efit_check test
time_efit_a = (efit_a.getNode('\efit_aeqdsk:time')).data()[good_time_slices]
beta_N_a = (efit_a.getNode('\efit_aeqdsk:betan')).data()[good_time_slices]; #normalized beta 
beta_p_a = (efit_a.getNode('\efit_aeqdsk:betap')).data()[good_time_slices]; #beta_poloidal 
beta_t_a = (efit_a.getNode('\efit_aeqdsk:betat')).data()[good_time_slices]; #beta_toroidal 
kappa_a = (efit_a.getNode('\efit_aeqdsk:eout')).data()[good_time_slices]; #elongation - vertical 
triang_l_a = (efit_a.getNode('\efit_aeqdsk:doutl')).data()[good_time_slices]; #lower triangularity  
triang_u_a = (efit_a.getNode('\efit_aeqdsk:doutu')).data()[good_time_slices]; #upper triangularity of lcfs 
triang_a = (triang_u + triang_l)/2. #overall triangularity - horizontal (major radius)
li_a = (efit_a.getNode('\efit_aeqdsk:li')).data()[good_time_slices]; #internal inductance 
areao_a = (efit_a.getNode('\efit_aeqdsk:areao')).data()[good_time_slices]/(100.*100.); #area of lcfs 
vout_a = (efit_a.getNode('\efit_aeqdsk:vout')).data()[good_time_slices]/(100.*100.*100.); #volume of lcfs
aout_a = (efit_a.getNode('\efit_aeqdsk:aout')).data()[good_time_slices]/(100.); #minor radius of lcfs
rout_a = (efit_a.getNode('\efit_aeqdsk:rout')).data()[good_time_slices]/(100.); #major radius of geometric center
zout_a = (efit_a.getNode('\efit_aeqdsk:zout')).data()[good_time_slices]/(100.); #z of lcfs (constructed)
rmag_a = (efit_a.getNode('\efit_aeqdsk:rmagx')).data()[good_time_slices]/(100.)
zmag_a = (efit_a.getNode('\efit_aeqdsk:zmagx')).data()[good_time_slices]/(100.); #z of magnetic axis
rseps_a = (efit_a.getNode('\efit_aeqdsk:rseps')).data()[good_time_slices]; #r of upper and lower xpts
rsep_lower_a = rseps_a[:,0]/(100.)
rsep_upper_a = rseps_a[:,1]/(100.)    
zseps_a = (efit_a.getNode('\efit_aeqdsk:zseps')).data()[good_time_slices]; #z of upper and lower xpts
zsep_lower_a = zseps_a[:,0]/(100.)
zsep_upper_a = zseps_a[:,1] /(100.)  
rvsin_a = (efit_a.getNode('\efit_aeqdsk:rvsin')).data()[good_time_slices]/(100.); #r of inner strike point
zvsin_a = (efit_a.getNode('\efit_aeqdsk:zvsin')).data()[good_time_slices]/(100.); #z of inner strike point
rvsout_a = (efit_a.getNode('\efit_aeqdsk:rvsout')).data()[good_time_slices]/(100.); #r of outer strike point
zvsout_a = (efit_a.getNode('\efit_aeqdsk:zvsout')).data()[good_time_slices]/(100.); #z of outer strike point
upper_gap_a = (efit_a.getNode('\efit_aeqdsk:otop')).data()[good_time_slices]/100.; # meters
lower_gap_a = (efit_a.getNode('\efit_aeqdsk:obott')).data()[good_time_slices]/100.; # meters
q0_a = (efit_a.getNode('\efit_aeqdsk:q0')).data()[good_time_slices]; #safety factor at center
qstar_a = (efit_a.getNode('\efit_aeqdsk:qstar')).data()[good_time_slices]; #cylindrical safety factor
q95_a = (efit_a.getNode('\efit_aeqdsk:q95')).data()[good_time_slices]; #edge safety factor
qout_a = (efit_a.getNode('\efit_aeqdsk:qout')).data()[good_time_slices]
cpasma_a = (efit_a.getNode('\efit_aeqdsk:cpasma')).data()[good_time_slices] #calculated plasma current
BtVac_a = (efit_a.getNode('\efit_aeqdsk:btaxv')).data()[good_time_slices] #on-axis plasma toroidal field 
BtPlasma_a = (efit_a.getNode('\efit_aeqdsk:btaxp')).data()[good_time_slices] #on-axis plasma toroidal field
BpAvg_a = (efit_a.getNode('\efit_aeqdsk:bpolav')).data()[good_time_slices] #average poloidal field
V_loop_efit_a = (efit_a.getNode('\efit_aeqdsk:vloopt')).data()[good_time_slices]; #loop voltage
V_surf_efit_a = (efit_a.getNode('\efit_aeqdsk:vsurfa')).data()[good_time_slices]; #surface voltage
Wmhd_a = (efit_a.getNode('\efit_aeqdsk:wplasm')).data()[good_time_slices]; #diamagnetic/stored energy, [J]
ssep_a = (efit_a.getNode('\efit_aeqdsk:ssep')).data()[good_time_slices]/100.; # distance on midplane between 1st and 2nd separatrices [m]
n_over_ncrit_a = (efit_a.getNode('\efit_aeqdsk:xnnc')).data()[good_time_slices]; #vertical stability criterion (EFIT name: xnnc)
inductance_a = 4.*np.pi*1.E-7 * 0.68 * li/2.; # For simplicity, we use R0 = 0.68 m, but we could use \efit_aeqdsk:rmagx
#dipdt_a = np.gradient(ip,time_efit)
#dipdt_smoothed_a = smooth(dipdt,11) #11-point smoothing (moving average box/by convolution)
#V_inductive_a = inductance_a*dipdt_smoothed_a
#V_resistive_a = V_loop_efit_a - V_inductive_a
#P_ohm_a = ip_a*V_resistive_a


plt.figure() 
plt.scatter(time_efit,cpasma,c='b',label='magnetics')
plt.scatter(time_ip,ip,c='g',label='analysis')  
plt.xlabel('t (s)')
plt.ylabel('ip')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.draw()

plt.figure() 
plt.scatter(time_efit,q95,c='b',label=tree_efit)
plt.scatter(time_efit_a,q95_a,c='g',label='analysis')  
plt.xlabel('t (s)')
plt.ylabel('q95')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.draw()
plt.savefig('q95_v_t_efits_{}.png'.format(shot))
plt.close()

plt.figure() 
plt.scatter(time_efit,Wmhd,c='b',label=tree_efit)
plt.scatter(time_efit_a,Wmhd_a,c='g',label='analysis')  
plt.xlabel('t (s)')
plt.ylabel('Wmhd')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.draw()
plt.savefig('Wmhd_v_t_efits_{}.png'.format(shot))
plt.close()

plt.figure() 
plt.scatter(time_efit,li,c='b',label=tree_efit)
plt.scatter(time_efit_a,li_a,c='g',label='analysis')  
plt.xlabel('t (s)')
plt.ylabel('li')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.draw()
plt.savefig('li_v_t_efits_{}.png'.format(shot))
plt.close()