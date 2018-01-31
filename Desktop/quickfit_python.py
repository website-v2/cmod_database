# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:53:42 2018

@author: mathewsa
"""

"""The output Te_array and ne_array provide output from the combined Thomson and ECE
QUICKFIT routine where the first index corresponds to the time slice, the second index is the
smoothed over radial coordinate (with 100 points) at that time slice, and the third index is 
the electron temperature as determined by the fitting routine using both TS and ECE. It similarly
provides this measure for electron density, ne, as well. The time range is over the same range
as spanned by the tree efit06, and the intervals are approximately every 33ms (TS time)

NEED TO FIGURE OUT HOW TO MAKE THIS CORRESPOND TO MY TABLE STANDARD TIMEBASE (E.G. INTERPOLATE WITH FINER TIME)

NEED TO SELECT THE PARTICULAR QUANTITIES/GRADIENTS AT SPECIFIC RADII

IF USING FOR THE GENERAL TABLE POPULATION CODE, THEN NEED TO USE EFIT_ (NOT JUST EFIT06)"""

import idlpy
from idlpy import *
import matplotlib.pyplot as plt 
import MDSplus 
from MDSplus import *

shot = 1100204004
tree_use = 'efit06' #choosing efit tree
efit = MDSplus.Tree('{}'.format(tree_use), shot)
time_efit = (efit.getNode('\efit_aeqdsk:time')).data() 

IDL.run("shot={}".format(shot))
IDL.run("quick_fit, shot, ne_deg=nedeg, /comb_te, efit_tree = '{}', adata, /nosave".format(tree_use))
IDL.run("f_fit = adata.afp") # [time, 100, 2] array containing ne and Te fits 
IDL.run("ne_array_total = F_FIT(*, *, *)") 
ne_array_total = getattr(IDL,"ne_array_total")
t_list = len(ne_array_total[0][0]) #this is the number of time points from TS during time interval of the efit
#(time_efit[-1] - time_efit[0])/t_list 
Te_quickfit = []
ne_quickfit = []

t_index = 0
while t_index < t_list:
    IDL.run("t_fit = adata.t_ts")   # time in seconds 
    IDL.run("r0_fit = adata.r_out") # major radius R0 on axis as function of time
    IDL.run("r_fit = adata.ax") # 100 pt array containing the smooth minor radius array
    IDL.run("t_ind = {}".format(t_index)) #midpoint in time array is 25, usually around 1 sec or so; in OLD TS data circa 2012
               # note that new data from 2015 has more timepoints (172) so midpoint in array t = 1sec is about 86
    IDL.run("ne_array = F_FIT(t_ind, *, 0)") # density array at t_ind
    IDL.run("te_array = F_FIT(t_ind, *, 1)/1e3") # Te array (keV)
    IDL.run("r_array = R_FIT + R0_FIT(t_ind)")
    #IDL.run("window, 0")
    #IDL.run("plot, r_array, ne_array, xtitle='R maj (cm)', ytitle='density (10!E20!Nm!E-3!N)'")
    #IDL.run("window, 1")
    #IDL.run("plot, r_array, Te_array, xtitle='R maj (cm)', ytitle='Te (keV)'")
    ##IDL.run("end")
    
    IDL.run("ishot = adata.ISHOT") 
    IDL.run("ne_cts = adata.NE_CTS")
    IDL.run("ne_ets = adata.NE_ETS")
    IDL.run("te_cts = adata.TE_CTS")
    IDL.run("te_ets = adata.TE_ETS")
    IDL.run("rmid_cts = adata.RMID_CTS")
    IDL.run("rmid_ets = adata.RMID_ETS")
    IDL.run("te_gpc = adata.TE_GPC")
    IDL.run("te_gpc2 = adata.TE_GPC2")
    #IDL.run("tgpc = adata.tgpc")
    #IDL.run("tgpc2 = adata.tgpc2")
    IDL.run("r_gpc = adata.R_GPC")
    IDL.run("r_gpc2 = adata.R_GPC2") 
    IDL.run("aa0gfp = adata.AA0GFP")
    
    adata = getattr(IDL,"adata") 
    quickfit_variables = list(adata.keys())
    
    t_fit = getattr(IDL,"t_fit") #thomson time
    r0_fit = getattr(IDL,"r0_fit")
    r_fit = getattr(IDL,"r_fit") #std minor radius axis all output profiles interpolated to  
    f_fit = getattr(IDL,"f_fit") #Least square profile fitting   
    ne_array = getattr(IDL,"ne_array")
    te_array = getattr(IDL,"te_array")
    r_array = getattr(IDL,"r_array")
     
    ishot = getattr(IDL,"ishot")
    ne_cts = getattr(IDL,"ne_cts")
    ne_ets = getattr(IDL,"ne_ets")
    te_cts = getattr(IDL,"te_cts")
    te_ets = getattr(IDL,"te_ets")
    rmid_cts = getattr(IDL,"rmid_cts")
    rmid_ets = getattr(IDL,"rmid_ets")
    te_gpc = getattr(IDL,"te_gpc")
    te_gpc2 = getattr(IDL,"te_gpc2")
    r_gpc = getattr(IDL,"r_gpc")
    r_gpc2 = getattr(IDL,"r_gpc2")
    te_gpc2 = getattr(IDL,"te_gpc2")
    #tgpc = getattr(IDL,"tgpc")
    #tgpc2 = getattr(IDL,"tgpc2")
    aa0gfp = getattr(IDL,"aa0gfp") #(a0/afp)*agp where agp is derivatives of afp to ax
    
    Te_quickfit.append((t_fit[t_index],r_array,te_array))
    ne_quickfit.append((t_fit[t_index],r_array,ne_array))
    
    t_index = t_index + 1


#fig=plt.figure()
#fig.show()
#ax=fig.add_subplot(111)
##ax.plot(time_efit129,V_loop_efit129,c='b',marker="^",ls='--',label='129x129',fillstyle='none')
##ax.plot(time_efit129,V_loop_efit33,c='g',marker=(8,2,0),ls='--',label='33x33')
#ax.plot(r_array,ne_array,c='k',ls='-',label='Thomson ne (10^20 m^-3)')
#ax.plot(r_array,te_array,c='r',marker="v",ls='-',label='Thomson  Te (keV)') 
#plt.legend(loc=1,prop={'size': 10})
#plt.show()
##fig.savefig('quickfit.png', dpi=fig.dpi)
#plt.close()

#def array_r(array):
#    i = 0  
#    app = []
#    while i < len(array):
#        app.append(array[i][t_index]) #25 corresponds to 40th time slice out of 300 time slices
#        i = i + 1 #can generalize this for all time slices
#    return app
    

#fig=plt.figure()
#fig.show()
#ax=fig.add_subplot(111)
#ax.plot(array_r(r_gpc),array_r(te_gpc),c='k',ls='-',label='Te GPC')
#ax.plot(array_r(r_gpc2),array_r(te_gpc2),c='r',marker="v",ls='-',label='Te GPC2')
#plt.legend(loc=1,prop={'size': 10})
#plt.show() 
#plt.close()