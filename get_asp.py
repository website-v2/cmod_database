# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:09:18 2018

@author: mathewsa
"""

####Why ind and ind2 defined twice?
####Why tmin = np.interp(min(rho),Y,t[I]) 
####Why np.arange(Ymax,Ymin,1) instead of Ymin first?
#what is proper offset?
#what is smooth/spline interp doing? (e.g. univariatespline?)
#are my fits/filter actually good/representative of data? 
##Can I make plots for all time, or is it one plot per plunge? 
##what are the fluctuations I am suppressing with filter? Are they important?
#Should I smooth rho with Gaussian filter as well and then sort this as x-axis?
#what does asp and aveMachProbe stand for? 

##I can tabulate \Delta V vs \Delta rho for peak to bottom, HFHM (half-width at half-max)
#and have these values stored versus P_threshold for transitions and for temperatures/densities/Wmhd/Ptot
#for each of the relevant shots 
#along with x-point position and strike point position tabulated...all in either a database or pandas table
#or excel table for the shots of interest
import MDSplus 
from MDSplus import *
import operator
import random
import sqlite3
import pymssql
from datetime import datetime
import numpy as np
import sys
import matplotlib
from matplotlib import rcParams
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns
sns.set_style("whitegrid",{'axes.facecolor': 'white','axes.grid': True,}) 
import pandas as pd 
import os
import sys
from os import getenv  

shot = 1160620011
tslice = 1293 #in milliseconds?
connect = pymssql.connect(host='ALCDB2',user='kuang',password='pfcworld',database='edge')
QueryFields = 'shot,time_slice,twindow,thick_slice,tstart,tend,offset_ASP,key_words'
TableName = 'master';
condition = 'WHERE (shot = '+str(shot)+') AND (time_slice = '+str(tslice)+')'
sql_query = 'SELECT'

cursor = connect.cursor() 
output = cursor.execute('SELECT {} from master WHERE (shot = {}) AND (time_slice = {})'.format(QueryFields,shot,tslice))
rows = cursor.fetchall()   
connect.commit()
connect.close()

twindow = [tslice*0.001 - 0.5*rows[0][2], tslice*0.001 + 0.5*rows[0][2]]
print('MLP Power Balance Calculation  provides rho offset [mm] of '+str(rows[0][7])+' (or not done)')
offsetASP = input('Provide MLP rho offset [mm]:')
offsetASP = offsetASP/1000;

def getASPfromTree(shot,twindow,path,offsetASP):
    #    mdsconnect('alcdata.psfc.mit.edu');
    #    mdsopen('edge',shot);
    edge_tree = MDSplus.Tree('edge',shot)
    ne = (edge_tree.getNode(path+'DENSITY_FIT')).data()
    time = (edge_tree.getNode(path+'DENSITY_FIT')).dim_of().data()
    te = (edge_tree.getNode(path+'TE_FIT')).data()
    vp = (edge_tree.getNode(path+'PHI_FIT')).data()
    jsat = (edge_tree.getNode(path+'JSAT_FIT')).data()
    rho = (edge_tree.getNode(path+'RHO')).data()
    rhotime = (edge_tree.getNode(path+'RHO')).dim_of().data()
    #true_time = time[(time > twindow[0]) & (time < twindow[1])] 
    ind = np.where((time > twindow[0]) & (time < twindow[1]))[0]
    ne = ne[ind]
    te = te[ind] 
    vp = vp[ind]
    jsat = jsat[ind]
    time_node = time[ind]
    rho = np.interp(time_node,rhotime,rho)+offsetASP
    
    return [jsat,ne,te,vp,rho,time_node] 
    
[jsat0,ne0,te0,vp0,rho0,t0] = getASPfromTree(shot,twindow,'\EDGE::TOP.PROBES.ASP.MLP.P0:',offsetASP);
[jsat1,ne1,te1,vp1,rho1,t1] = getASPfromTree(shot,twindow,'\EDGE::TOP.PROBES.ASP.MLP.P1:',offsetASP);
[jsat2,ne2,te2,vp2,rho2,t2] = getASPfromTree(shot,twindow,'\EDGE::TOP.PROBES.ASP.MLP.P2:',offsetASP);
[jsat3,ne3,te3,vp3,rho3,t3] = getASPfromTree(shot,twindow,'\EDGE::TOP.PROBES.ASP.MLP.P3:',offsetASP);

plt.figure()
plt.plot(rho0,ne0*1e-20,label='P0') 
plt.plot(rho1,ne1*1e-20,label='P1')
plt.plot(rho2,ne2*1e-20,label='P2')
plt.plot(rho3,ne3*1e-20,label='P3')
plt.ylabel(r"$n_e \ (10^{20} m^{-3})$")
plt.legend()    
plt.show()

plt.figure()
plt.plot(rho0,te0,label='P0') 
plt.plot(rho1,te1,label='P1')
plt.plot(rho2,te2,label='P2')
plt.plot(rho3,te3,label='P3')
plt.ylabel(r"$T_e \ (eV)$")
plt.legend()    
plt.show() 

plt.figure()
plt.plot(rho0,vp0,label='P0') 
plt.plot(rho1,vp1,label='P1')
plt.plot(rho2,vp2,label='P2')
plt.plot(rho3,vp3,label='P3')
plt.ylabel(r"$V_p \ (V)$")
plt.legend()    
plt.show()  

def aveMachProbe(Value0,Value1,Value2,Value3,x0,x1,x2,x3):
    #applies same base for rho
    Value1 = np.interp(x0,x1,Value1);
    Value2 = np.interp(x0,x2,Value2);
    Value3 = np.interp(x0,x3,Value3); 
     
    ind0 = np.argwhere(np.isfinite(Value0))
    ind1 = np.argwhere(np.isfinite(Value1))
    ind2 = np.argwhere(np.isfinite(Value2))
    ind3 = np.argwhere(np.isfinite(Value3))
    
    Ymin = min(ind0[-1],ind1[-1],ind2[-1],ind3[-1])[0];
    Ymax = max(ind0[0],ind1[0],ind2[0],ind3[0])[0];
    
    ind = np.arange(Ymax,Ymin,1)
    Value0 = Value0[ind];
    Value1 = Value1[ind];
    Value2 = Value2[ind];
    Value3 = Value3[ind];

    aveValue = Value0 + Value1 + Value2 + Value3;
    aveValue = 0.25*aveValue;
    
    allValue = [Value0,Value1,Value2,Value3];
    
    return [aveValue,allValue,ind]

#Use a master rho base
rho = rho0*1000;
t = t0;
[ne,ne_all,ind] = aveMachProbe(ne0,ne1,ne2,ne3,t0,t1,t2,t3);
[jsat,jsat_all,ind] = aveMachProbe(jsat0,jsat1,jsat2,jsat3,t0,t1,t2,t3);
[te,te_all,ind2] = aveMachProbe(te0,te1,te2,te3,t0,t1,t2,t3);
[vp,vp_all,ind2] = aveMachProbe(vp0,vp1,vp2,vp3,t0,t1,t2,t3);
t = t[ind];
rho = rho[ind];

#smoothing spline fit applied (cubic)
num_smooth = 20000 

def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))
    
span = 5

from scipy.interpolate import interp1d
from scipy.interpolate import spline #takes too much memory e.g. jsat_smooth = spline(t,jsat,t_smooth);
from scipy.interpolate import InterpolatedUnivariateSpline

t_smooth = np.arange(min(t),max(t),(max(t)-min(t))/(num_smooth-1));
rho_smooth = InterpolatedUnivariateSpline(t, rho, k=3) 
rho_smooth = np.interp(t_smooth,t,rho_smooth(t))
ne_smooth = InterpolatedUnivariateSpline(t, ne, k=3) 
ne_smooth = np.interp(t_smooth,t,ne_smooth(t))
ne_smooth = smooth(ne_smooth,span);
jsat_smooth = InterpolatedUnivariateSpline(t, jsat, k=3) 
jsat_smooth = np.interp(t_smooth,t,jsat_smooth(t))
jsat_smooth = smooth(jsat_smooth,span);
te_smooth = InterpolatedUnivariateSpline(t, te, k=3) 
te_smooth = np.interp(t_smooth,t,te_smooth(t))
te_smooth = smooth(te_smooth,span);
vp_smooth = InterpolatedUnivariateSpline(t, vp, k=3) 
vp_smooth = np.interp(t_smooth,t,vp_smooth(t))
vp_smooth = smooth(vp_smooth,span);

plt.figure()
plt.plot(rho,ne*1e-20,label='Average')  
plt.plot(rho_smooth,ne_smooth*1e-20,label='Spline Fit') 
plt.ylabel(r"$n_e \ (10^{20} m^{-3})$")
plt.legend()    
plt.show()

plt.figure()
plt.plot(rho,te,label='Average')  
plt.plot(rho_smooth,te_smooth,label='Spline Fit') 
plt.ylabel(r"$T_e \ (eV)$")
plt.legend()    
plt.show() 

plt.figure()
plt.plot(rho,vp,label='Average')  
plt.plot(rho_smooth,vp_smooth,label='Spline Fit') 
plt.ylabel(r"$V_p \ (V)$")
plt.legend()    
plt.show()  

t_smooth = np.arange(min(t),max(t),(max(t)-min(t))/(num_smooth-1)); 
rho_smooth = np.interp(t_smooth,t,rho) 
ne_smooth = np.interp(t_smooth,t,ne)  
jsat_smooth = np.interp(t_smooth,t,jsat)  
te_smooth = np.interp(t_smooth,t,te)  
vp_smooth = np.interp(t_smooth,t,vp)  

index = np.argsort(rho_smooth)
rho_smooth = rho_smooth[index]
ne_smooth = ne_smooth[index]
jsat_smooth = jsat_smooth[index]
te_smooth = te_smooth[index]
vp_smooth = vp_smooth[index] 
index = np.argsort(rho)
rho = rho[index]
ne = ne[index]
jsat = jsat[index]
te = te[index]
vp = vp[index]

#smoothing with convolution of Gaussian kernel
box_pts = 500
import scipy.integrate as integrate
from scipy.integrate import quad
def gaussian(x, sigma):
    return np.exp(-(x/sigma)**2/2) 

def smooth(y,box_pts):
#    box = np.ones(box_pts)/box_pts
    x = (np.linspace(-box_pts/2.,box_pts/2.,box_pts + 1))
    sigma = box_pts/5.
    integral = quad(gaussian, x[0], x[-1], args=(sigma))[0]
    box = gaussian(x, sigma)
    box = gaussian(x, sigma)/box.sum() #integral
    y_smooth = np.convolve(y,box,mode='same')
    return y_smooth
    
plt.figure()
plt.plot(rho[box_pts:-1],smooth(ne*1e-20,box_pts)[box_pts:-1],label='Average (Gaussian Filter)')  
plt.plot(rho_smooth[box_pts:-1],smooth(ne_smooth*1e-20,box_pts)[box_pts:-1],label='Spline Fit (Gaussian Filter)') 
plt.ylabel(r"$n_e \ (10^{20} m^{-3})$")
plt.legend()    
plt.show()

plt.figure()
plt.plot(rho[box_pts:-1],smooth(te,box_pts)[box_pts:-1],label='Average (Gaussian Filter)')  
plt.plot(rho_smooth[box_pts:-1],smooth(te_smooth,box_pts)[box_pts:-1],label='Spline Fit (Gaussian Filter)') 
plt.ylabel(r"$T_e \ (eV)$")
plt.legend()    
plt.show() 

plt.figure()
plt.plot(rho[box_pts:-1],smooth(vp,box_pts)[box_pts:-1],label='Average (Gaussian Filter)')  
plt.plot(rho_smooth[box_pts:-1],smooth(vp_smooth,box_pts)[box_pts:-1],label='Spline Fit (Gaussian Filter)') 
plt.ylabel(r"$V_p \ (V)$")
plt.legend()    
plt.show()