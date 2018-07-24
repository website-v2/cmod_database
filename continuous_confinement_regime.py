# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:36:44 2018

@author: mathewsa

This code is based on convolution of (dtau_E/dt)*tau_E with Gaussian
kernel of specified width and sigma to reduce noise and interpret transition
regions of wide variety (e.g. ohmic H-mode, hot H-mode as indicated by density increase)
"""

import matplotlib.pyplot as plt 
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import MDSplus 
from MDSplus import *
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import VotingClassifier
import numpy as np   
import os
import sys
from os import getenv
from datetime import datetime
sys.path.append('/home/mathewsa/Desktop/confinement_table/codes+/')
import fast_efit06
from fast_efit06 import path_shot
sys.path.append('/home/mathewsa/Desktop/confinement_table/codes+/')
import data_acquire_MDSplus 
import itertools
import operator
import random
import sqlite3 
from mpl_toolkits.mplot3d import Axes3D  
from itertools import product
import pickle
import matplotlib
from matplotlib import rcParams
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid",{'axes.facecolor': 'white','axes.grid': True,}) 
from matplotlib.ticker import FuncFormatter
import scipy.integrate as integrate
from scipy.integrate import quad

def y_fmt(y, pos):
    decades = [1e21, 1e20, 1e19, 1e18, 1e17, 1e16, 1e15, 1e14, 1e13, 1e12, 1e11,\
    1e10, 1e9, 1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3,\
    1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12,]
    suffix  = [r"$\times 10^{21}$", r"$\times  10^{20}$", r"$\times  10^{19}$",\
    r"$\times 10^{18}$", r"$\times  10^{17}$", r"$\times  10^{16}$", r"$\times 10^{15}$",\
    r"$\times  10^{14}$", r"$\times  10^{13}$", r"$\times  10^{12}$",\
    r"$\times 10^{11}$", r"$\times 10^{10}$", r"$\times 10^{9}$",\
    r"$\times  10^{8}$", r"$\times  10^{7}$", r"$\times 10^{6}$", r"$\times  10^{5}$",\
    r"$\times  10^{4}$", r"$\times 10^{3}$", r"$\times  10^{2}$", r"$\times  10^{1}$", r"$\times  10^{0}$",\
    r"$\times  10^{-1}$", r"$\times  10^{-2}$", r"$\times 10^{-3}$",\
    r"$\times  10^{-4}$", r"$\times  10^{-5}$", r"$\times 10^{-6}$",\
    r"$\times  10^{-7}$", r"$\times  10^{-8}$", r"$\times 10^{-9}$",\
    r"$\times  10^{-10}$", r"$\times  10^{-11}$", r"$\times  10^{-12}$"]
    if y == 0:
        return r"$0.0 \times  10^{0}$"
    for i, d in enumerate(decades):
        if np.abs(y) >=d:
            val = np.around(y/float(d),decimals=1)
            val_latex = r"${}$".format(np.around(y/float(d),decimals=1))
            tx = '{}'.format(val_latex)+'{}'.format(suffix[i])
            return tx 
            
def gaussian(x, sigma):
    return np.exp(-(x/sigma)**2/2) 

def smooth(y,box_pts):
#    box = np.ones(box_pts)/box_pts
    x = (np.linspace(-box_pts/2.,box_pts/2.,box_pts + 1))
    sigma = box_pts/5.
    integral = quad(gaussian, x[0], x[-1], args=(sigma))[0]
    box = gaussian(x, sigma)/integral
    y_smooth = np.convolve(y,box,mode='same')
    return y_smooth
    
start_time = 0.383 #seconds
end_time = 1.744 #seconds
timebase = np.arange(round(start_time,3),round(end_time,3),0.001) #using 1 millisecond constant interval
shot = 1160930033
tstart = 1000.*start_time #milliseconds
dt = 1.0 #milliseconds
tend = 1000.*end_time #milliseconds     
fast_efit06.main(shot,tstart,tend,dt) 
data_acquire_MDSplus.main(shot,timebase,path_shot)
data = np.load('/home/mathewsa/Desktop/single_shot_training_table_py.npz')
extra_variables = np.load('/home/mathewsa/Desktop/extra_variables.npz')

X_data = []
j = 0
time = round(((extra_variables['timebase'])[j]),3)
while time < round(((extra_variables['timebase'])[-1]),3):  
    time = round(((extra_variables['timebase'])[j]),3) 
    X_data.append([(data['Wmhd'])[j],(data['nebar_efit'])[j],\
    (data['beta_p'])[j],(data['P_ohm'])[j],(data['li'])[j],\
    (data['rmag'])[j],(data['Halpha'])[j]])
    j = j + 1
    
box_size = 50
Pinput = (data['P_ohm'] + data['p_icrf'] - np.gradient(data['Wmhd'], timebase))
tau_E = smooth(data['Wmhd'],box_size)/smooth(Pinput,box_size)
dtau_Edt = smooth(np.gradient(tau_E, timebase),box_size)
    
plt.figure()
plt.plot(timebase,tau_E*dtau_Edt,color='green')
plt.ylabel(r"$(\frac {d\tau_E}{dt}) \tau_E $")
plt.xlabel(r'Time (s)')
plt.show()

dnebar_efit_Edt = smooth(np.gradient(data['nebar_efit'], timebase),box_size)
plt.figure()
plt.plot(timebase,dnebar_efit_Edt/data['nebar_efit'],color='green')
plt.ylabel(r"$(\frac {d\bar{n}}{dt})/{\bar {n}} $")
plt.xlabel(r'Time (s)')
plt.show()