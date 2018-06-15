# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:12:24 2018

@author: mathewsa

Currently configured to conduct analysis/prediction on a single shot

"""
import matplotlib.pyplot as plt 
import MDSplus 
from MDSplus import *
from sklearn.preprocessing import StandardScaler  
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
import pickle

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
    (data['beta_p'])[j],(data['P_ohm'])[j],(data['b_bot_mks'])[j]])
    j = j + 1
    
#Loading saved model
RF_LH_pkl_filename = '/home/mathewsa/Desktop/RF_classifier_LH-shot-b_bot_Wmhd-nebar_efit-beta_p-P_ohm.pkl'
RF_LH_model_pkl = open(RF_LH_pkl_filename, 'rb')
RF_LH_model = pickle.load(RF_LH_model_pkl)
NN_LH_pkl_filename = '/home/mathewsa/Desktop/NN_classifier_LH-shot-b_bot_Wmhd-nebar_efit-beta_p-P_ohm.pkl'
NN_LH_model_pkl = open(NN_LH_pkl_filename, 'rb')
NN_LH_model = pickle.load(NN_LH_model_pkl)
GNB_LH_pkl_filename = '/home/mathewsa/Desktop/GNB_classifier_LH-shot-b_bot_Wmhd-nebar_efit-beta_p-P_ohm.pkl'
GNB_LH_model_pkl = open(GNB_LH_pkl_filename, 'rb')
GNB_LH_model = pickle.load(GNB_LH_model_pkl)
LR_LH_pkl_filename = '/home/mathewsa/Desktop/LR_classifier_LH-shot-b_bot_Wmhd-nebar_efit-beta_p-P_ohm.pkl'
LR_LH_model_pkl = open(LR_LH_pkl_filename, 'rb')
LR_LH_model = pickle.load(LR_LH_model_pkl)

scalerfile = '/home/mathewsa/Desktop/scaler_LH-shot-b_bot_Wmhd-nebar_efit-beta_p-P_ohm.sav'
scaler = pickle.load(open(scalerfile, 'rb')) #must use transformation applied to training data

#score = RF_LH_model.score(X_test, y_test)
X_data_normalized = scaler.transform(X_data)
prediction_RF = RF_LH_model.predict(X_data_normalized)
prediction_NN = NN_LH_model.predict(X_data_normalized)
prediction_GNB = GNB_LH_model.predict(X_data_normalized)
prediction_LR = LR_LH_model.predict(X_data_normalized)
    
plt.figure()
plt.plot(timebase,prediction_RF,color='black',label='Random Forest')
plt.plot(timebase,prediction_NN,label='Neural Net')
plt.plot(timebase,prediction_GNB,label='Gaussian naive Bayes')
plt.plot(timebase,prediction_LR,label='Logistic Regression') 
plt.ylabel(r'Prediction')
plt.xlabel(r'time (s)')
plt.ylim([-0.1,1.1])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.figure()
plt.plot(timebase,(data['Wmhd']),color='green')
plt.ylabel(r"$W_{mhd}$")
plt.xlabel(r'time (s)')
plt.show()
plt.figure()
plt.plot(timebase,(data['nebar_efit']),color='green') 
plt.ylabel(r"$\bar{n}$")
plt.xlabel(r'time (s)')
plt.show()
plt.figure()
plt.plot(timebase,(data['beta_p']),color='green')
plt.ylabel(r"$\beta_p$")
plt.xlabel(r'time (s)')
plt.show()
plt.figure()
plt.plot(timebase,(data['P_ohm']),color='green')
plt.ylabel(r"$P_{ohm}$")
plt.xlabel(r'time (s)')
plt.show()
plt.figure()
plt.plot(timebase,(data['b_bot_mks']),color='green')
plt.ylabel(r'b_bot (mks)')
plt.xlabel(r'time (s)')
plt.show()
plt.figure()
plt.plot(timebase,(extra_variables['gpc2_te0']),color='red')
plt.ylabel(r'gpc2_te0')
plt.xlabel(r'time (s)')
plt.show()
plt.figure()
plt.plot(timebase,(extra_variables['gpc_te8']),color='red')
plt.ylabel(r'gpc_te8')
plt.xlabel(r'time (s)')
plt.show()
plt.figure()
plt.plot(timebase,(extra_variables['NL_04']),color='red')
plt.ylabel(r'NL_04')
plt.xlabel(r'time (s)')
plt.show()
plt.figure()
plt.plot(timebase,(extra_variables['ne_t']),color='red')
plt.ylabel(r'ne_t')
plt.xlabel(r'time (s)')
plt.show()

plt.figure()
plt.plot(timebase,prediction_RF,color='black',label='Random Forest')
plt.plot(timebase,prediction_NN,label='Neural Net')
plt.plot(timebase,prediction_GNB,label='Gaussian naive Bayes')
plt.plot(timebase,prediction_LR,label='Logistic Regression') 
plt.ylabel(r'Prediction')
plt.xlabel(r'time (s)')
plt.xlim([1.5,1.8])
plt.ylim([-0.1,1.1])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.plot(timebase,(data['Wmhd']),color='green')
plt.ylabel(r"$W_{mhd}$")
plt.xlabel(r'time (s)')
plt.xlim([1.5,1.8])
plt.show()
plt.figure()
plt.plot(timebase,(data['nebar_efit']),color='green')
plt.ylabel(r"$\bar{n}$")
plt.xlabel(r'time (s)')
plt.xlim([1.5,1.8])
plt.show()
plt.figure()
plt.plot(timebase,(data['beta_p']),color='green')
plt.ylabel(r"$\beta_p$")
plt.xlabel(r'time (s)')
plt.xlim([1.5,1.8])
plt.show()
plt.figure()
plt.plot(timebase,(data['P_ohm']),color='green')
plt.ylabel(r"$P_{ohm}$")
plt.xlabel(r'time (s)')
plt.xlim([1.5,1.8])
plt.show()
plt.figure()
plt.plot(timebase,(data['b_bot_mks']),color='green')
plt.ylabel(r'b_bot (mks)')
plt.xlim([1.5,1.8])
plt.show()
plt.figure()
plt.plot(timebase,(extra_variables['gpc2_te0']),color='red')
plt.ylabel(r'gpc2_te0')
plt.xlabel(r'time (s)')
plt.xlim([1.5,1.8])
plt.show()
plt.figure()
plt.plot(timebase,(extra_variables['gpc_te8']),color='red')
plt.ylabel(r'gpc_te8')
plt.xlabel(r'time (s)')
plt.xlim([1.5,1.8])
plt.show()
plt.figure()
plt.plot(timebase,(extra_variables['NL_04']),color='red')
plt.ylabel(r'NL_04')
plt.xlabel(r'time (s)')
plt.xlim([1.5,1.8])
plt.show()
plt.figure()
plt.plot(timebase,(extra_variables['ne_t']),color='red')
plt.ylabel(r'ne_t')
plt.xlabel(r'time (s)')
plt.xlim([1.5,1.8])
plt.show()