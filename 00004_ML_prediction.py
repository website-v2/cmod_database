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
import pandas as pd
import seaborn as sn

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
    
#Loading saved model
RF_LHI_pkl_filename = '/home/mathewsa/Desktop/00004_RF_classifier_LHI.pkl'
RF_LHI_model_pkl = open(RF_LHI_pkl_filename, 'rb')
RF_LHI_model = pickle.load(RF_LHI_model_pkl)
NN_LHI_pkl_filename = '/home/mathewsa/Desktop/00004_NN_classifier_LHI.pkl'
NN_LHI_model_pkl = open(NN_LHI_pkl_filename, 'rb')
NN_LHI_model = pickle.load(NN_LHI_model_pkl)
GNB_LHI_pkl_filename = '/home/mathewsa/Desktop/00004_NB_classifier_LHI.pkl'
GNB_LHI_model_pkl = open(GNB_LHI_pkl_filename, 'rb')
GNB_LHI_model = pickle.load(GNB_LHI_model_pkl)
LR_LHI_pkl_filename = '/home/mathewsa/Desktop/00004_LR_classifier_LHI.pkl'
LR_LHI_model_pkl = open(LR_LHI_pkl_filename, 'rb')
LR_LHI_model = pickle.load(LR_LHI_model_pkl)

scalerfile = '/home/mathewsa/Desktop/00004_scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb')) #must use transformation applied to training data

#score = RF_LH_model.score(X_test, y_test)
X_data_normalized = scaler.transform(X_data)
prediction_RF = RF_LHI_model.predict(X_data_normalized)
prediction_proba_RF_L = RF_LHI_model.predict_proba(X_data_normalized)[:,0]
prediction_proba_RF_H = RF_LHI_model.predict_proba(X_data_normalized)[:,1]
prediction_proba_RF_I = RF_LHI_model.predict_proba(X_data_normalized)[:,2]
prediction_NN = NN_LHI_model.predict(X_data_normalized)
prediction_proba_NN_L = NN_LHI_model.predict_proba(X_data_normalized)[:,0]
prediction_proba_NN_H = NN_LHI_model.predict_proba(X_data_normalized)[:,1]
prediction_proba_NN_I = NN_LHI_model.predict_proba(X_data_normalized)[:,2]
prediction_GNB = GNB_LHI_model.predict(X_data_normalized)
prediction_proba_GNB_L = GNB_LHI_model.predict_proba(X_data_normalized)[:,0]
prediction_proba_GNB_H = GNB_LHI_model.predict_proba(X_data_normalized)[:,1]
prediction_proba_GNB_I = GNB_LHI_model.predict_proba(X_data_normalized)[:,2]
prediction_LR = LR_LHI_model.predict(X_data_normalized)
prediction_proba_LR_L = LR_LHI_model.predict_proba(X_data_normalized)[:,0]
prediction_proba_LR_H = LR_LHI_model.predict_proba(X_data_normalized)[:,1]
prediction_proba_LR_I = LR_LHI_model.predict_proba(X_data_normalized)[:,2]
    
plt.figure()
plt.plot(timebase,prediction_RF,label='Random Forest')
plt.plot(timebase,prediction_NN,label='Neural Net')
plt.plot(timebase,prediction_GNB,label='Gaussian naive Bayes')
plt.plot(timebase,prediction_LR,label='Logistic Regression') 
plt.ylabel(r'Prediction')
plt.xlabel(r'time (s)')
plt.ylim([-0.1,1.1])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.figure()
plt.plot(timebase,prediction_proba_RF_L,label='L-mode')
plt.plot(timebase,prediction_proba_RF_H,label='H-mode')
plt.plot(timebase,prediction_proba_RF_I,label='I-mode')  
plt.ylabel(r'Prediction Probability (RF)')
plt.xlabel(r'time (s)')
plt.ylim([-0.1,1.1])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.figure()
plt.plot(timebase,prediction_proba_NN_L,label='L-mode')
plt.plot(timebase,prediction_proba_NN_H,label='H-mode')
plt.plot(timebase,prediction_proba_NN_I,label='I-mode')  
plt.ylabel(r'Prediction Probability (NN)')
plt.xlabel(r'time (s)')
plt.ylim([-0.1,1.1])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.figure()
plt.plot(timebase,prediction_proba_GNB_L,label='L-mode')
plt.plot(timebase,prediction_proba_GNB_H,label='H-mode')
plt.plot(timebase,prediction_proba_GNB_I,label='I-mode')  
plt.ylabel(r'Prediction Probability (GNB)')
plt.xlabel(r'time (s)')
plt.ylim([-0.1,1.1])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.figure()
plt.plot(timebase,prediction_proba_LR_L,label='L-mode')
plt.plot(timebase,prediction_proba_LR_H,label='H-mode')
plt.plot(timebase,prediction_proba_LR_I,label='I-mode')  
plt.ylabel(r'Prediction Probability (LR)')
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
plt.plot(timebase,(data['li']),color='green')
plt.ylabel(r"$l_{i}$")
plt.xlabel(r'time (s)')
plt.show()
plt.figure()
plt.plot(timebase,(data['rmag']),color='green')
plt.ylabel(r"$r_{mag}$")
plt.xlabel(r'time (s)')
plt.show()
plt.figure()
plt.plot(timebase,(data['Halpha']),color='red')
plt.ylabel(r"$H_{\alpha}$")
plt.xlabel(r'time (s)')
plt.show()
plt.figure()
plt.plot(timebase,(data['b_bot_mks']),color='red')
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

#plt.figure()
#plt.plot(timebase,prediction_RF,color='black',label='Random Forest')
#plt.plot(timebase,prediction_NN,label='Neural Net')
#plt.plot(timebase,prediction_GNB,label='Gaussian naive Bayes')
#plt.plot(timebase,prediction_LR,label='Logistic Regression') 
#plt.ylabel(r'Prediction')
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.ylim([-0.1,1.1])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,prediction_proba_RF,color='black',label='Random Forest')
#plt.plot(timebase,prediction_proba_NN,label='Neural Net')
#plt.plot(timebase,prediction_proba_GNB,label='Gaussian naive Bayes')
#plt.plot(timebase,prediction_proba_LR,label='Logistic Regression') 
#plt.ylabel(r'Prediction Probability')
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.ylim([-0.1,1.1])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['Wmhd']),color='green')
#plt.ylabel(r"$W_{mhd}$")
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['nebar_efit']),color='green')
#plt.ylabel(r"$\bar{n}$")
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['beta_p']),color='green')
#plt.ylabel(r"$\beta_p$")
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['P_ohm']),color='green')
#plt.ylabel(r"$P_{ohm}$")
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['li']),color='green')
#plt.ylabel(r"$l_{i}$")
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['rmag']),color='green')
#plt.ylabel(r"$r_{mag}$")
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['Halpha']),color='red')
#plt.ylabel(r"$H_{\alpha}$")
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['b_bot_mks']),color='red')
#plt.ylabel(r'b_bot (mks)')
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(extra_variables['gpc2_te0']),color='red')
#plt.ylabel(r'gpc2_te0')
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(extra_variables['gpc_te8']),color='red')
#plt.ylabel(r'gpc_te8')
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(extra_variables['NL_04']),color='red')
#plt.ylabel(r'NL_04')
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(extra_variables['ne_t']),color='red')
#plt.ylabel(r'ne_t')
#plt.xlabel(r'time (s)')
#plt.xlim([1.5,1.8])
#plt.show()