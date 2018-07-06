# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:12:24 2018

@author: mathewsa

Currently configured to conduct analysis/prediction on a single shot

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
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid",{'axes.facecolor': 'white','axes.grid': True,}) 
from matplotlib.ticker import FuncFormatter

def y_fmt(y, pos):
    decades = [1e21, 1e20, 1e19, 1e18, 1e17, 1e16, 1e15, 1e14, 1e13, 1e12, 1e11,\
    1e10, 1e9, 1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3,\
    1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12,]
    suffix  = [r"$\! \times 10^{21}$", r"$\! \times  10^{20}$", r"$\! \times  10^{19}$",\
    r"$\! \times 10^{18}$", r"$\! \times  10^{17}$", r"$\! \times  10^{16}$", r"$\! \times 10^{15}$",\
    r"$\! \times  10^{14}$", r"$\! \times  10^{13}$", r"$\! \times  10^{12}$",\
    r"$\! \times 10^{11}$", r"$\! \times 10^{10}$", r"$\! \times 10^{9}$",\
    r"$\! \times  10^{8}$", r"$\! \times  10^{7}$", r"$\! \times 10^{6}$", r"$\! \times  10^{5}$",\
    r"$\! \times  10^{4}$", r"$\! \times 10^{3}$", r"$\! \times  10^{2}$", r"$\! \times  10^{1}$", "" ,\
    r"$\! \times  10^{-1}$", r"$\! \times  10^{-2}$", r"$\! \times 10^{-3}$",\
    r"$\! \times  10^{-4}$", r"$\! \times  10^{-5}$", r"$\! \times 10^{-6}$",\
    r"$\! \times  10^{-7}$", r"$\! \times  10^{-8}$", r"$\! \times 10^{-9}$",\
    r"$\! \times  10^{-10}$", r"$\! \times  10^{-11}$", r"$\! \times  10^{-12}$"]
    if y == 0:
        return str(0)
    for i, d in enumerate(decades):
        if np.abs(y) >=d:
            val = np.around(y/float(d),decimals=1)
            signf = len(str(val).split(".")[1])
            if signf == 0:
                return '{val:d} {suffix}'.format(val=int(val), suffix=suffix[i])
            else:
                if signf == 1:
#                    print val, signf
                    if str(val).split(".")[1] == "0":
                       return '{val:d} {suffix}'.format(val=int(round(val)), suffix=suffix[i]) 
                tx = "{"+"val:.{signf}f".format(signf = signf) +"} {suffix}"
                return tx.format(val=val, suffix=suffix[i])

                #return y
    return y
    
start_time = 0.383 #seconds
end_time = 1.744 #seconds
timebase = np.arange(round(start_time,3),round(end_time,3),0.001) #using 1 millisecond constant interval
shot = 1160930033
tstart = 1000.*start_time #milliseconds
dt = 1.0 #milliseconds
tend = 1000.*end_time #milliseconds     
fast_efit06.main(shot,tstart,tend,dt) 
#data_acquire_MDSplus.main(shot,timebase,path_shot)
#data = np.load('/home/mathewsa/Desktop/single_shot_training_table_py.npz')
#extra_variables = np.load('/home/mathewsa/Desktop/extra_variables.npz')
#
#X_data = []
#j = 0
#time = round(((extra_variables['timebase'])[j]),3)
#while time < round(((extra_variables['timebase'])[-1]),3):  
#    time = round(((extra_variables['timebase'])[j]),3) 
#    X_data.append([(data['Wmhd'])[j],(data['nebar_efit'])[j],\
#    (data['beta_p'])[j],(data['P_ohm'])[j],(data['li'])[j],\
#    (data['rmag'])[j],(data['Halpha'])[j]])
#    j = j + 1
    
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

##score = RF_LH_model.score(X_test, y_test)
#X_data_normalized = scaler.transform(X_data)
#prediction_RF = RF_LHI_model.predict(X_data_normalized)
#prediction_proba_RF_L = RF_LHI_model.predict_proba(X_data_normalized)[:,0]
#prediction_proba_RF_H = RF_LHI_model.predict_proba(X_data_normalized)[:,1]
#prediction_proba_RF_I = RF_LHI_model.predict_proba(X_data_normalized)[:,2]
#prediction_NN = NN_LHI_model.predict(X_data_normalized)
#prediction_proba_NN_L = NN_LHI_model.predict_proba(X_data_normalized)[:,0]
#prediction_proba_NN_H = NN_LHI_model.predict_proba(X_data_normalized)[:,1]
#prediction_proba_NN_I = NN_LHI_model.predict_proba(X_data_normalized)[:,2]
#prediction_GNB = GNB_LHI_model.predict(X_data_normalized)
#prediction_proba_GNB_L = GNB_LHI_model.predict_proba(X_data_normalized)[:,0]
#prediction_proba_GNB_H = GNB_LHI_model.predict_proba(X_data_normalized)[:,1]
#prediction_proba_GNB_I = GNB_LHI_model.predict_proba(X_data_normalized)[:,2]
#prediction_LR = LR_LHI_model.predict(X_data_normalized)
#prediction_proba_LR_L = LR_LHI_model.predict_proba(X_data_normalized)[:,0]
#prediction_proba_LR_H = LR_LHI_model.predict_proba(X_data_normalized)[:,1]
#prediction_proba_LR_I = LR_LHI_model.predict_proba(X_data_normalized)[:,2]
#    
#    
#plt.figure()
#plt.plot(timebase,prediction_proba_RF_H,color='black',label='Random Forest')
#plt.plot(timebase,prediction_proba_NN_H,label='Neural Net')
#plt.plot(timebase,prediction_proba_GNB_H,label='Gaussian naive Bayes')
#plt.plot(timebase,prediction_proba_LR_H,label='Logistic Regression') 
#plt.ylabel(r'Prediction Probability (H-mode)')
#plt.xlabel(r'time (s)')
#plt.ylim([-0.1,1.1])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,prediction_proba_RF_I,color='black',label='Random Forest')
#plt.plot(timebase,prediction_proba_NN_I,label='Neural Net')
#plt.plot(timebase,prediction_proba_GNB_I,label='Gaussian naive Bayes')
#plt.plot(timebase,prediction_proba_LR_I,label='Logistic Regression') 
#plt.ylabel(r'Prediction Probability (I-mode)')
#plt.xlabel(r'time (s)') 
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,prediction_proba_RF_L,label='L-mode')
#plt.plot(timebase,prediction_proba_RF_H,label='H-mode')
#plt.plot(timebase,prediction_proba_RF_I,label='I-mode')  
#plt.ylabel(r'Prediction Probability (RF)')
#plt.xlabel(r'Time (s)')
#plt.ylim([-0.1,1.1])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,prediction_proba_NN_L,label='L-mode')
#plt.plot(timebase,prediction_proba_NN_H,label='H-mode')
#plt.plot(timebase,prediction_proba_NN_I,label='I-mode')  
#plt.ylabel(r'Prediction Probability (NN)')
#plt.xlabel(r'Time (s)')
#plt.ylim([-0.1,1.1])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,prediction_proba_GNB_L,label='L-mode')
#plt.plot(timebase,prediction_proba_GNB_H,label='H-mode')
#plt.plot(timebase,prediction_proba_GNB_I,label='I-mode')  
#plt.ylabel(r'Prediction Probability (GNB)')
#plt.xlabel(r'Time (s)')
#plt.ylim([-0.1,1.1])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,prediction_proba_LR_L,label='L-mode')
#plt.plot(timebase,prediction_proba_LR_H,label='H-mode')
#plt.plot(timebase,prediction_proba_LR_I,label='I-mode')  
#plt.ylabel(r'Prediction Probability (LR)')
#plt.xlabel(r'Time (s)')
#plt.ylim([-0.1,1.1])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['Wmhd']),color='green')
#plt.ylabel(r"$W_{mhd}$")
#plt.xlabel(r'Time (s)')
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['nebar_efit']),color='green') 
#plt.ylabel(r"$\bar{n}$")
#plt.xlabel(r'Time (s)')
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['beta_p']),color='green')
#plt.ylabel(r"$\beta_p$")
#plt.xlabel(r'Time (s)')
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['P_ohm']),color='green')
#plt.ylabel(r"$P_{ohm}$")
#plt.xlabel(r'Time (s)')
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['li']),color='green')
#plt.ylabel(r"$l_{i}$")
#plt.xlabel(r'Time (s)')
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['rmag']),color='green')
#plt.ylabel(r"$r_{mag}$")
#plt.xlabel(r'Time (s)')
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['Halpha']),color='green')
#plt.ylabel(r"$H_{\alpha}$")
#plt.xlabel(r'Time (s)')
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['b_bot_mks']),color='red')
#plt.ylabel(r'b_bot (mks)')
#plt.xlabel(r'Time (s)')
#plt.show()
#plt.figure()
#plt.plot(timebase,(extra_variables['gpc2_te0']),color='red')
#plt.ylabel(r'gpc2_te0')
#plt.xlabel(r'Time (s)')
#plt.show()
#plt.figure()
#plt.plot(timebase,(extra_variables['gpc_te8']),color='red')
#plt.ylabel(r'gpc_te8')
#plt.xlabel(r'Time (s)')
#plt.show()
#plt.figure()
#plt.plot(timebase,(extra_variables['NL_04']),color='red')
#plt.ylabel(r'NL_04')
#plt.xlabel(r'Time (s)')
#plt.show()
#plt.figure()
#plt.plot(timebase,(extra_variables['ne_t']),color='red')
#plt.ylabel(r'ne_t')
#plt.xlabel(r'Time (s)')
#plt.show()
#plt.figure()
#plt.plot(timebase,prediction_proba_RF_H,color='black',label='Random Forest')
#plt.plot(timebase,prediction_proba_NN_H,label='Neural Net')
#plt.plot(timebase,prediction_proba_GNB_H,label='Gaussian naive Bayes')
#plt.plot(timebase,prediction_proba_LR_H,label='Logistic Regression') 
#plt.ylabel(r'Prediction Probability (H-mode)')
#plt.xlabel(r'time (s)')
#plt.ylim([-0.1,1.1])
#plt.xlim([1.5,1.8])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,prediction_proba_RF_I,color='black',label='Random Forest')
#plt.plot(timebase,prediction_proba_NN_I,label='Neural Net')
#plt.plot(timebase,prediction_proba_GNB_I,label='Gaussian naive Bayes')
#plt.plot(timebase,prediction_proba_LR_I,label='Logistic Regression') 
#plt.ylabel(r'Prediction Probability (I-mode)')
#plt.xlabel(r'time (s)') 
#plt.xlim([1.5,1.8])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,prediction_proba_RF_L,label='L-mode')
#plt.plot(timebase,prediction_proba_RF_H,label='H-mode')
#plt.plot(timebase,prediction_proba_RF_I,label='I-mode')  
#plt.ylabel(r'Prediction Probability (RF)')
#plt.xlabel(r'Time (s)')
#plt.ylim([-0.1,1.1])
#plt.xlim([1.5,1.8])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,prediction_proba_NN_L,label='L-mode')
#plt.plot(timebase,prediction_proba_NN_H,label='H-mode')
#plt.plot(timebase,prediction_proba_NN_I,label='I-mode')  
#plt.ylabel(r'Prediction Probability (NN)')
#plt.xlabel(r'Time (s)')
#plt.ylim([-0.1,1.1])
#plt.xlim([1.5,1.8])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,prediction_proba_GNB_L,label='L-mode')
#plt.plot(timebase,prediction_proba_GNB_H,label='H-mode')
#plt.plot(timebase,prediction_proba_GNB_I,label='I-mode')  
#plt.ylabel(r'Prediction Probability (GNB)')
#plt.xlabel(r'Time (s)')
#plt.ylim([-0.1,1.1])
#plt.xlim([1.5,1.8])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,prediction_proba_LR_L,label='L-mode')
#plt.plot(timebase,prediction_proba_LR_H,label='H-mode')
#plt.plot(timebase,prediction_proba_LR_I,label='I-mode')  
#plt.ylabel(r'Prediction Probability (LR)')
#plt.xlabel(r'Time (s)')
#plt.ylim([-0.1,1.1])
#plt.xlim([1.5,1.8])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['Wmhd']),color='green')
#plt.ylabel(r"$W_{mhd}$")
#plt.xlabel(r'Time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['nebar_efit']),color='green') 
#plt.ylabel(r"$\bar{n}$")
#plt.xlabel(r'Time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['beta_p']),color='green')
#plt.ylabel(r"$\beta_p$")
#plt.xlabel(r'Time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['P_ohm']),color='green')
#plt.ylabel(r"$P_{ohm}$")
#plt.xlabel(r'Time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['li']),color='green')
#plt.ylabel(r"$l_{i}$")
#plt.xlabel(r'Time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['rmag']),color='green')
#plt.ylabel(r"$r_{mag}$")
#plt.xlabel(r'Time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['Halpha']),color='green')
#plt.ylabel(r"$H_{\alpha}$")
#plt.xlabel(r'Time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(data['b_bot_mks']),color='red')
#plt.ylabel(r'b_bot (mks)')
#plt.xlabel(r'Time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(extra_variables['gpc2_te0']),color='red')
#plt.ylabel(r'gpc2_te0')
#plt.xlabel(r'Time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(extra_variables['gpc_te8']),color='red')
#plt.ylabel(r'gpc_te8')
#plt.xlabel(r'Time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(extra_variables['NL_04']),color='red')
#plt.ylabel(r'NL_04')
#plt.xlabel(r'Time (s)')
#plt.xlim([1.5,1.8])
#plt.show()
#plt.figure()
#plt.plot(timebase,(extra_variables['ne_t']),color='red')
#plt.ylabel(r'ne_t')
#plt.xlabel(r'Time (s)')
#plt.xlim([1.5,1.8])
#plt.show() 

"""Creating visualization of decision boundary for general data with 
trained classifier over selected feature space (arbitrarily chosen).
The below is not specific to a particular shot, but is specific to a 
particular ID of a trained classifier."""
 
sqlite_file = '/home/mathewsa/Desktop/am_transitions.db'
table_name = 'confinement_table'
table_name_transitions = 'transitions' 

conn = sqlite3.connect(sqlite_file)
cursor = conn.cursor() 
    
cursor.execute('select shot,id,present_mode,next_mode,time,time_at_transition,ip,\
               btor,p_lh,p_icrf,p_icrf_d,p_icrf_e,p_icrf_j3,p_icrf_j4,freq_icrf_d,\
               freq_icrf_e,freq_icrf_j,beta_N,beta_p,beta_t,kappa,triang_l,triang_u,\
               triang,li,psurfa,areao,vout,aout,rout,zout,zmag,rmag,zsep_lower,zsep_upper,\
               rsep_lower,rsep_upper,zvsin,rvsin,zvsout,rvsout,upper_gap,lower_gap,\
               q0,qstar,q95,V_loop_efit,V_surf_efit,Wmhd,cpasma,ssep,P_ohm,HoverHD,\
               Halpha,Dalpha,z_ave,p_rad,p_rad_core,nLave_04,NL_04,nebar_efit,\
               piezo_4_gas_input,g_side_rat,e_bot_mks,b_bot_mks,update_time from\
               {}'.format(table_name))
rows = cursor.fetchall()   
conn.commit()
conn.close()

columns = ['shot','id','present_mode','next_mode','time','time_at_transition','ip',\
        'btor','p_lh','p_icrf','p_icrf_d','p_icrf_e','p_icrf_j3','p_icrf_j4','freq_icrf_d',\
        'freq_icrf_e','freq_icrf_j','beta_N','beta_p','beta_t','kappa','triang_l','triang_u',\
        'triang','li','psurfa','areao','vout','aout','rout','zout','zmag','rmag','zsep_lower','zsep_upper',\
        'rsep_lower','rsep_upper','zvsin','rvsin','zvsout','rvsout','upper_gap','lower_gap',\
        'q0','qstar','q95','V_loop_efit','V_surf_efit','Wmhd','cpasma','ssep','P_ohm','HoverHD',\
        'Halpha','Dalpha','z_ave','p_rad','p_rad_core','nLave_04','NL_04','nebar_efit',\
        'piezo_4_gas_input','g_side_rat','e_bot_mks','b_bot_mks','update_time']

conn = sqlite3.connect(sqlite_file)
cursor = conn.cursor() 
    
cursor.execute('select shot,id,present_mode,next_mode,time,time_at_transition from\
               {}'.format(table_name_transitions))
rows_transitions = cursor.fetchall()   
conn.commit()
conn.close()

def array3x3(array_): #converts 2x2 to 3x3
    if len(array_) == 2:
        zeros_v = np.array([0.,0.])
        zeros_h = np.array([[0.],[0.],[0.]])
        new_ = np.vstack((array_,zeros_v))
        output_ = np.hstack((new_,zeros_h))
    else:
        output_ = array_
    output_ = np.array(output_,dtype=np.int64)
    return output_
    
transitions_start = {}
transitions_end = {}
i_t = 0
prev_shot = 0
while i_t < len(rows_transitions):
    if prev_shot != rows_transitions[i_t][0]:
        transitions_start['{}'.format(rows_transitions[i_t][0])] = rows_transitions[i_t][4]
    if rows_transitions[i_t][3] == 'end':
        transitions_end['{}'.format(rows_transitions[i_t][0])] = rows_transitions[i_t][5]
    prev_shot = rows_transitions[i_t][0]
    i_t = i_t + 1
    
values = {} #dictionary where all values for each column can be found 
i_column = 0
for column in columns:
    i = 0
    values[column] = []
    while i < len(rows):
        values[column].append(rows[i][i_column])
        i = i + 1  
    i_column = i_column + 1

Y_data0 = []
X_data0 = []
total_x_data = []
bad_shot = 0 #initialize
i = 0 
while i < len(rows):  
    if (values['ip'][i] != None) and (values['btor'][i] != None) and (values['Wmhd'][i] != None) and (values['nebar_efit'][i] != None) and (values['beta_p'][i] != None) and (values['P_ohm'][i] != None) and (values['li'][i] != None) and (values['rmag'][i] != None) and (values['Halpha'][i] != None) and (values['psurfa'][i] != None):
        Y_data0.append((values['present_mode'])[i])
        X_data0.append([(values['shot'])[i],(values['Wmhd'])[i],(values['nebar_efit'])[i],(values['beta_p'])[i],
                        (values['P_ohm'])[i],(values['li'])[i],(values['rmag'])[i],(values['Halpha'])[i]]) #first element must be shot!
        total_x_data.append([(values['shot'])[i],(values['ip'])[i],(values['btor'])[i],(values['li'])[i],
              (values['q95'])[i],(values['Wmhd'])[i],(values['p_icrf'])[i],
              (values['beta_N'])[i],(values['nebar_efit'])[i],(values['beta_p'])[i],
              (values['beta_t'])[i],(values['kappa'])[i],(values['triang'])[i],
              (values['psurfa'])[i],(values['areao'])[i],(values['vout'])[i],(values['aout'])[i],
              (values['rout'])[i],(values['zout'])[i],
              (values['zmag'])[i],(values['rmag'])[i],(values['zsep_lower'])[i],
              (values['zsep_upper'])[i],(values['rsep_lower'])[i],(values['rsep_upper'])[i],
              (values['zvsin'])[i],(values['rvsin'])[i],(values['zvsout'])[i],
              (values['rvsout'])[i],(values['upper_gap'])[i],(values['lower_gap'])[i],
              (values['qstar'])[i],(values['V_loop_efit'])[i],
              (values['V_surf_efit'])[i],(values['cpasma'])[i],(values['ssep'])[i],
              (values['P_ohm'])[i],(values['NL_04'])[i],(values['g_side_rat'])[i],
              (values['e_bot_mks'])[i],(values['b_bot_mks'])[i]])
        #the above for loop just ensures there is stored value for
        #those quantities being indexed, otherwise skip that column 
    if (((values['q95'])[i]) < 2.0) or (((values['li'])[i]) < 1.0) or (((values['e_bot_mks'])[i]) > 200.0):
        if rows[i][0] != bad_shot:
            print('Possibly check ',rows[i][0])
            bad_shot = rows[i][0] 
    i = i + 1

i = 0
shots_number = 0
shot_old = 0
while i < len(X_data0):
    if X_data0[i][0] != shot_old:
        shots_number = shots_number + 1
        shot_old = X_data0[i][0]
    i = i + 1
print(shots_number,' distinct shots in this dataset being considered')


Y_data0 = np.where(np.array(Y_data0) == 'L', 0, Y_data0)
Y_data0 = np.where(Y_data0 == 'H', 1, Y_data0)
Y_data0 = np.where(Y_data0 == 'I', 2, Y_data0)
class_names = ['L','H','I']

q = 0
p = 0
while q < len(Y_data0):
    if (Y_data0[q] == '1') or (Y_data0[q] == 1):
        p = p + 1
    q = q + 1
print('H-mode fraction to total dataset time slices: ',p,'/',len(Y_data0))


q = 0
p_i = 0
while q < len(Y_data0):
    if (Y_data0[q] == '2') or (Y_data0[q] == 2):
        p_i = p_i + 1
    q = q + 1
print('I-mode fraction to total dataset time slices: ',p_i,'/',len(Y_data0))


#------------------------------------------------------
#Below is the regions as distinguished by the classifiers
#They are generalized and not specific to any shot but instead the entire database
#------------------------------------------------------

eclf = VotingClassifier(estimators=[('rf', RF_LHI_model), ('mlp', NN_LHI_model),
                                    ('lr', LR_LHI_model), ('gnb', GNB_LHI_model)],
                                    voting='soft')#, weights=[2, 1, 2])
origin = 'lower'
#origin = 'upper'                    
# Plotting decision regions
#subtract/adding 2.0 simply increases region over which predictions applied
grid_steps = 100
X_data0 = (np.array(X_data0))[:,1:] #this removes shot from feature
X_data_normalized = scaler.transform(X_data0)
Wmhd_min, Wmhd_max = X_data_normalized[:, 0].min() - 2.0, X_data_normalized[:, 0].max() + 2.0
nebar_efit_min, nebar_efit_max = X_data_normalized[:, 1].min() - 2.0, X_data_normalized[:, 1].max() + 2.0
beta_p_min, beta_p_max = X_data_normalized[:, 2].min() - 2.0, X_data_normalized[:, 2].max() + 2.0
P_ohm_min, P_ohm_max = X_data_normalized[:, 3].min() - 2.0, X_data_normalized[:, 3].max() + 2.0
li_min, li_max = X_data_normalized[:, 4].min() - 2.0, X_data_normalized[:, 4].max() + 2.0
rmag_min, rmag_max = X_data_normalized[:, 5].min() - 2.0, X_data_normalized[:, 5].max() + 2.0
Halpha_min, Halpha_max = X_data_normalized[:, 6].min() - 2.0, X_data_normalized[:, 6].max() + 2.0

#these are the actual true min values
true_values_min = scaler.inverse_transform([Wmhd_min, nebar_efit_min, beta_p_min,\
    P_ohm_min, li_min, rmag_min, Halpha_min])
true_values_max = scaler.inverse_transform([Wmhd_max, nebar_efit_max, beta_p_max,\
    P_ohm_max, li_max, rmag_max, Halpha_max])
    
min_Wmhd = true_values_min[0]
min_nebar_efit = true_values_min[1]
min_beta_p = true_values_min[2]
min_P_ohm = true_values_min[3]
min_li = true_values_min[4]
min_rmag = true_values_min[5]
min_Halpha = true_values_min[6]

max_Wmhd = true_values_max[0]
max_nebar_efit = true_values_max[1]
max_beta_p = true_values_max[2]
max_P_ohm = true_values_max[3]
max_li = true_values_max[4]
max_rmag = true_values_max[5]
max_Halpha = true_values_max[6]

#in this example, the following 2 features will be plotted on contour map
#while the other features will be set at a constant
Wmhd_, nebar_efit_ = np.meshgrid(np.linspace(Wmhd_min, Wmhd_max, grid_steps),
                     np.linspace(nebar_efit_min, nebar_efit_max, grid_steps))
Wmhd_constant = 0.
nebar_efit = 0.
beta_p_constant = 0.35
P_ohm_constant = 1700000.
li_constant = 1.15
rmag_constant = 0.68
Halpha_constant = 1.87

inputs_normalized = scaler.transform([Wmhd_constant,nebar_efit,beta_p_constant,\
                P_ohm_constant,li_constant,rmag_constant,Halpha_constant]) 

Wmhd_input = Wmhd_.ravel()
nebar_efit_input = nebar_efit_.ravel()
beta_p_input = inputs_normalized[2]*np.ones(grid_steps*grid_steps)
P_ohm_input = inputs_normalized[3]*np.ones(grid_steps*grid_steps)
li_input = inputs_normalized[4]*np.ones(grid_steps*grid_steps)
rmag_input = inputs_normalized[5]*np.ones(grid_steps*grid_steps)
Halpha_input = inputs_normalized[6]*np.ones(grid_steps*grid_steps)

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(11, 11))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [RF_LHI_model, NN_LHI_model, LR_LHI_model, GNB_LHI_model],#, eclf],
                        ['Random Forest', 'NeuralNet', 'Logistic Regression',\
                         'Gaussian naive Bayes']):#, 'Soft Voting']):

    Z = clf.predict(np.c_[Wmhd_input, nebar_efit_input, beta_p_input, P_ohm_input,
                          li_input, rmag_input, Halpha_input])
    Z = Z.reshape((grid_steps,grid_steps))

    cntr = axarr[idx[0], idx[1]].contourf(Wmhd_, nebar_efit_, Z, alpha=0.4, cmap="RdBu_r", origin = origin)
    x_ticks = np.around(np.linspace(np.min(Wmhd_),np.max(Wmhd_),9))                              
    x_labels = list(np.around(np.linspace(min_Wmhd,max_Wmhd,9),decimals=1))
    y_ticks = np.around(np.linspace(np.min(nebar_efit_),np.max(nebar_efit_),9))                                 
    y_labels = list(np.around(np.linspace(min_nebar_efit,max_nebar_efit,9),decimals=1))
    
    axarr[idx[0], idx[1]].set_xticks(x_ticks)    
    axarr[idx[0], idx[1]].set_xticklabels(x_labels,rotation=90)    
    axarr[idx[0], idx[1]].set_yticks(y_ticks)                                                           
    axarr[idx[0], idx[1]].set_yticklabels(y_labels)    
    axarr[idx[0], idx[1]].set_title(tt)
    axarr[idx[0], idx[1]].set_xlabel((r"$W_{mhd} \ (J)$"))
    axarr[idx[0], idx[1]].set_ylabel(r"$\bar{n} \ (m^{-3})$") 
    axarr[idx[0], idx[1]].set_xlim([np.min(Wmhd_),np.max(Wmhd_)])
    axarr[idx[0], idx[1]].set_ylim([np.min(nebar_efit_),np.max(nebar_efit_)])
    axarr[idx[0], idx[1]].legend(loc="upper center", ncol=2)
f.colorbar(cntr, orientation="horizontal", pad=0.25)   
plt.show() 


sns.set(style="white")

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(11, 11))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [RF_LHI_model, NN_LHI_model, LR_LHI_model, GNB_LHI_model],#, eclf],
                        ['Random Forest', 'NeuralNet', 'Logistic Regression',\
                         'Gaussian naive Bayes']):#, 'Soft Voting']):

    Z = clf.predict(np.c_[Wmhd_input, nebar_efit_input, beta_p_input, P_ohm_input,
                          li_input, rmag_input, Halpha_input])
    Z = Z.reshape((grid_steps,grid_steps))
    
    grid = np.c_[Wmhd_.ravel(), nebar_efit_.ravel()]
    probs = clf.predict_proba(np.c_[Wmhd_input, nebar_efit_input, beta_p_input, P_ohm_input,
                          li_input, rmag_input, Halpha_input])[:, 1].reshape((grid_steps,grid_steps))
    bounds=np.linspace(0.0,1.0,9)                      
    cntr = axarr[idx[0], idx[1]].contourf(Wmhd_, nebar_efit_, probs, vmin = 0.0,\
    vmax = 1.0, levels = bounds, alpha=0.4, cmap="Greens", origin = origin)  
    x_ticks = np.around(np.linspace(np.min(Wmhd_),np.max(Wmhd_),9))                              
    x_labels = list(np.around(np.linspace(min_Wmhd,max_Wmhd,9),decimals=1))
    y_ticks = np.around(np.linspace(np.min(nebar_efit_),np.max(nebar_efit_),9))                                 
    y_labels = list(np.around(np.linspace(min_nebar_efit,max_nebar_efit,9),decimals=1))
    
    axarr[idx[0], idx[1]].set_xticks(x_ticks)    
    axarr[idx[0], idx[1]].set_xticklabels(x_labels,rotation=90)    
    axarr[idx[0], idx[1]].set_yticks(y_ticks)                                                           
    axarr[idx[0], idx[1]].set_yticklabels(y_labels)    
    axarr[idx[0], idx[1]].set_title(tt)
    axarr[idx[0], idx[1]].set_xlabel((r"$W_{mhd} \ (J)$"))
    axarr[idx[0], idx[1]].set_ylabel(r"$\bar{n} \ (m^{-3})$") 
    axarr[idx[0], idx[1]].set_xlim([np.min(Wmhd_),np.max(Wmhd_)])
    axarr[idx[0], idx[1]].set_ylim([np.min(nebar_efit_),np.max(nebar_efit_)])
    axarr[idx[0], idx[1]].legend(loc="upper center", ncol=2) 
f.colorbar(cntr, orientation="horizontal", pad=0.25)   
plt.show()  