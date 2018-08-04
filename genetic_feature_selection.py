# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:55:14 2018

@author: mathewsa
"""
from __future__ import print_function
import numpy as np
from sklearn import datasets, linear_model
from genetic_selection import GeneticSelectionCV
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE 
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import itertools
from sklearn.metrics import roc_curve, auc
import operator
import random
import sqlite3
from datetime import datetime 
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
    if (values['ip'][i] != None) and (values['btor'][i] != None) and (values['Wmhd'][i] > 0.) and (values['nebar_efit'][i] > 0.) and (values['beta_p'][i] > 0.) and (values['li'][i] != None) and (values['rmag'][i] != None) and (values['p_icrf'][i] != None) and (values['Halpha'][i] != None) and (values['b_bot_mks'][i] != None):
        Y_data0.append((values['present_mode'])[i])
        X_data0.append([(values['shot'])[i],(values['ip'])[i],(values['btor'])[i],(values['li'])[i],
              (values['q95'])[i],(values['Wmhd'])[i],(values['p_icrf'])[i],
              (values['nebar_efit'])[i],(values['beta_p'])[i],
              (values['kappa'])[i],(values['triang'])[i],(values['ssep'])[i],
              (values['P_ohm'])[i],
              (values['b_bot_mks'])[i],(values['Halpha'])[i]]) #first element must be shot!
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
              (values['e_bot_mks'])[i],(values['b_bot_mks'])[i],(values['Halpha'])[i]])
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

L_index = [i for i,val in enumerate(Y_data0) if val=='0']
H_index = [i for i,val in enumerate(Y_data0) if val=='1']
I_index = [i for i,val in enumerate(Y_data0) if val=='2']

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

print(shots_number,' distinct shots in this dataset being considered',\
'H-mode fraction to total dataset time slices: ',p,'/',len(Y_data0),\
'I-mode fraction to total dataset time slices: ',p_i,'/',len(Y_data0))    
data = np.insert(X_data0, len(X_data0[0]), values=Y_data0, axis=-1)
together = [list(i) for _, i in itertools.groupby(data, operator.itemgetter(0))]
random.shuffle(together) #groups based on first item of x_data, which should be shot!
final_random = [i for j in together for i in j]
X_data = (np.array(final_random))[:,1:-1]
Y_data = (np.array(final_random, dtype = int))[:,-1]
scaler = StandardScaler()
scaler.fit(X_data)
X_data_v2 = scaler.transform(X_data)


X = X_data_v2
y = Y_data

estimator = linear_model.LogisticRegression()
 
selector = GeneticSelectionCV(estimator,
                              cv=5,
                              verbose=1,
                              scoring="accuracy",
                              n_population=50,
                              crossover_proba=0.5,
                              mutation_proba=0.2,
                              n_generations=40,
                              crossover_independent_proba=0.5,
                              mutation_independent_proba=0.05,
                              tournament_size=3,
                              caching=True,
                              n_jobs=-1)
selector = selector.fit(X, y)

print(selector.support_)