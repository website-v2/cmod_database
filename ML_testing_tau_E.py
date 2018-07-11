# -*- coding: utf-8 -*-
""" 
Created on Wed Jun 13 14:33:32 2018

@author: mathewsa 

This code is only used for multi-class regression of  confinement time (tau_E) for
multiple modes (L, H, and I) using supervised machine learning methods from scikit learn
""" 
from sklearn.ensemble import RandomForestRegressor 
from sklearn import neighbors
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_blobs 
from sklearn.calibration import calibration_curve 
from sklearn.metrics import log_loss 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler   
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import mean_squared_error, r2_score
import itertools
import operator
import random
import sqlite3
from datetime import datetime
import numpy as np
import sys
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
tau_E_data = []
bad_shot = 0 #initialize
i = 0 
while i < len(rows):  
    if (values['ip'][i] != None) and (values['btor'][i] != None) and (values['Wmhd'][i]) > 0. and (values['nebar_efit'][i] != None) and (values['beta_p'][i] != None) and (values['P_ohm'][i] > 0.) and (values['li'][i] != None) and (values['rmag'][i] != None) and (values['Halpha'][i] != None) and (values['psurfa'][i] != None) and (values['p_icrf'][i] > 0.):
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
        tau_E_data.append([(values['shot'])[i],(values['Wmhd'])[i],(values['p_icrf'])[i],(values['P_ohm'])[i]])
        #the above for loop just ensures there is stored value for
        #those quantities being indexed, otherwise skip that column 
    if (((values['q95'])[i]) < 2.0) or (((values['li'])[i]) < 1.0) or (((values['e_bot_mks'])[i]) > 200.0):
        if rows[i][0] != bad_shot:
            print('Possibly check ',rows[i][0])
            bad_shot = rows[i][0] 
    i = i + 1
    
Nth = -5
tau_E = np.array(tau_E_data)[:,1]/(np.array(tau_E_data)[:,2] + np.array(tau_E_data)[:,3])
list_tau_E = list(tau_E)
sorted_tau_E = tau_E.sort()
index_N = list_tau_E.index(tau_E[Nth]) #finds index for 5th largest element
print('Nth largest tau_E',tau_E_data[index_N])

plt.xlabel(r"$\tau_E$")
plt.ylabel('Counts')
n, bins, patches = plt.hist(tau_E, 500, facecolor='g', alpha=0.75)
plt.axis([0., 0.2, 0., 0.5*len(tau_E)])
plt.grid(True)
plt.show() 

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
 
variance_KNN = []
variance_valid_KNN = [] 
variance_RFR = []
variance_valid_RFR = []
variance_REG = []
variance_valid_REG = []
variance_NN = []
variance_valid_NN = []

fraction_ = 0.80
train_valid_frac = 0.80
update_index = 0#(spectroscopy.getNode('\SPECTROSCOPY::z_ave')).units_of()
cycles = 100 #runs
while update_index < cycles:
    print('Fraction of total data for training + validation = ',train_valid_frac)
    print('Fraction of training + validation data used for training = ',fraction_)
    #use below 4 lines if randomizing shots AND time slices for train/validation set
    print("ML_testing_all_normalized_NN_100x100x100_layers_([(values['shot'])[i],(values['Wmhd'])[i],(values['nebar_efit'])[i],(values['beta_p'])[i],\
                        (values['P_ohm'])[i],(values['li'])[i],(values['rmag'])[i],(values['Halpha'])[i]]), cycles =",cycles,\
    shots_number,' distinct shots in this dataset being considered',\
    'H-mode fraction to total dataset time slices: ',p,'/',len(Y_data0),\
    'I-mode fraction to total dataset time slices: ',p_i,'/',len(Y_data0))    
    data = np.insert(X_data0, len(X_data0[0]), values=tau_E, axis=-1)
    together = [list(i) for _, i in itertools.groupby(data, operator.itemgetter(0))]
    random.shuffle(together) #groups based on first item of x_data, which should be shot!
    final_random = [i for j in together for i in j]
    X_data = (np.array(final_random))[:,1:-1]
    Y_data = (np.array(final_random, dtype = int))[:,-1]
    #this train_valid is data that is training, then to be validated
    X_train, y_train = X_data[:int(train_valid_frac*len(X_data))], tau_E [:int(train_valid_frac*len(X_data))]
    
    X_train, y_train = shuffle(X_train, y_train, random_state=0) #or use this single line; random_state allows predictable output

    X_train_valid, y_train_valid = X_train[:int(fraction_*len(X_train))], y_train[:int(fraction_*len(X_train))]
    X_valid, y_valid = X_train[int(fraction_*len(X_train)):], y_train[int(fraction_*len(X_train)):]
    X_test, y_test = X_data[int(train_valid_frac*len(X_data)):], Y_data[int(train_valid_frac*len(X_data)):]
    y_valid_np = np.array([int(numeric_string) for numeric_string in y_valid])
    y_test_np = np.array([int(numeric_string) for numeric_string in y_test])
    y_train_np = np.array([int(numeric_string) for numeric_string in y_train_valid])
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_validNN = scaler.transform(X_train_valid)
    X_validNN = scaler.transform(X_valid)
    X_testNN = scaler.transform(X_test)
    
    #uncommenting below 3 lines ensures all algorithms get normalized data
    #this means subtracting mean and dividing by standard deviation
    X_train_valid = X_train_validNN
    X_valid = X_validNN
    X_test = X_testNN  
    
    n_neighbors = 10
    
    rfr = RandomForestRegressor(n_estimators=100,max_features="sqrt")
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
    reg = linear_model.Ridge (alpha = .5)
    mlp = MLPRegressor(hidden_layer_sizes=(100,100,100))
    
    prediction = {}
    prediction_valid = {}
     
    for clf, name in [(knn, 'KNeighborsRegressor'),
                  (reg, 'Ridge Regression'),
#                  (svc, 'Support Vector Classification'),
                  (rfr, 'Random Forest'),
                  (mlp, 'NeuralNet')]: 
        clf.fit(X_train_valid, y_train_valid)
        prediction[str(name)] = clf.predict(X_test)
        avg_error = mean_squared_error(y_test_np, prediction[str(name)])
        print("Mean squared error (test): ",avg_error,name)
        prediction_valid[str(name)] = clf.predict(X_valid)
        avg_error_valid = mean_squared_error(y_valid_np, prediction_valid[str(name)])
        print("Mean squared error (valid): ",avg_error_valid,name)
        if name == 'KNeighborsRegressor':
            variance_KNN.append(avg_error)
            variance_valid_KNN.append(avg_error_valid)
        if name == 'Ridge Regression':    
            variance_RFR.append(avg_error)
            variance_valid_RFR.append(avg_error_valid)
        if name == 'Random Forest':
            variance_REG.append(avg_error)
            variance_valid_REG.append(avg_error_valid)
        if name == 'NeuralNet':
            variance_NN.append(avg_error)
            variance_valid_NN.append(avg_error_valid)
    update_index = update_index + 1
     
print('precision_KNN_tau_E:', np.mean(variance_KNN),' +/- ',np.std(variance_KNN))
print('precision_valid_KNN_tau_E:', np.mean(variance_valid_KNN),' +/- ',np.std(variance_valid_KNN))
print('precision_RFR_tau_E:', np.mean(variance_RFR),' +/- ',np.std(variance_RFR))
print('precision_valid_RFR_tau_E:', np.mean(variance_valid_RFR),' +/- ',np.std(variance_valid_RFR))
print('precision_REG_tau_E:', np.mean(variance_REG),' +/- ',np.std(variance_REG))
print('precision_valid_REG_tau_E:', np.mean(variance_valid_REG),' +/- ',np.std(variance_valid_REG))
print('precision_NN_tau_E:', np.mean(variance_NN),' +/- ',np.std(variance_NN))
print('precision_valid_NN_tau_E:', np.mean(variance_valid_NN),' +/- ',np.std(variance_valid_NN))

import pickle
#Saving created model
RFR_tau_E_pkl_filename = '/home/mathewsa/Desktop/RFR_regression_tau_E.pkl'
RFR_tau_E_model_pkl = open(RFR_tau_E_pkl_filename, 'wb')
pickle.dump(rfr, RFR_tau_E_model_pkl)
RFR_tau_E_model_pkl.close()
KNN_tau_E_pkl_filename = '/home/mathewsa/Desktop/KNN_regression_tau_E.pkl'
KNN_tau_E_model_pkl = open(KNN_tau_E_pkl_filename, 'wb')
pickle.dump(knn, KNN_tau_E_model_pkl)
KNN_tau_E_model_pkl.close()
REG_tau_E_pkl_filename = '/home/mathewsa/Desktop/REG_regression_tau_E.pkl'
REG_tau_E_model_pkl = open(REG_tau_E_pkl_filename, 'wb')
pickle.dump(reg, REG_tau_E_model_pkl)
REG_tau_E_model_pkl.close()
NN_tau_E_pkl_filename = '/home/mathewsa/Desktop/NN_regression_tau_E.pkl'
NN_tau_E_model_pkl = open(NN_tau_E_pkl_filename, 'wb')
pickle.dump(mlp, NN_tau_E_model_pkl)
NN_tau_E_model_pkl.close()
scalerfile = '/home/mathewsa/Desktop/scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))

#Loading saved model
#RF_LHI_model_pkl = open(RF_LHI_pkl_filename, 'rb')
#RF_LHI_model = pickle.load(RF_LHI_model_pkl)
#print("Loaded model :: ", RF_LHI_model)
#scalerfile = '/home/mathewsa/Desktop/scaler.sav'
#scaler = pickle.load(open(scalerfile, 'rb')) 