#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 02:20:13 2018

@author: Abhilash

This code is only used for binary classification of L-H confinement
modes using supervised machine learning methods from scikit learn
""" 
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE 
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC, SVC
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
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
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
    if (values['ip'][i] != None) and (values['btor'][i] != None) and (values['Wmhd'][i] != None) and (values['nebar_efit'][i] != None) and (values['beta_p'][i] != None) and (values['P_ohm'][i] != None) and (values['li'][i] != None) and (values['rmag'][i] != None) and (values['Halpha'][i] != None):
        if values['present_mode'][i] != 'I': #not considering I-modes right now
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
class_names = ['L','H']

#use below 4 lines if randomizing shots AND time slices
#data = np.insert(X_data, len(X_data[0]), values=Y_data, axis=-1)
#random = (np.random.permutation(data))
#X_data = random[:,:-1]
#Y_data = random[:,-1]

##lb = preprocessing.OneHotEncoder()
##Y_data = lb.fit(Y_data)
#X_train, y_train = X_data[:int(0.6*len(X_data))], Y_data[:int(0.6*len(X_data))]
#X_valid, y_valid = X_data[int(0.6*len(X_data)):int(0.8*len(X_data))], Y_data[int(0.6*len(X_data)):int(0.8*len(X_data))]
#X_train_valid, y_train_valid = X_data[:int(0.8*len(X_data))], Y_data[:int(0.8*len(X_data))]
#X_test, y_test = X_data[int(0.8*len(X_data)):], Y_data[int(0.8*len(X_data)):]
#y_test_np = np.array([int(numeric_string) for numeric_string in y_test])

q = 0
p = 0
while q < len(Y_data0):
    if (Y_data0[q] == '1') or (Y_data0[q] == 1):
        p = p + 1
    q = q + 1
print('H-mode fraction to total dataset time slices: ',p,'/',len(Y_data0))

#train, test, and validation set
#based on taking (train_valid_frac*100)% of total data and randomly partitioning fraction_ into training and validation
#the other (1 - train_valid_frac)*100% of total (ordered) data is just for validation
true_L_RF = []
true_H_RF = []
false_L_RF = []
false_H_RF = []
true_L_NB = []
true_H_NB = []
false_L_NB = []
false_H_NB = []
true_L_NN = []
true_H_NN = []
false_L_NN = []
false_H_NN = []
true_L_Logistic = []
true_H_Logistic = []
false_L_Logistic = []
false_H_Logistic = []

RF_00 = []
RF_01 = []
RF_10  = []
RF_11  = []
NB_00 = []
NB_01 = []
NB_10  = []
NB_11  = []
NN_00 = []
NN_01 = []
NN_10  = []
NN_11  = []
LR_00 = []
LR_01 = []
LR_10  = []
LR_11  = []

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fraction_ = 0.80
train_valid_frac = 0.80
update_index = 0#(spectroscopy.getNode('\SPECTROSCOPY::z_ave')).units_of()
cycles = 100 #runs
while update_index < cycles:
    print('Fraction of total data for training + validation = ',train_valid_frac)
    print('Fraction of training + validation data used for training = ',fraction_)
    #use below 4 lines if randomizing shots AND time slices for train/validation set
    print("ML_testing_all_normalized_100trees_NN_100x100x100_layers_([(values['shot'])[i],(values['Wmhd'])[i],(values['nebar_efit'])[i],(values['beta_p'])[i],\
                            (values['P_ohm'])[i],(values['li'])[i],(values['rmag'])[i],(values['Halpha'])[i]]), cycles =",cycles,\
                    shots_number,' distinct shots in this dataset being considered',\
                    'H-mode fraction to total dataset time slices: ',p,'/',len(Y_data0))    
    data = np.insert(X_data0, len(X_data0[0]), values=Y_data0, axis=-1)
    together = [list(i) for _, i in itertools.groupby(data, operator.itemgetter(0))]
    random.shuffle(together) #groups based on first item of x_data, which should be shot!
    final_random = [i for j in together for i in j]
    X_data = (np.array(final_random))[:,1:-1]
    Y_data = (np.array(final_random))[:,-1]
    #this train_valid is data that is training, then to be validated
    X_train, y_train = X_data[:int(train_valid_frac*len(X_data))], Y_data[:int(train_valid_frac*len(X_data))]
    
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
    
    # Create classifiers
    lr = LogisticRegression()
    gnb = GaussianNB() 
    rfc = RandomForestClassifier(n_estimators=100,max_features="sqrt")
    mlp = MLPClassifier(hidden_layer_sizes=(100,100,100))
    
    tree_depth_max = [estimator.tree_.depth for estimator in rfc.estimators_]
    
    # Plot calibration plots
    plt.figure(figsize=(10, 14))
    ax1 = plt.subplot2grid((6, 1), (0, 0))
    ax2 = plt.subplot2grid((6, 1), (1, 0))
    ax3 = plt.subplot2grid((6, 1), (2, 0))
    ax4 = plt.subplot2grid((6, 1), (3, 0))
    ax5 = plt.subplot2grid((6, 1), (4, 0))
    ax6 = plt.subplot2grid((6, 1), (5, 0))
    
    prediction_prob = {}
    prediction = {}
    prediction_prob_valid = {}
    prediction_valid = {}
    sum_array = {}
    accuracy = {}
    c_matrix = {} #confusion matrix
    c_matrix1 = {}
    sum_array_valid = {}
    accuracy_valid = {}
    c_matrix_valid = {} #confusion matrix
    c_matrix1_valid = {}
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax3.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (gnb, 'Naive Bayes'),
    #                  (svc, 'Support Vector Classification'),
                      (rfc, 'Random Forest'),
                      (mlp, 'NeuralNet')]:
        if name == 'NeuralNet1':
            clf.fit(X_train_validNN, y_train_valid)
            prob_pos = clf.predict_proba(X_testNN)[:, 1] #probability of 1, or H-mode
            prob_pos_valid = clf.predict_proba(X_validNN)[:, 1]
            prediction_prob[str(name)] = np.array([int(numeric_string) for numeric_string in prob_pos])
            prediction_prob_valid[str(name)] = np.array([int(numeric_string) for numeric_string in prob_pos_valid])
            prediction[str(name)] = np.array([int(numeric_string) for numeric_string in clf.predict(X_testNN)])
            prediction_valid[str(name)] = np.array([int(numeric_string) for numeric_string in clf.predict(X_validNN)])
            predictions = mlp.predict(X_testNN)
            print(confusion_matrix(y_test,predictions))
            print(classification_report(y_test,predictions))
        else:
            clf.fit(X_train_valid, y_train_valid)
#        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1] #probability of 1, or H-mode
            prob_pos_valid = clf.predict_proba(X_valid)[:, 1]
            prediction_prob[str(name)] = np.array([int(numeric_string) for numeric_string in prob_pos])
            prediction_prob_valid[str(name)] = np.array([int(numeric_string) for numeric_string in prob_pos_valid])
            prediction[str(name)] = np.array([int(numeric_string) for numeric_string in clf.predict(X_test)])
            prediction_valid[str(name)] = np.array([int(numeric_string) for numeric_string in clf.predict(X_valid)])
        
        sum_array[str(name)] = y_test_np + prediction[str(name)]
        sum_array_valid[str(name)] = y_valid_np + prediction_valid[str(name)]
        accuracy[str(name)] = (sum(i == 2 for i in sum_array[str(name)]) + sum(i == 0 for i in sum_array[str(name)]))/float(len(y_test))
        accuracy_valid[str(name)] = (sum(i == 2 for i in sum_array_valid[str(name)]) + sum(i == 0 for i in sum_array_valid[str(name)]))/float(len(y_valid))
        c_matrix[str(name)] = confusion_matrix(y_test_np, prediction[str(name)])
        c_matrix_valid[str(name)] = confusion_matrix(y_valid_np, prediction_valid[str(name)])
        c_matrix1[str(name)+' predicted L-mode correctly'] = c_matrix[str(name)][0][0]/float(c_matrix[str(name)][0][0] + (c_matrix[str(name)][1][0]))
        c_matrix1[str(name)+' predicted H-mode correctly'] = c_matrix[str(name)][1][1]/float(c_matrix[str(name)][1][1] + (c_matrix[str(name)][0][1]))
        c_matrix1_valid[str(name)+' predicted L-mode correctly'] = c_matrix_valid[str(name)][0][0]/float(c_matrix_valid[str(name)][0][0] + (c_matrix_valid[str(name)][1][0]))
        c_matrix1_valid[str(name)+' predicted H-mode correctly'] = c_matrix_valid[str(name)][1][1]/float(c_matrix_valid[str(name)][1][1] + (c_matrix_valid[str(name)][0][1]))
    #2 is H-mode correctly classified; 0 is L-mode correctly classified; 1 is misclassfication     
        
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=20)
        fraction_of_positives_valid, mean_predicted_value_valid = calibration_curve(y_valid, prob_pos_valid, n_bins=20)
    
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                     label="%s test" % (name, ))
        ax2.hist(prob_pos, range=(0, 1), bins=20, label=name,
                     histtype="step", lw=2) 
        
        ax3.plot(mean_predicted_value_valid, fraction_of_positives_valid, "s-",
                     label="%s valid" % (name, ))
        ax4.hist(prob_pos_valid, range=(0, 1), bins=20, label=name,
                     histtype="step", lw=2)
    
        fpr, tpr, thresholds = roc_curve(y_test_np,prob_pos)     
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax5.plot(fpr, tpr, lw=1, alpha=0.4,
             label='ROC fold %s (AUC = %0.2f)' % (name, roc_auc))
        ax6.plot(fpr, tpr, lw=1, alpha=0.4,
         label='ROC fold %s (AUC = %0.2f)' % (name, roc_auc))          
             
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots - testing (reliability curve)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2) 
            
    ax3.set_ylabel("Fraction of positives")
    ax3.set_ylim([-0.05, 1.05])
    ax3.legend(loc="lower right")
    ax3.set_title('Calibration plots - validation (reliability curve)')
    ax4.set_xlabel("Mean predicted value")
    ax4.set_ylabel("Count")
    ax4.legend(loc="upper center", ncol=2)   
    
    ax5.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.4)
    ax6.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.4)
         
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax5.plot(mean_fpr, mean_tpr, color='black',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.4)
    ax6.plot(mean_fpr, mean_tpr, color='black',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.4)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax5.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    ax6.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    ax5.set_xlim([0.0, 1.05])
    ax5.set_ylim([0.0, 1.05])
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.set_title('Receiver Operating Characteristic')
    ax5.legend(loc="lower right")
    
    ax6.set_xlim([0.0, 0.10])
    ax6.set_ylim([0.0, 1.05])
    ax6.set_xlabel('False Positive Rate')
    ax6.set_ylabel('True Positive Rate')
    ax6.set_title('Receiver Operating Characteristic')
    ax6.legend(loc="lower right")
    
    plt.tight_layout() 
    plt.show() 
    
    true_L_RF.append(c_matrix['Random Forest'][0][0]/float(c_matrix['Random Forest'][0][0] + c_matrix['Random Forest'][0][1])) #true negative
    false_H_RF.append(c_matrix['Random Forest'][0][1]/float(c_matrix['Random Forest'][0][0] + c_matrix['Random Forest'][0][1])) #false positive rate
    false_L_RF.append(c_matrix['Random Forest'][1][0]/float(c_matrix['Random Forest'][1][0] + c_matrix['Random Forest'][1][1])) #false negative rate
    true_H_RF.append(c_matrix['Random Forest'][1][1]/float(c_matrix['Random Forest'][1][0] + c_matrix['Random Forest'][1][1])) #true positive
    true_L_NB.append(c_matrix['Naive Bayes'][0][0]/float(c_matrix['Naive Bayes'][0][0] + c_matrix['Naive Bayes'][0][1]))
    false_H_NB.append(c_matrix['Naive Bayes'][0][1]/float(c_matrix['Naive Bayes'][0][0] + c_matrix['Naive Bayes'][0][1]))
    false_L_NB.append(c_matrix['Naive Bayes'][1][0]/float(c_matrix['Naive Bayes'][1][0] + c_matrix['Naive Bayes'][1][1]))
    true_H_NB.append(c_matrix['Naive Bayes'][1][1]/float(c_matrix['Naive Bayes'][1][0] + c_matrix['Naive Bayes'][1][1]))
    true_L_NN.append(c_matrix['NeuralNet'][0][0]/float(c_matrix['NeuralNet'][0][0] + c_matrix['NeuralNet'][0][1]))
    false_H_NN.append(c_matrix['NeuralNet'][0][1]/float(c_matrix['NeuralNet'][0][0] + c_matrix['NeuralNet'][0][1]))
    false_L_NN.append(c_matrix['NeuralNet'][1][0]/float(c_matrix['NeuralNet'][1][0] + c_matrix['NeuralNet'][1][1]))
    true_H_NN.append(c_matrix['NeuralNet'][1][1]/float(c_matrix['NeuralNet'][1][0] + c_matrix['NeuralNet'][1][1]))
    true_L_Logistic.append(c_matrix['Logistic'][0][0]/float(c_matrix['Logistic'][0][0] + c_matrix['Logistic'][0][1]))
    false_H_Logistic.append(c_matrix['Logistic'][0][1]/float(c_matrix['Logistic'][0][0] + c_matrix['Logistic'][0][1]))
    false_L_Logistic.append(c_matrix['Logistic'][1][0]/float(c_matrix['Logistic'][1][0] + c_matrix['Logistic'][1][1]))
    true_H_Logistic.append(c_matrix['Logistic'][1][1]/float(c_matrix['Logistic'][1][0] + c_matrix['Logistic'][1][1]))
    
    RF_00.append(float(c_matrix['Random Forest'][0][0]))
    RF_01.append(float(c_matrix['Random Forest'][0][1]))
    RF_10.append(float(c_matrix['Random Forest'][1][0]))
    RF_11.append(float(c_matrix['Random Forest'][1][1]))
    NB_00.append(float(c_matrix['Naive Bayes'][0][0]))
    NB_01.append(float(c_matrix['Naive Bayes'][0][1]))
    NB_10.append(float(c_matrix['Naive Bayes'][1][0]))
    NB_11.append(float(c_matrix['Naive Bayes'][1][1]))
    NN_00.append(float(c_matrix['NeuralNet'][0][0]))
    NN_01.append(float(c_matrix['NeuralNet'][0][1]))
    NN_10.append(float(c_matrix['NeuralNet'][1][0]))
    NN_11.append(float(c_matrix['NeuralNet'][1][1]))
    LR_00.append(float(c_matrix['Logistic'][0][0]))
    LR_01.append(float(c_matrix['Logistic'][0][1]))
    LR_10.append(float(c_matrix['Logistic'][1][0]))
    LR_11.append(float(c_matrix['Logistic'][1][1]))    
    
    print('Total accuracy: ', accuracy)
    #print(c_matrix)
    print('Training set has '+str(np.sum(y_train_np == 1))+' H-modes and '+str(np.sum(y_train_np == 0))+' L-modes')
    print('Test set has '+str(np.sum(y_test_np == 1))+' H-modes and '+str(np.sum(y_test_np == 0))+' L-modes')
    print('Validation set has '+str(np.sum(y_valid_np == 1))+' H-modes and '+str(np.sum(y_valid_np == 0))+' L-modes')
    #print(c_matrix1)
    
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #        print("Normalized confusion matrix")
    #    else:
    #        print('Confusion matrix, without normalization')
    #
    #    print(cm)
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label') 
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(c_matrix['Logistic'], classes=class_names,
                          title='Confusion matrix, without normalization, Logistic') 
    plt.figure()
    plot_confusion_matrix(c_matrix['Logistic'], classes=class_names, normalize=True,
                          title='Normalized confusion matrix, Logistic')
    plt.figure()
    plot_confusion_matrix(c_matrix['Naive Bayes'], classes=class_names,
                          title='Confusion matrix, without normalization, NB') 
    plt.figure()
    plot_confusion_matrix(c_matrix['Naive Bayes'], classes=class_names, normalize=True,
                          title='Normalized confusion matrix, NB') 
    plt.figure()
    plot_confusion_matrix(c_matrix['Random Forest'], classes=class_names,
                          title='Confusion matrix, without normalization, RF') 
    plt.figure()
    plot_confusion_matrix(c_matrix['Random Forest'], classes=class_names, normalize=True,
                          title='Normalized confusion matrix, RF')
    plt.figure()
    plot_confusion_matrix(c_matrix['NeuralNet'], classes=class_names,
                          title='Confusion matrix, without normalization, NN') 
    plt.figure()
    plot_confusion_matrix(c_matrix['NeuralNet'], classes=class_names, normalize=True,
                          title='Normalized confusion matrix, NN')
    plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(c_matrix_valid['Logistic'], classes=class_names,
                          title='Confusion matrix (validation), without normalization, Logistic') 
    plt.figure()
    plot_confusion_matrix(c_matrix_valid['Logistic'], classes=class_names, normalize=True,
                          title='Normalized confusion matrix (validation), Logistic')
    plt.figure()
    plot_confusion_matrix(c_matrix_valid['Naive Bayes'], classes=class_names,
                          title='Confusion matrix (validation), without normalization, NB') 
    plt.figure()
    plot_confusion_matrix(c_matrix_valid['Naive Bayes'], classes=class_names, normalize=True,
                          title='Normalized confusion matrix (validation), NB') 
    plt.figure()
    plot_confusion_matrix(c_matrix_valid['Random Forest'], classes=class_names,
                          title='Confusion matrix (validation), without normalization, RF') 
    plt.figure()
    plot_confusion_matrix(c_matrix_valid['Random Forest'], classes=class_names, normalize=True,
                          title='Normalized confusion matrix (validation), RF')
    plt.figure()
    plot_confusion_matrix(c_matrix_valid['NeuralNet'], classes=class_names,
                          title='Confusion matrix (validation), without normalization, NN') 
    plt.figure()
    plot_confusion_matrix(c_matrix_valid['NeuralNet'], classes=class_names, normalize=True,
                          title='Normalized confusion matrix (validation), NN')
    plt.show()
    
    X_test = np.array(X_test)
    X_train_valid = np.array(X_train_valid)
#    n, bins, patches = plt.hist(X_train_valid[:,0], 20, label = 'Ip, train', facecolor='g', alpha=0.75)
#    n, bins, patches = plt.hist(X_test[:,0], 20, label = 'Ip, test', facecolor='r', alpha=0.75)
#    plt.legend()
#    plt.show()
#    n, bins, patches = plt.hist(X_train_valid[:,1], 20,  label = 'Btor, train', facecolor='g', alpha=0.75)
#    n, bins, patches = plt.hist(X_test[:,1], 20, label = 'Btor, test', facecolor='r', alpha=0.75)
#    plt.legend()
#    plt.show()
#    n, bins, patches = plt.hist(X_train_valid[:,2], 20, label = 'li, train', facecolor='g', alpha=0.75)
#    n, bins, patches = plt.hist(X_test[:,2], 20, label = 'li, test', facecolor='r', alpha=0.75)
#    plt.legend()
#    plt.show()
#    n, bins, patches = plt.hist(X_train_valid[:,3], 20, label = 'q95, train', facecolor='g', alpha=0.75)
#    n, bins, patches = plt.hist(X_test[:,3], 20, label = 'q95, test', facecolor='r', alpha=0.75)
#    plt.legend()
#    plt.show()
#    n, bins, patches = plt.hist(X_train_valid[:,4], 20, label = 'Wmhd, train', facecolor='g', alpha=0.75)
#    n, bins, patches = plt.hist(X_test[:,4], 20, label = 'Wmhd, test', facecolor='r', alpha=0.75)
#    plt.legend()
#    plt.show()
#    n, bins, patches = plt.hist(X_train_valid[:,5], 20, label = 'p_icrf, train', facecolor='g', alpha=0.75)
#    n, bins, patches = plt.hist(X_test[:,5], 20, label = 'p_icrf, test', facecolor='r', alpha=0.75)
#    plt.legend()
#    plt.show()
#    n, bins, patches = plt.hist(X_train_valid[:,6], 20, label = 'beta_p, train', facecolor='g', alpha=0.75)
#    n, bins, patches = plt.hist(X_test[:,6], 20, label = 'beta_p, test', facecolor='r', alpha=0.75)
#    plt.legend()
#    plt.show() 
#    n, bins, patches = plt.hist(X_train_valid[:,7], 20, label = 'P_ohm, train', facecolor='g', alpha=0.75)
#    n, bins, patches = plt.hist(X_test[:,7], 20, label = 'P_ohm, test', facecolor='r', alpha=0.75)
#    plt.legend()
#    plt.show() 
    
#    X_std = StandardScaler().fit_transform(total_x_data)
#    mean_vec = np.mean(X_std, axis=0)
#    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
#    #print('Covariance matrix \n%s' %cov_mat)
#    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#    #print('Eigenvectors \n%s' %eig_vecs)
#    #print('\nEigenvalues \n%s' %eig_vals)
#    # Make a list of (eigenvalue, eigenvector) tuples
#    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
#    
#    # Sort the (eigenvalue, eigenvector) tuples from high to low
#    eig_pairs.sort()
#    eig_pairs.reverse()
#    
#    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
#    #print('Eigenvalues in descending order:')
#    #for i in eig_pairs:
#    #    print(i[0])
#        
#    tot = sum(eig_vals)
#    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
#    cum_var_exp = np.cumsum(var_exp) 
    
#    plt.plot(range(0,len(total_x_data[0])),var_exp)
#    plt.plot(range(0,len(total_x_data[0])),cum_var_exp,label='cumulative explained variance')
#    plt.ylabel('Explained variance in percent')
#    plt.title('Explained variance by different principal components') 
#    plt.legend()
#    plt.show()
    
    
    #list_vector = []
#    names = ['ip','btor','li','q95','Wmhd','p_icrf','beta_N','nebar_efit','beta_p','beta_t','kappa','triang','areao','vout','aout','rout','zout','zmag','rmag','zsep_lower','zsep_upper','rsep_lower','rsep_upper','zvsin','rvsin','zvsout','rvsout','upper_gap','lower_gap','qstar','V_loop_efit','V_surf_efit','cpasma','ssep','P_ohm','NL_04','g_side_rat','e_bot_mks','b_bot_mks','Halpha','Dalpha']
    #k = 0 
    #while k < len(eig_vecs):
    #    j = 0
    #    for i in eig_vecs[k]:
    #        list_vector.append([k,i,j,names[j]]) #j and names[j] is essentially redundant here
    #        j = j + 1
    #    k = k + 1
    #    
    #k = 0
    #while k < len(eig_vecs):
    #    temp_sort = list_vector[(40*k):(39 + 40*k)]
    #    temp_sort = sorted(temp_sort, key=lambda tup: abs(tup[1]),reverse=True)
    #    print(temp_sort[0:20])
    #    k = k + 1
    #    
    importances = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(len(X_test[0])):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(X_test[0])), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(X_test[0])), indices)
    plt.xlim([-1,len(X_test[0])])
    plt.show()
#    model = LogisticRegression()
#    rfe = RFE(model, 1)
#    fit = rfe.fit(X, y) 
#    j = 0
#    print('LogisticRegression')
#    for i in list(fit.ranking_):
#        print(names[j], 'rank:', i)
#        j = j + 1
        
    #model = LinearSVC(C=1.0)
    #rfe = RFE(model, 8)
    #fit = rfe.fit(X, y) 
    #j = 0
    #print('LinearSVC')
    #for i in list(fit.ranking_):
    #    print(names[j], i)
    #    j = j + 1
    #    
    #model = RandomForestClassifier(n_estimators=100)
    #rfe = RFE(model, 8)
    #fit = rfe.fit(X, y) 
    #j = 0
    #print('RandomForestClassifier')
    #for i in list(fit.ranking_):
    #    print(names[j], i)
    #    j = j + 1
        
        
    #simply plots confusion matrix for different ntrees; but not significant...
    #ntree = 198
    #while ntree < 200:
    #    rfc = RandomForestClassifier(n_estimators=ntree)
    #    rfc.fit(X_train_valid, y_train_valid)
    #    c_matrix['Random Forest'] = confusion_matrix(y_test_np, prediction['Random Forest'])
    #    plt.figure()
    #    plot_confusion_matrix(c_matrix['Random Forest'], classes=class_names,
    #                      title='Confusion matrix, without normalization, RF') 
    #    plt.figure()
    #    plot_confusion_matrix(c_matrix['Random Forest'], classes=class_names, normalize=True,
    #                      title='Normalized confusion matrix, RF')
    #    plt.show()
    #    ntree = ntree + 1
    update_index = update_index + 1
    
    
from collections import OrderedDict 
RANDOM_STATE = 123

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("m = sqrt",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("m = log2 ",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("m = None ",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
] # m is max_features for trees in RandomForestClassifier

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
data1 = np.insert(X_data0, len(X_data0[0]), values=Y_data0, axis=-1)
together1 = [list(i) for _, i in itertools.groupby(data1, operator.itemgetter(0))]
random.shuffle(together1)
final_random1 = [i for j in together1 for i in j]
X_data1 = (np.array(final_random1))[:,1:-1]
Y_data1 = (np.array(final_random1))[:,-1]
#this train_valid is data that is training, then to be validated
X_train1, y_train1 = X_data1[:int(train_valid_frac*len(X_data1))], Y_data1[:int(train_valid_frac*len(X_data1))]

X_train1, y_train1 = shuffle(X_train1, y_train1, random_state=0) #or use this single line; random_state allows predictable output

X_train_valid1, y_train_valid1 = X_train1[:int(fraction_*len(X_train1))], y_train1[:int(fraction_*len(X_train1))]
X_valid1, y_valid1 = X_train1[int(fraction_*len(X_train1)):], y_train1[int(fraction_*len(X_train1)):]
X_test1, y_test1 = X_data1[int(train_valid_frac*len(X_data1)):], Y_data1[int(train_valid_frac*len(X_data1)):]
y_valid_np1 = np.array([int(numeric_string) for numeric_string in y_valid1])
y_test_np1 = np.array([int(numeric_string) for numeric_string in y_test1])
y_train_np1 = np.array([int(numeric_string) for numeric_string in y_train_valid1])

min_estimators = 30
max_estimators = 150
prediction0 = {}
c_matrix0 = {}
c_matrix10 = {} 
L_mode_acc = []
H_mode_acc = []
for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1)[::1]:
        clf.set_params(n_estimators=i)
        clf.fit(np.array(X_train1), np.array(y_train1))
        prediction0[str(label)+','+str(i)] = np.array([int(numeric_string) for numeric_string in clf.predict(X_test1)])
        c_matrix0[str(label)+','+str(i)] = confusion_matrix(y_test_np1, prediction0[str(label)+','+str(i)])
        c_matrix10[str(label)+','+str(i)+' predicted H-mode correctly'] = c_matrix0[str(label)+','+str(i)][1][1]/float(c_matrix0[str(label)+','+str(i)][1][1] + (c_matrix0[str(label)+','+str(i)][0][1]))
        c_matrix10[str(label)+','+str(i)+' predicted L-mode correctly'] = c_matrix0[str(label)+','+str(i)][0][0]/float(c_matrix0[str(label)+','+str(i)][0][0] + (c_matrix0[str(label)+','+str(i)][1][0]))
        L_mode_acc.append([label,i,(c_matrix10[str(label)+','+str(i)+' predicted L-mode correctly'])])
        H_mode_acc.append([label,i,(c_matrix10[str(label)+','+str(i)+' predicted H-mode correctly'])])
# Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate (Random Forest Classifier)")
plt.legend(loc="upper right")
plt.show()

L_mode_acc_sqrt = L_mode_acc[0:121]
L_mode_acc_log2 = L_mode_acc[121:242]
L_mode_acc_None = L_mode_acc[242:363]
H_mode_acc_sqrt = H_mode_acc[0:121]
H_mode_acc_log2 = H_mode_acc[121:242]
H_mode_acc_None = H_mode_acc[242:363]

def column(matrix, i):
    return [row[i] for row in matrix]

plt.figure()
plt.plot(xs,column(L_mode_acc_sqrt,2),label='m = sqrt')
plt.plot(xs,column(L_mode_acc_log2,2),label='m = log2')
plt.plot(xs,column(L_mode_acc_None,2),label='m = None')
plt.title('L-mode accuracy vs trees')
plt.ylabel('Accuracy (correct prediction/(correct + incorrect prediction)')
plt.xlabel('Trees')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

plt.figure()
plt.plot(xs,column(H_mode_acc_sqrt,2),label='m = sqrt')
plt.plot(xs,column(H_mode_acc_log2,2),label='m = log2')
plt.plot(xs,column(H_mode_acc_None,2),label='m = None')
plt.title('H-mode accuracy vs trees')
plt.ylabel('Accuracy (correct prediction/(correct + incorrect prediction)')
plt.xlabel('Trees')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


##univariate selection via chi-square statistical test for non-negative features
#X = np.array(total_x_data)
#Y = np.array(Y_data0)
#test = SelectKBest(score_func=chi2, k=8)
#fit = test.fit(np.abs(X), Y)
#np.set_printoptions(precision=3) 
#features = fit.transform(X)
#best_features = []
#i = 0
#while i < len(names): 
#    best_features.append([fit.scores_[i],names[i]])
#    i = i + 1
#best_features = sorted(best_features, key=lambda x: x[0], reverse=True)
#print(best_features)
##recursive feature elimination on logistic regression
#model = LogisticRegression()
#rfe = RFE(model, 8)
#fit = rfe.fit(X, Y)
#best_features = []
#i = 0
#while i < len(names): 
#    best_features.append([fit.ranking_[i],names[i]])
#    i = i + 1
#best_features = sorted(best_features, key=lambda x: x[0], reverse=True)
#print(best_features) 
##feature importance via bagged decision trees
#model = ExtraTreesClassifier()
#model.fit(X, Y)
#best_features = []
#i = 0
#while i < len(names): 
#    best_features.append([model.feature_importances_[i],names[i]])
#    i = i + 1
#best_features = sorted(best_features, key=lambda x: x[0], reverse=True)
#print(best_features) 
##recursive feature elimination on random forest using scikit-learn
#class RandomForestClassifierWithCoef(RandomForestClassifier):
#    def fit(self, *args, **kwargs):
#        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
#        self.coef_ = self.feature_importances_
#rf = RandomForestClassifierWithCoef(n_estimators=100)
#rfecv = RFECV(estimator=rf, step=1, cv=2, scoring='roc_auc', verbose=2)
#Y0_ = [int(numeric_string) for numeric_string in Y]
#selector=rfecv.fit(X,Y0_)
#takes a little/long while to run

##Selecting features with genetic algorithm (logistic regression)
#from genetic_selection import GeneticSelectionCV
#estimator = LogisticRegression()
#selector = GeneticSelectionCV(estimator,cv=5,verbose=1,scoring="accuracy",n_population=50,
#    crossover_proba=0.5,mutation_proba=0.2,n_generations=40,crossover_independent_proba=0.5,
#    mutation_independent_proba=0.05,tournament_size=3,caching=True,n_jobs=-1)
#selector = selector.fit(X, Y)
#print(selector.support_)
#best_features = []
#i = 0
#while i < len(names): 
#    best_features.append([selector.support_[i],names[i]])
#    i = i + 1
#best_features = sorted(best_features, key=lambda x: x[0], reverse=True)
#print(best_features)

n, bins, patches = plt.hist(true_H_RF, 20, label = 'TP/(TP+FN)', alpha=0.75) 
n, bins, patches = plt.hist(false_H_RF, 20, label = 'FP/(TN+FP)', alpha=0.75) 
n, bins, patches = plt.hist(true_L_RF, 20, label = 'TN/(TN+FP)', alpha=0.75) 
n, bins, patches = plt.hist(false_L_RF, 20, label = 'FN/(TN+FP)', alpha=0.75) 
plt.title('Random Forest')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
n, bins, patches = plt.hist(true_H_NB, 20, label = 'TP/(TP+FN)', alpha=0.75) 
n, bins, patches = plt.hist(false_H_NB, 20, label = 'FP/(TN+FP)', alpha=0.75) 
n, bins, patches = plt.hist(true_L_NB, 20, label = 'TN/(TN+FP)', alpha=0.75) 
n, bins, patches = plt.hist(false_L_NB, 20, label = 'FN/(TN+FP)', alpha=0.75) 
plt.title('Gaussian Naive Bayes')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
n, bins, patches = plt.hist(true_H_NN, 20, label = 'TP/(TP+FN)', alpha=0.75) 
n, bins, patches = plt.hist(false_H_NN, 20, label = 'FP/(TN+FP)', alpha=0.75) 
n, bins, patches = plt.hist(true_L_NN, 20, label = 'TN/(TN+FP)', alpha=0.75) 
n, bins, patches = plt.hist(false_L_NN, 20, label = 'FN/(TN+FP)', alpha=0.75) 
plt.title('NeuralNet')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
n, bins, patches = plt.hist(true_H_Logistic, 20, label = 'TP/(TP+FN)', alpha=0.75) 
n, bins, patches = plt.hist(false_H_Logistic, 20, label = 'FP/(TN+FP)', alpha=0.75) 
n, bins, patches = plt.hist(true_L_Logistic, 20, label = 'TN/(TN+FP)', alpha=0.75) 
n, bins, patches = plt.hist(false_L_Logistic, 20, label = 'FN/(TN+FP)', alpha=0.75) 
plt.title('Logistic Regression')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show() 

print('Logistic') 
print('trueH - ',np.mean(true_H_Logistic),' +/- ',np.std(true_H_Logistic))
print('FalseH - ',np.mean(false_H_Logistic),' +/- ',np.std(false_H_Logistic))
print('trueL - ',np.mean(true_L_Logistic),' +/- ',np.std(true_L_Logistic))
print('falseL - ',np.mean(false_L_Logistic),' +/- ',np.std(false_L_Logistic))
print('Gaussian Naive Bayes')
print('trueH - ',np.mean(true_H_NB),' +/- ',np.std(true_H_NB))
print('FalseH - ',np.mean(false_H_NB),' +/- ',np.std(false_H_NB))
print('trueL - ',np.mean(true_L_NB),' +/- ',np.std(true_L_NB))
print('falseL - ',np.mean(false_L_NB),' +/- ',np.std(false_L_NB))
print('NeuralNet')
print('trueH - ',np.mean(true_H_NN),' +/- ',np.std(true_H_NN))
print('FalseH - ',np.mean(false_H_NN),' +/- ',np.std(false_H_NN))
print('trueL - ',np.mean(true_L_NN),' +/- ',np.std(true_L_NN))
print('falseL - ',np.mean(false_L_NN),' +/- ',np.std(false_L_NN))
print('Random Forest')
print('trueH - ',np.mean(true_H_RF),' +/- ',np.std(true_H_RF))
print('FalseH - ',np.mean(false_H_RF),' +/- ',np.std(false_H_RF))
print('trueL - ',np.mean(true_L_RF),' +/- ',np.std(true_L_RF))
print('falseL - ',np.mean(false_L_RF),' +/- ',np.std(false_L_RF))

RF_00 = np.array(RF_00)
RF_01 = np.array(RF_01) 
RF_10 = np.array(RF_10)
RF_11 = np.array(RF_11) 
NB_00 = np.array(NB_00)
NB_01 = np.array(NB_01) 
NB_10 = np.array(NB_10)
NB_11 = np.array(NB_11) 
NN_00 = np.array(NN_00)
NN_01 = np.array(NN_01) 
NN_10 = np.array(NN_10)
NN_11 = np.array(NN_11) 
LR_00 = np.array(LR_00)
LR_01 = np.array(LR_01) 
LR_10 = np.array(LR_10)
LR_11 = np.array(LR_11) 

precision_RF = RF_11/(RF_11 + RF_01) #actually H-mode out of total predicted H-mode
recall_RF = RF_11/(RF_11 + RF_10) #sensitivity
F1_score_RF = 2.*precision_RF*recall_RF/(precision_RF + recall_RF)
precision_NB = NB_11/(NB_11 + NB_01) #actually H-mode out of total predicted H-mode
recall_NB = NB_11/(NB_11 + NB_10) #sensitivity
F1_score_NB = 2.*precision_NB*recall_NB/(precision_NB + recall_NB)
precision_NN = NN_11/(NN_11 + NN_01) #actually H-mode out of total predicted H-mode
recall_NN = NN_11/(NN_11 + NN_10) #sensitivity
F1_score_NN = 2.*precision_NN*recall_NN/(precision_NN + recall_NN)
precision_LR = LR_11/(LR_11 + LR_01) #actually H-mode out of total predicted H-mode
recall_LR = LR_11/(LR_11 + LR_10) #sensitivity
F1_score_LR = 2.*precision_LR*recall_LR/(precision_LR + recall_LR)

print('precision_RF:', np.mean(precision_RF),' +/- ',np.std(precision_RF))
print('recall_RF:', np.mean(recall_RF),' +/- ',np.std(recall_RF))
print('F1_score_RF:', np.mean(F1_score_RF),' +/- ',np.std(F1_score_RF)) 
print('precision_NB:', np.mean(precision_NB),' +/- ',np.std(precision_NB))
print('recall_NB:', np.mean(recall_NB),' +/- ',np.std(recall_NB))
print('F1_score_NB:', np.mean(F1_score_NB),' +/- ',np.std(F1_score_NB)) 
print('precision_NN:', np.mean(precision_NN),' +/- ',np.std(precision_NN))
print('recall_NN:', np.mean(recall_NN),' +/- ',np.std(recall_NN))
print('F1_score_NN:', np.mean(F1_score_NN),' +/- ',np.std(F1_score_NN))
print('precision_LR:', np.mean(precision_LR),' +/- ',np.std(precision_LR))
print('recall_LR:', np.mean(recall_LR),' +/- ',np.std(recall_LR))
print('F1_score_LR:', np.mean(F1_score_LR),' +/- ',np.std(F1_score_LR))

import pickle
#Saving created RF model
RF_LH_pkl_filename = '/home/mathewsa/Desktop/RF_classifier_LH.pkl'
RF_LH_model_pkl = open(RF_LH_pkl_filename, 'wb')
pickle.dump(rfc, RF_LH_model_pkl)
RF_LH_model_pkl.close()

#Loading saved model
RF_LH_model_pkl = open(RF_LH_pkl_filename, 'rb')
RF_LH_model = pickle.load(RF_LH_model_pkl)
print("Loaded model :: ", RF_LH_model)

#Saving created NN model 
NN_LH_pkl_filename = '/home/mathewsa/Desktop/NN_classifier_LH.pkl'
NN_LH_model_pkl = open(NN_LH_pkl_filename, 'wb')
pickle.dump(mlp, NN_LH_model_pkl)
NN_LH_model_pkl.close()

#Loading saved model
NN_LH_model_pkl = open(NN_LH_pkl_filename, 'rb')
NN_LH_model = pickle.load(NN_LH_model_pkl)
print("Loaded model :: ", NN_LH_model)

#Saving created GNB model 
GNB_LH_pkl_filename = '/home/mathewsa/Desktop/GNB_classifier_LH.pkl'
GNB_LH_model_pkl = open(GNB_LH_pkl_filename, 'wb')
pickle.dump(gnb, GNB_LH_model_pkl)
GNB_LH_model_pkl.close()

#Loading saved model
GNB_LH_model_pkl = open(GNB_LH_pkl_filename, 'rb')
GNB_LH_model = pickle.load(GNB_LH_model_pkl)
print("Loaded model :: ", GNB_LH_model)

#Saving created LR model 
LR_LH_pkl_filename = '/home/mathewsa/Desktop/LR_classifier_LH.pkl'
LR_LH_model_pkl = open(LR_LH_pkl_filename, 'wb')
pickle.dump(lr, LR_LH_model_pkl)
LR_LH_model_pkl.close()

#Loading saved model
LR_LH_model_pkl = open(LR_LH_pkl_filename, 'rb')
LR_LH_model = pickle.load(LR_LH_model_pkl)
print("Loaded model :: ", LR_LH_model)

scalerfile = '/home/mathewsa/Desktop/scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))

scalerfile = '/home/mathewsa/Desktop/scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb')) 

df = pd.DataFrame(X_data0)
corr = df.corr()
corr.style.background_gradient().set_precision(3)
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.3f', mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
            
print(LR_LH_model.coef_)