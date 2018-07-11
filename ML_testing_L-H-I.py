# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:33:32 2018

@author: mathewsa 

This code is only used for multi-class classification of  confinement
modes (L, H, and I) using supervised machine learning methods from scikit learn
""" 
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

c_matrix_RF = []
c_matrix_valid_RF = [] 
c_matrix_LR = []
c_matrix_valid_LR = []
c_matrix_NB = []
c_matrix_valid_NB = []
c_matrix_NN = []
c_matrix_valid_NN = []

RF_00 = []
RF_01 = []
RF_02 = []
RF_10  = []
RF_11  = []
RF_12 = []
RF_20  = []
RF_21  = []
RF_22 = []
NB_00 = []
NB_01 = []
NB_02 = []
NB_10  = []
NB_11  = []
NB_12 = []
NB_20  = []
NB_21  = []
NB_22 = []
NN_00 = []
NN_01 = []
NN_02 = []
NN_10  = []
NN_11  = []
NN_12 = []
NN_20  = []
NN_21  = []
NN_22 = []
LR_00 = []
LR_01 = []
LR_02 = []
LR_10  = []
LR_11  = []
LR_12 = []
LR_20  = []
LR_21  = []
LR_22 = []

tprs_H = []
aucs_H = []
mean_fpr_H = np.linspace(0, 1, 100)
tprs_I = []
aucs_I = []
mean_fpr_I = np.linspace(0, 1, 100)

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
    data = np.insert(X_data0, len(X_data0[0]), values=Y_data0, axis=-1)
    together = [list(i) for _, i in itertools.groupby(data, operator.itemgetter(0))]
    random.shuffle(together) #groups based on first item of x_data, which should be shot!
    final_random = [i for j in together for i in j]
    X_data = (np.array(final_random))[:,1:-1]
    Y_data = (np.array(final_random, dtype = int))[:,-1]
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
    
#    print('--------------------SVM--------------------')
#    # training a linear SVM classifier
#    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
#    svm_predictions = svm_model_linear.predict(X_test)
#     
#    # model accuracy for X_test  
#    accuracy = svm_model_linear.score(X_test, y_test)
#     
#    # creating a confusion matrix
#    cm = confusion_matrix(y_test, svm_predictions)
#    print(cm)
#    print('--------------------SVM--------------------')
#    
#    print('--------------------KNN--------------------')
#    knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
#     
#    # accuracy on X_test
#    accuracy = knn.score(X_test, y_test)
#    print accuracy
#     
#    # creating a confusion matrix
#    knn_predictions = knn.predict(X_test) 
#    cm = confusion_matrix(y_test, knn_predictions)
#    print(cm)
#    print('--------------------KNN--------------------')
#    
#    print('--------------------GNB--------------------')
#    gnb = GaussianNB().fit(X_train, y_train)
#    gnb_predictions = gnb.predict(X_test)
#     
#    # accuracy on X_test
#    accuracy = gnb.score(X_test, y_test)
#    print accuracy
#     
#    # creating a confusion matrix
#    cm = confusion_matrix(y_test, gnb_predictions)
#    print(cm)
#    print('--------------------GNB--------------------')
    
#    print('--------------------RF--------------------')
    rfc = RandomForestClassifier(n_estimators=100,max_features="sqrt")
    lr = LogisticRegression()
    gnb = GaussianNB()  
    mlp = MLPClassifier(hidden_layer_sizes=(100,100,100))
    
    prediction_prob_H = {}
    prediction_prob_I = {}
    prediction = {}
    prediction_prob_valid_H = {}
    prediction_prob_valid_I = {}
    prediction_valid = {}
    sum_array = {}
    accuracy = {}
    c_matrix = {} #confusion matrix 
    sum_array_valid = {}
    accuracy_valid = {}
    c_matrix_valid = {} #confusion matrix 
    
    # Plot calibration plots
    plt.figure(figsize=(10, 28))
    
    ax1 = plt.subplot2grid((6, 1), (0, 0))
    ax2 = plt.subplot2grid((6, 1), (1, 0))
    ax3 = plt.subplot2grid((6, 1), (2, 0))
    ax4 = plt.subplot2grid((6, 1), (3, 0))
    ax5 = plt.subplot2grid((6, 1), (4, 0))
    ax6 = plt.subplot2grid((6, 1), (5, 0))
    
    ax7 = plt.subplot2grid((6, 1), (0, 0))
    ax8 = plt.subplot2grid((6, 1), (1, 0))
    ax9 = plt.subplot2grid((6, 1), (2, 0))
    ax10 = plt.subplot2grid((6, 1), (3, 0))
    ax11 = plt.subplot2grid((6, 1), (4, 0))
    ax12 = plt.subplot2grid((6, 1), (5, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax3.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    ax7.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax9.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
#                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest'),
                  (mlp, 'NeuralNet')]: 
        clf.fit(X_train_valid, y_train_valid)
    #   if hasattr(clf, "predict_proba"):
        prob_pos_H = clf.predict_proba(X_test)[:, 1] #probability of 1, or H-mode
        prob_pos_valid_H = clf.predict_proba(X_valid)[:, 1]
        prob_pos_I = clf.predict_proba(X_test)[:, 2] #probability of 1, or H-mode
        prob_pos_valid_I = clf.predict_proba(X_valid)[:, 2]
        prediction_prob_H[str(name)] = np.array([int(numeric_string) for numeric_string in prob_pos_H])
        prediction_prob_valid_H[str(name)] = np.array([int(numeric_string) for numeric_string in prob_pos_valid_H])
        prediction_prob_I[str(name)] = np.array([int(numeric_string) for numeric_string in prob_pos_I])
        prediction_prob_valid_I[str(name)] = np.array([int(numeric_string) for numeric_string in prob_pos_valid_I])        
        prediction[str(name)] = np.array([int(numeric_string) for numeric_string in clf.predict(X_test)])
        prediction_valid[str(name)] = np.array([int(numeric_string) for numeric_string in clf.predict(X_valid)])
    
        c_matrix[str(name)] = array3x3(confusion_matrix(y_test_np, prediction[str(name)]))
        c_matrix_valid[str(name)] = array3x3(confusion_matrix(y_valid_np, prediction_valid[str(name)]))
        
        y_test_H = np.where(y_test == 0, 0, y_test)
        y_test_H = np.where(y_test_H == 2, 0, y_test_H)
        y_test_I = np.where(y_test == 0, 0, y_test)
        y_test_I = np.where(y_test_I == 1, 0, y_test_I)
        y_test_I = np.where(y_test_I == 2, 1, y_test_I)
        
        y_valid_H = np.where(y_valid == 0, 0, y_valid)
        y_valid_H = np.where(y_valid_H == 2, 0, y_valid_H)
        y_valid_I = np.where(y_valid == 0, 0, y_valid)
        y_valid_I = np.where(y_valid_I == 1, 0, y_valid_I)
        y_valid_I = np.where(y_valid_I == 2, 1, y_valid_I)

        fraction_of_positives_H, mean_predicted_value_H = calibration_curve(y_test_H, prob_pos_H, n_bins=20)
        fraction_of_positives_valid_H, mean_predicted_value_valid_H = calibration_curve(y_valid_H, prob_pos_valid_H, n_bins=20)
        fraction_of_positives_I, mean_predicted_value_I = calibration_curve(y_test_I, prob_pos_I, n_bins=20)
        fraction_of_positives_valid_I, mean_predicted_value_valid_I = calibration_curve(y_valid_I, prob_pos_valid_I, n_bins=20)    
    
        ax1.plot(mean_predicted_value_H, fraction_of_positives_H, "s-",
                     label="%s test" % (name, ))
        ax2.hist(prob_pos_H, range=(0, 1), bins=20, label=name,
                     histtype="step", lw=2) 
        
        ax3.plot(mean_predicted_value_valid_H, fraction_of_positives_valid_H, "s-",
                     label="%s valid" % (name, ))
        ax4.hist(prob_pos_valid_H, range=(0, 1), bins=20, label=name,
                     histtype="step", lw=2)
    
        fpr_H, tpr_H, thresholds = roc_curve(y_test_H,prob_pos_H)     
        tprs_H.append(np.interp(mean_fpr_H, fpr_H, tpr_H))
        tprs_H[-1][0] = 0.0
        roc_auc_H = auc(fpr_H, tpr_H)
        aucs_H.append(roc_auc_H)
        ax5.plot(fpr_H, tpr_H, lw=1, alpha=0.4,
             label='ROC fold %s (AUC = %0.2f)' % (name, roc_auc_H))
        ax6.plot(fpr_H, tpr_H, lw=1, alpha=0.4,
         label='ROC fold %s (AUC = %0.2f)' % (name, roc_auc_H))     
         
        ax7.plot(mean_predicted_value_I, fraction_of_positives_I, "s-",
                     label="%s test" % (name, ))
        ax8.hist(prob_pos_I, range=(0, 1), bins=20, label=name,
                     histtype="step", lw=2) 
        
        ax9.plot(mean_predicted_value_valid_I, fraction_of_positives_valid_I, "s-",
                     label="%s valid" % (name, ))
        ax10.hist(prob_pos_valid_I, range=(0, 1), bins=20, label=name,
                     histtype="step", lw=2)
    
        fpr_I, tpr_I, thresholds = roc_curve(y_test_I,prob_pos_I)     
        tprs_I.append(np.interp(mean_fpr_I, fpr_I, tpr_I))
        tprs_I[-1][0] = 0.0
        roc_auc_I = auc(fpr_I, tpr_I)
        aucs_I.append(roc_auc_I)
        ax11.plot(fpr_I, tpr_I, lw=1, alpha=0.4,
             label='ROC fold %s (AUC = %0.2f)' % (name, roc_auc_I))
        ax12.plot(fpr_I, tpr_I, lw=1, alpha=0.4,
         label='ROC fold %s (AUC = %0.2f)' % (name, roc_auc_I))          
         
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('H-mode Calibration plots - testing (reliability curve)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2) 
            
    ax3.set_ylabel("Fraction of positives")
    ax3.set_ylim([-0.05, 1.05])
    ax3.legend(loc="lower right")
    ax3.set_title('H-mode Calibration plots - validation (reliability curve)')
    ax4.set_xlabel("Mean predicted value")
    ax4.set_ylabel("Count")
    ax4.legend(loc="upper center", ncol=2)   
    
    ax5.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.4)
    ax6.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.4)
         
    mean_tpr_H = np.mean(tprs_H, axis=0)
    mean_tpr_H[-1] = 1.0
    mean_auc_H = auc(mean_fpr_H, mean_tpr_H)
    std_auc_H = np.std(aucs_H)
    ax5.plot(mean_fpr_H, mean_tpr_H, color='black',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_H, std_auc_H),
             lw=2, alpha=.4)
    ax6.plot(mean_fpr_H, mean_tpr_H, color='black',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_H, std_auc_H),
         lw=2, alpha=.4)
    
    std_tpr_H = np.std(tprs_H, axis=0)
    tprs_upper_H = np.minimum(mean_tpr_H + std_tpr_H, 1)
    tprs_lower_H = np.maximum(mean_tpr_H - std_tpr_H, 0)
    ax5.fill_between(mean_fpr_H, tprs_lower_H, tprs_upper_H, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    ax6.fill_between(mean_fpr_H, tprs_lower_H, tprs_upper_H, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    ax5.set_xlim([0.0, 1.05])
    ax5.set_ylim([0.0, 1.05])
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.set_title('H-mode Receiver Operating Characteristic')
    ax5.legend(loc="lower right")
    
    ax6.set_xlim([0.0, 0.10])
    ax6.set_ylim([0.0, 1.05])
    ax6.set_xlabel('False Positive Rate')
    ax6.set_ylabel('True Positive Rate')
    ax6.set_title('H-mode Receiver Operating Characteristic')
    ax6.legend(loc="lower right")
    
    
    ax7.set_ylabel("Fraction of positives")
    ax7.set_ylim([-0.05, 1.05])
    ax7.legend(loc="lower right")
    ax7.set_title('I-mode Calibration plots - testing (reliability curve)')
    ax8.set_xlabel("Mean predicted value")
    ax8.set_ylabel("Count")
    ax8.legend(loc="upper center", ncol=2) 
           
    ax9.set_ylabel("Fraction of positives")
    ax9.set_ylim([-0.05, 1.05])
    ax9.legend(loc="lower right")
    ax9.set_title('I-mode Calibration plots - validation (reliability curve)')
    ax10.set_xlabel("Mean predicted value")
    ax10.set_ylabel("Count")
    ax10.legend(loc="upper center", ncol=2)   
    
    ax11.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.4)
    ax12.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.4)
         
    mean_tpr_I = np.mean(tprs_I, axis=0)
    mean_tpr_I[-1] = 1.0
    mean_auc_I = auc(mean_fpr_I, mean_tpr_I)
    std_auc_I = np.std(aucs_I)
    ax11.plot(mean_fpr_I, mean_tpr_I, color='black',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_I, std_auc_I),
             lw=2, alpha=.4)
    ax12.plot(mean_fpr_I, mean_tpr_I, color='black',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_I, std_auc_I),
         lw=2, alpha=.4)
    
    std_tpr_I = np.std(tprs_I, axis=0)
    tprs_upper_I = np.minimum(mean_tpr_I + std_tpr_I, 1)
    tprs_lower_I = np.maximum(mean_tpr_I - std_tpr_I, 0)
    ax11.fill_between(mean_fpr_I, tprs_lower_I, tprs_upper_I, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    ax12.fill_between(mean_fpr_I, tprs_lower_I, tprs_upper_I, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    ax11.set_xlim([0.0, 1.05])
    ax11.set_ylim([0.0, 1.05])
    ax11.set_xlabel('False Positive Rate')
    ax11.set_ylabel('True Positive Rate')
    ax11.set_title('I-mode Receiver Operating Characteristic')
    ax11.legend(loc="lower right")
    
    ax12.set_xlim([0.0, 0.10])
    ax12.set_ylim([0.0, 1.05])
    ax12.set_xlabel('False Positive Rate')
    ax12.set_ylabel('True Positive Rate')
    ax12.set_title('I-mode Receiver Operating Characteristic')
    ax12.legend(loc="lower right")
    
    
    plt.tight_layout() 
    plt.show() 
    
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

    c_matrix_RF.append(c_matrix['Random Forest'])
    c_matrix_valid_RF.append(c_matrix_valid['Random Forest'])
    c_matrix_LR.append(c_matrix['Logistic'])
    c_matrix_valid_LR.append(c_matrix_valid['Logistic'])
    c_matrix_NB.append(c_matrix['Naive Bayes'])
    c_matrix_valid_NB.append(c_matrix_valid['Naive Bayes'])
    c_matrix_NN.append(c_matrix['NeuralNet'])
    c_matrix_valid_NN.append(c_matrix_valid['NeuralNet'])
    
    RF_00.append(float(c_matrix['Random Forest'][0][0]))
    RF_01.append(float(c_matrix['Random Forest'][0][1]))
    RF_02.append(float(c_matrix['Random Forest'][0][2]))
    RF_10.append(float(c_matrix['Random Forest'][1][0]))
    RF_11.append(float(c_matrix['Random Forest'][1][1]))
    RF_12.append(float(c_matrix['Random Forest'][1][2]))
    RF_20.append(float(c_matrix['Random Forest'][2][0]))
    RF_21.append(float(c_matrix['Random Forest'][2][1]))
    RF_22.append(float(c_matrix['Random Forest'][2][2]))
    NB_00.append(float(c_matrix['Naive Bayes'][0][0]))
    NB_01.append(float(c_matrix['Naive Bayes'][0][1]))
    NB_02.append(float(c_matrix['Naive Bayes'][0][2]))
    NB_10.append(float(c_matrix['Naive Bayes'][1][0]))
    NB_11.append(float(c_matrix['Naive Bayes'][1][1]))
    NB_12.append(float(c_matrix['Naive Bayes'][1][2]))
    NB_20.append(float(c_matrix['Naive Bayes'][2][0]))
    NB_21.append(float(c_matrix['Naive Bayes'][2][1]))
    NB_22.append(float(c_matrix['Naive Bayes'][2][2])) 
    NN_00.append(float(c_matrix['NeuralNet'][0][0]))
    NN_01.append(float(c_matrix['NeuralNet'][0][1]))
    NN_02.append(float(c_matrix['NeuralNet'][0][2]))
    NN_10.append(float(c_matrix['NeuralNet'][1][0]))
    NN_11.append(float(c_matrix['NeuralNet'][1][1]))
    NN_12.append(float(c_matrix['NeuralNet'][1][2]))
    NN_20.append(float(c_matrix['NeuralNet'][2][0]))
    NN_21.append(float(c_matrix['NeuralNet'][2][1]))
    NN_22.append(float(c_matrix['NeuralNet'][2][2]))  
    LR_00.append(float(c_matrix['Logistic'][0][0]))
    LR_01.append(float(c_matrix['Logistic'][0][1]))
    LR_02.append(float(c_matrix['Logistic'][0][2]))
    LR_10.append(float(c_matrix['Logistic'][1][0]))
    LR_11.append(float(c_matrix['Logistic'][1][1]))
    LR_12.append(float(c_matrix['Logistic'][1][2]))
    LR_20.append(float(c_matrix['Logistic'][2][0]))
    LR_21.append(float(c_matrix['Logistic'][2][1]))
    LR_22.append(float(c_matrix['Logistic'][2][2])) 

#    clf = RandomForestClassifier(n_estimators=25)
#    clf.fit(X_train_valid, y_train_valid)
#    clf_probs = clf.predict_proba(X_test)
#    score = log_loss(y_test, clf_probs)
#    
#    # Train random forest classifier, calibrate on validation data and evaluate
#    # on test data
#    clf = RandomForestClassifier(n_estimators=25)
#    clf.fit(X_train, y_train)
#    clf_probs = clf.predict_proba(X_test)
#    sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
#    sig_clf.fit(X_valid, y_valid)
#    sig_clf_probs = sig_clf.predict_proba(X_test)
#    sig_score = log_loss(y_test, sig_clf_probs)
#    
#    # Plot changes in predicted probabilities via arrows
#    plt.figure(0)
#    colors = ["r", "g", "b"]
#    for i in range(clf_probs.shape[0]):
#        plt.arrow(clf_probs[i, 0], clf_probs[i, 1],
#                  sig_clf_probs[i, 0] - clf_probs[i, 0],
#                  sig_clf_probs[i, 1] - clf_probs[i, 1],
#                  color=colors[y_test[i]], head_width=1e-2)
#    
#    # Plot perfect predictions
#    plt.plot([1.0], [0.0], 'ro', ms=20, label="Class 1")
#    plt.plot([0.0], [1.0], 'go', ms=20, label="Class 2")
#    plt.plot([0.0], [0.0], 'bo', ms=20, label="Class 3")
#    
#    # Plot boundaries of unit simplex
#    plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")
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
#    # Annotate points on the simplex
#    plt.annotate(r'($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)',
#                 xy=(1.0/3, 1.0/3), xytext=(1.0/3, .23), xycoords='data',
#                 arrowprops=dict(facecolor='black', shrink=0.05),
#                 horizontalalignment='center', verticalalignment='center')
#    plt.plot([1.0/3], [1.0/3], 'ko', ms=5)
#    plt.annotate(r'($\frac{1}{2}$, $0$, $\frac{1}{2}$)',
#                 xy=(.5, .0), xytext=(.5, .1), xycoords='data',
#                 arrowprops=dict(facecolor='black', shrink=0.05),
#                 horizontalalignment='center', verticalalignment='center')
#    plt.annotate(r'($0$, $\frac{1}{2}$, $\frac{1}{2}$)',
#                 xy=(.0, .5), xytext=(.1, .5), xycoords='data',
#                 arrowprops=dict(facecolor='black', shrink=0.05),
#                 horizontalalignment='center', verticalalignment='center')
#    plt.annotate(r'($\frac{1}{2}$, $\frac{1}{2}$, $0$)',
#                 xy=(.5, .5), xytext=(.6, .6), xycoords='data',
#                 arrowprops=dict(facecolor='black', shrink=0.05),
#                 horizontalalignment='center', verticalalignment='center')
#    plt.annotate(print(total_confusion_matrix_RF/cycles) #this is averaged over number of cyclesr'($0$, $0$, $1$)',
#                 xy=(0, 0), xytext=(.1, .1), xycoords='data',
#                 arrowprops=dict(facecolor='black', shrink=0.05),
#                 horizontalalignment='center', verticalalignment='center')
#    plt.annotate(r'($1$, $0$, $0$)',
#                 xy=(1, 0), xytext=(1, .1), xycoords='data',
#                 arrowprops=dict(facecolor='black', shrink=0.05),
#                 horizontalalignment='center', verticalalignment='center')
#    plt.annotate(r'($0$, $1$, $0$)',
#                 xy=(0, 1), xytext=(.1, 1), xycoords='data',
#                 arrowprops=dict(facecolor='black', shrink=0.05),
#                 horizontalalignment='center', verticalalignment='center')
#    # Add grid
#    plt.grid("off")
#    for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#        plt.plot([0, x], [x, 0], 'k', alpha=0.2)
#        plt.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
#        plt.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)
#    
#    plt.title("Change of predicted probabilities after sigmoid calibration")
#    plt.xlabel("Probability class 1")
#    plt.ylabel("Probability class 2")
#    plt.xlim(-0.05, 1.05)
#    plt.ylim(-0.05, 1.05)
#    plt.legend(loc="best")
#    
#    print("Log-loss of")
#    print(" * uncalibrated classifier trained on 800 datapoints: %.3f "
#          % score)
#    print(" * classifier trained on 600 datapoints and calibrated on "
#          "200 datapoint: %.3f" % sig_score)
#    
#    # Illustrate calibrator
#    plt.figure(1)
#    # generate grid over 2-simplex
#    p1d = np.linspace(0, 1, 20)
#    p0, p1 = np.meshgrid(p1d, p1d)
#    p2 = 1 - p0 - p1
#    p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
#    p = p[p[:, 2] >= 0]
#    
#    calibrated_classifier = sig_clf.calibrated_classifiers_[0]
#    prediction = np.vstack([calibrator.predict(this_p)
#                            for calibrator, this_p in
#                            zip(calibrated_classifier.calibrators_, p.T)]).T
#    prediction /= prediction.sum(axis=1)[:, None]
#    
#    # Plot modifications of calibrator
#    for i in range(prediction.shape[0]):
#        plt.arrow(p[i, 0], p[i, 1],
#                  prediction[i, 0] - p[i, 0], prediction[i, 1] - p[i, 1],
#                  head_width=1e-2, color=colors[np.argmax(p[i])])
#    # Plot boundaries of unit simplex
#    plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")
#    
#    plt.grid("off")
#    for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#        plt.plot([0, x], [x, 0], 'k', alpha=0.2)
#        plt.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
#        plt.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)
#    
#    plt.title("Illustration of sigmoid calibrator")
#    plt.xlabel("Probability class 1")
#    plt.ylabel("Probability class 2")
#    plt.xlim(-0.05, 1.05)
#    plt.ylim(-0.05, 1.05)
#    
#    plt.show()
    
#    print('--------------------RF--------------------')    
    
    update_index = update_index + 1
     
L_mode_accuracy_RF = []
H_mode_accuracy_RF = []
I_mode_accuracy_RF = []
L_mode_accuracy_valid_RF = []
H_mode_accuracy_valid_RF = []
I_mode_accuracy_valid_RF = []

i = 0 
while i < cycles:
    L_mode_accuracy_RF.append((c_matrix_RF[i][0,0]*1./(c_matrix_RF[i][0,0] + c_matrix_RF[i][0,1] + c_matrix_RF[i][0,2])))
    H_mode_accuracy_RF.append((c_matrix_RF[i][1,1]*1./(c_matrix_RF[i][1,0] + c_matrix_RF[i][1,1] + c_matrix_RF[i][1,2])))
    I_mode_accuracy_RF.append((c_matrix_RF[i][2,2]*1./(c_matrix_RF[i][2,0] + c_matrix_RF[i][2,1] + c_matrix_RF[i][2,2])))
    L_mode_accuracy_valid_RF.append((c_matrix_valid_RF[i][0,0]*1./(c_matrix_valid_RF[i][0,0] + c_matrix_valid_RF[i][0,1] + c_matrix_valid_RF[i][0,2])))
    H_mode_accuracy_valid_RF.append((c_matrix_valid_RF[i][1,1]*1./(c_matrix_valid_RF[i][1,0] + c_matrix_valid_RF[i][1,1] + c_matrix_valid_RF[i][1,2])))
    I_mode_accuracy_valid_RF.append((c_matrix_valid_RF[i][2,2]*1./(c_matrix_valid_RF[i][2,0] + c_matrix_valid_RF[i][2,1] + c_matrix_valid_RF[i][2,2])))
    i = i + 1 
    
print('Random Forest')
print('L-mode accuracy ',np.mean(L_mode_accuracy_RF),' +/- ',np.std(L_mode_accuracy_RF))
print('H-mode accuracy ',np.mean(H_mode_accuracy_RF),' +/- ',np.std(H_mode_accuracy_RF))
print('I-mode accuracy ',np.nanmean(I_mode_accuracy_RF),' +/- ',np.nanstd(I_mode_accuracy_RF)) #use nanmean to account for nan present for I-modes
print('L-mode accuracy (validation) ',np.mean(L_mode_accuracy_valid_RF),' +/- ',np.std(L_mode_accuracy_valid_RF))
print('H-mode accuracy (validation) ',np.mean(H_mode_accuracy_valid_RF),' +/- ',np.std(H_mode_accuracy_valid_RF))
print('I-mode accuracy (validation) ',np.nanmean(I_mode_accuracy_valid_RF),' +/- ',np.nanstd(I_mode_accuracy_valid_RF))

L_mode_accuracy_LR = []
H_mode_accuracy_LR = []
I_mode_accuracy_LR = []
L_mode_accuracy_valid_LR = []
H_mode_accuracy_valid_LR = []
I_mode_accuracy_valid_LR = []

i = 0 
while i < cycles:
    L_mode_accuracy_LR.append((c_matrix_LR[i][0,0]*1./(c_matrix_LR[i][0,0] + c_matrix_LR[i][0,1] + c_matrix_LR[i][0,2])))
    H_mode_accuracy_LR.append((c_matrix_LR[i][1,1]*1./(c_matrix_LR[i][1,0] + c_matrix_LR[i][1,1] + c_matrix_LR[i][1,2])))
    I_mode_accuracy_LR.append((c_matrix_LR[i][2,2]*1./(c_matrix_LR[i][2,0] + c_matrix_LR[i][2,1] + c_matrix_LR[i][2,2])))
    L_mode_accuracy_valid_LR.append((c_matrix_valid_LR[i][0,0]*1./(c_matrix_valid_LR[i][0,0] + c_matrix_valid_LR[i][0,1] + c_matrix_valid_LR[i][0,2])))
    H_mode_accuracy_valid_LR.append((c_matrix_valid_LR[i][1,1]*1./(c_matrix_valid_LR[i][1,0] + c_matrix_valid_LR[i][1,1] + c_matrix_valid_LR[i][1,2])))
    I_mode_accuracy_valid_LR.append((c_matrix_valid_LR[i][2,2]*1./(c_matrix_valid_LR[i][2,0] + c_matrix_valid_LR[i][2,1] + c_matrix_valid_LR[i][2,2])))
    i = i + 1 
    
print('Logistic Regression')
print('L-mode accuracy ',np.mean(L_mode_accuracy_LR),' +/- ',np.std(L_mode_accuracy_LR))
print('H-mode accuracy ',np.mean(H_mode_accuracy_LR),' +/- ',np.std(H_mode_accuracy_LR))
print('I-mode accuracy ',np.nanmean(I_mode_accuracy_LR),' +/- ',np.nanstd(I_mode_accuracy_LR)) #use nanmean to account for nan present for I-modes
print('L-mode accuracy (validation) ',np.mean(L_mode_accuracy_valid_LR),' +/- ',np.std(L_mode_accuracy_valid_LR))
print('H-mode accuracy (validation) ',np.mean(H_mode_accuracy_valid_LR),' +/- ',np.std(H_mode_accuracy_valid_LR))
print('I-mode accuracy (validation) ',np.nanmean(I_mode_accuracy_valid_LR),' +/- ',np.nanstd(I_mode_accuracy_valid_LR))

L_mode_accuracy_NB = []
H_mode_accuracy_NB = []
I_mode_accuracy_NB = []
L_mode_accuracy_valid_NB = []
H_mode_accuracy_valid_NB = []
I_mode_accuracy_valid_NB = []

i = 0 
while i < cycles:
    L_mode_accuracy_NB.append((c_matrix_NB[i][0,0]*1./(c_matrix_NB[i][0,0] + c_matrix_NB[i][0,1] + c_matrix_NB[i][0,2])))
    H_mode_accuracy_NB.append((c_matrix_NB[i][1,1]*1./(c_matrix_NB[i][1,0] + c_matrix_NB[i][1,1] + c_matrix_NB[i][1,2])))
    I_mode_accuracy_NB.append((c_matrix_NB[i][2,2]*1./(c_matrix_NB[i][2,0] + c_matrix_NB[i][2,1] + c_matrix_NB[i][2,2])))
    L_mode_accuracy_valid_NB.append((c_matrix_valid_NB[i][0,0]*1./(c_matrix_valid_NB[i][0,0] + c_matrix_valid_NB[i][0,1] + c_matrix_valid_NB[i][0,2])))
    H_mode_accuracy_valid_NB.append((c_matrix_valid_NB[i][1,1]*1./(c_matrix_valid_NB[i][1,0] + c_matrix_valid_NB[i][1,1] + c_matrix_valid_NB[i][1,2])))
    I_mode_accuracy_valid_NB.append((c_matrix_valid_NB[i][2,2]*1./(c_matrix_valid_NB[i][2,0] + c_matrix_valid_NB[i][2,1] + c_matrix_valid_NB[i][2,2])))
    i = i + 1 
    
print('Naive Bayes')
print('L-mode accuracy ',np.mean(L_mode_accuracy_NB),' +/- ',np.std(L_mode_accuracy_NB))
print('H-mode accuracy ',np.mean(H_mode_accuracy_NB),' +/- ',np.std(H_mode_accuracy_NB))
print('I-mode accuracy ',np.nanmean(I_mode_accuracy_NB),' +/- ',np.nanstd(I_mode_accuracy_NB)) #use nanmean to account for nan present for I-modes
print('L-mode accuracy (validation) ',np.mean(L_mode_accuracy_valid_NB),' +/- ',np.std(L_mode_accuracy_valid_NB))
print('H-mode accuracy (validation) ',np.mean(H_mode_accuracy_valid_NB),' +/- ',np.std(H_mode_accuracy_valid_NB))
print('I-mode accuracy (validation) ',np.nanmean(I_mode_accuracy_valid_NB),' +/- ',np.nanstd(I_mode_accuracy_valid_NB))

L_mode_accuracy_NN = []
H_mode_accuracy_NN = []
I_mode_accuracy_NN = []
L_mode_accuracy_valid_NN = []
H_mode_accuracy_valid_NN = []
I_mode_accuracy_valid_NN = []

i = 0 
while i < cycles:
    L_mode_accuracy_NN.append((c_matrix_NN[i][0,0]*1./(c_matrix_NN[i][0,0] + c_matrix_NN[i][0,1] + c_matrix_NN[i][0,2])))
    H_mode_accuracy_NN.append((c_matrix_NN[i][1,1]*1./(c_matrix_NN[i][1,0] + c_matrix_NN[i][1,1] + c_matrix_NN[i][1,2])))
    I_mode_accuracy_NN.append((c_matrix_NN[i][2,2]*1./(c_matrix_NN[i][2,0] + c_matrix_NN[i][2,1] + c_matrix_NN[i][2,2])))
    L_mode_accuracy_valid_NN.append((c_matrix_valid_NN[i][0,0]*1./(c_matrix_valid_NN[i][0,0] + c_matrix_valid_NN[i][0,1] + c_matrix_valid_NN[i][0,2])))
    H_mode_accuracy_valid_NN.append((c_matrix_valid_NN[i][1,1]*1./(c_matrix_valid_NN[i][1,0] + c_matrix_valid_NN[i][1,1] + c_matrix_valid_NN[i][1,2])))
    I_mode_accuracy_valid_NN.append((c_matrix_valid_NN[i][2,2]*1./(c_matrix_valid_NN[i][2,0] + c_matrix_valid_NN[i][2,1] + c_matrix_valid_NN[i][2,2])))
    i = i + 1 
    
print('NeuralNet')
print('L-mode accuracy ',np.mean(L_mode_accuracy_NN),' +/- ',np.std(L_mode_accuracy_NN))
print('H-mode accuracy ',np.mean(H_mode_accuracy_NN),' +/- ',np.std(H_mode_accuracy_NN))
print('I-mode accuracy ',np.nanmean(I_mode_accuracy_NN),' +/- ',np.nanstd(I_mode_accuracy_NN)) #use nanmean to account for nan present for I-modes
print('L-mode accuracy (validation) ',np.mean(L_mode_accuracy_valid_NN),' +/- ',np.std(L_mode_accuracy_valid_NN))
print('H-mode accuracy (validation) ',np.mean(H_mode_accuracy_valid_NN),' +/- ',np.std(H_mode_accuracy_valid_NN))
print('I-mode accuracy (validation) ',np.nanmean(I_mode_accuracy_valid_NN),' +/- ',np.nanstd(I_mode_accuracy_valid_NN))

#print('Be wary of below results as mean and std of confusion matrix is\
#meaningless since different numbers of each mode present in test set in each\
#cycle; better to calculate average precision/recall/f1 score for each cycle and average that')
#print('RF confusion matrix 00:', np.mean(RF_00),' +/- ',np.std(RF_00))
#print('RF confusion matrix 01:', np.mean(RF_01),' +/- ',np.std(RF_01))
#print('RF confusion matrix 02:', np.mean(RF_02),' +/- ',np.std(RF_02))
#print('RF confusion matrix 10:', np.mean(RF_10),' +/- ',np.std(RF_10))
#print('RF confusion matrix 11:', np.mean(RF_11),' +/- ',np.std(RF_11))
#print('RF confusion matrix 12:', np.mean(RF_12),' +/- ',np.std(RF_12))
#print('RF confusion matrix 20:', np.mean(RF_20),' +/- ',np.std(RF_20))
#print('RF confusion matrix 21:', np.mean(RF_21),' +/- ',np.std(RF_21))
#print('RF confusion matrix 22:', np.mean(RF_22),' +/- ',np.std(RF_22))
#print('NB confusion matrix 00:', np.mean(NB_00),' +/- ',np.std(NB_00))
#print('NB confusion matrix 01:', np.mean(NB_01),' +/- ',np.std(NB_01))
#print('NB confusion matrix 02:', np.mean(NB_02),' +/- ',np.std(NB_02))
#print('NB confusion matrix 10:', np.mean(NB_10),' +/- ',np.std(NB_10))
#print('NB confusion matrix 11:', np.mean(NB_11),' +/- ',np.std(NB_11))
#print('NB confusion matrix 12:', np.mean(NB_12),' +/- ',np.std(NB_12))
#print('NB confusion matrix 20:', np.mean(NB_20),' +/- ',np.std(NB_20))
#print('NB confusion matrix 21:', np.mean(NB_21),' +/- ',np.std(NB_21))
#print('NB confusion matrix 22:', np.mean(NB_22),' +/- ',np.std(NB_22))
#print('NN confusion matrix 00:', np.mean(NN_00),' +/- ',np.std(NN_00))
#print('NN confusion matrix 01:', np.mean(NN_01),' +/- ',np.std(NN_01))
#print('NN confusion matrix 02:', np.mean(NN_02),' +/- ',np.std(NN_02))
#print('NN confusion matrix 10:', np.mean(NN_10),' +/- ',np.std(NN_10))
#print('NN confusion matrix 11:', np.mean(NN_11),' +/- ',np.std(NN_11))
#print('NN confusion matrix 12:', np.mean(NN_12),' +/- ',np.std(NN_12))
#print('NN confusion matrix 20:', np.mean(NN_20),' +/- ',np.std(NN_20))
#print('NN confusion matrix 21:', np.mean(NN_21),' +/- ',np.std(NN_21))
#print('NN confusion matrix 22:', np.mean(NN_22),' +/- ',np.std(NN_22))
#print('LR confusion matrix 00:', np.mean(LR_00),' +/- ',np.std(LR_00))
#print('LR confusion matrix 01:', np.mean(LR_01),' +/- ',np.std(LR_01))
#print('LR confusion matrix 02:', np.mean(LR_02),' +/- ',np.std(LR_02))
#print('LR confusion matrix 10:', np.mean(LR_10),' +/- ',np.std(LR_10))
#print('LR confusion matrix 11:', np.mean(LR_11),' +/- ',np.std(LR_11))
#print('LR confusion matrix 12:', np.mean(LR_12),' +/- ',np.std(LR_12))
#print('LR confusion matrix 20:', np.mean(LR_20),' +/- ',np.std(LR_20))
#print('LR confusion matrix 21:', np.mean(LR_21),' +/- ',np.std(LR_21))
#print('LR confusion matrix 22:', np.mean(LR_22),' +/- ',np.std(LR_22))
  
RF_00 = np.array(RF_00)
RF_01 = np.array(RF_01)
RF_02 = np.array(RF_02)
RF_10 = np.array(RF_10)
RF_11 = np.array(RF_11)
RF_12 = np.array(RF_12)
RF_20 = np.array(RF_20)
RF_21 = np.array(RF_21)
RF_22 = np.array(RF_22)
NB_00 = np.array(NB_00)
NB_01 = np.array(NB_01)
NB_02 = np.array(NB_02)
NB_10 = np.array(NB_10)
NB_11 = np.array(NB_11)
NB_12 = np.array(NB_12)
NB_20 = np.array(NB_20)
NB_21 = np.array(NB_21)
NB_22 = np.array(NB_22)
NN_00 = np.array(NN_00)
NN_01 = np.array(NN_01)
NN_02 = np.array(NN_02)
NN_10 = np.array(NN_10)
NN_11 = np.array(NN_11)
NN_12 = np.array(NN_12)
NN_20 = np.array(NN_20)
NN_21 = np.array(NN_21)
NN_22 = np.array(NN_22)
LR_00 = np.array(LR_00)
LR_01 = np.array(LR_01)
LR_02 = np.array(LR_02)
LR_10 = np.array(LR_10)
LR_11 = np.array(LR_11)
LR_12 = np.array(LR_12)
LR_20 = np.array(LR_20)
LR_21 = np.array(LR_21)
LR_22 = np.array(LR_22)
    
precision_RF_L = RF_00/(RF_20 + RF_10 + RF_00) #actually H-mode out of total predicted H-mode
recall_RF_L = RF_00/(RF_02 + RF_01 + RF_00) #sensitivity
F1_score_RF_L = 2.*precision_RF_L*recall_RF_L/(precision_RF_L + recall_RF_L)
precision_RF_H = RF_11/(RF_21 + RF_11 + RF_01) #actually H-mode out of total predicted H-mode
recall_RF_H = RF_11/(RF_12 + RF_11 + RF_10) #sensitivity
F1_score_RF_H = 2.*precision_RF_H*recall_RF_H/(precision_RF_H + recall_RF_H)
precision_RF_I = RF_22/(RF_22 + RF_12 + RF_02) #actually I-mode out of total predicted H-mode
recall_RF_I = RF_22/(RF_22 + RF_21 + RF_20) #sensitivity
F1_score_RF_I = 2.*precision_RF_I*recall_RF_I/(precision_RF_I + recall_RF_I)
precision_NB_L = NB_00/(NB_20 + NB_10 + NB_00) #actually H-mode out of total predicted H-mode
recall_NB_L = NB_00/(NB_02 + NB_01 + NB_00) #sensitivity
F1_score_NB_L = 2.*precision_NB_L*recall_NB_L/(precision_NB_L + recall_NB_L)
precision_NB_H = NB_11/(NB_21 + NB_11 + NB_01) #actually H-mode out of total predicted H-mode
recall_NB_H = NB_11/(NB_12 + NB_11 + NB_10) #sensitivity
F1_score_NB_H = 2.*precision_NB_H*recall_NB_H/(precision_NB_H + recall_NB_H)
precision_NB_I = NB_22/(NB_22 + NB_12 + NB_02) #actually I-mode out of total predicted H-mode
recall_NB_I = NB_22/(NB_22 + NB_21 + NB_20) #sensitivity
F1_score_NB_I = 2.*precision_NB_I*recall_NB_I/(precision_NB_I + recall_NB_I)
precision_NN_L = NN_00/(NN_20 + NN_10 + NN_00) #actually H-mode out of total predicted H-mode
recall_NN_L = NN_00/(NN_02 + NN_01 + NN_00) #sensitivity
F1_score_NN_L = 2.*precision_NN_L*recall_NN_L/(precision_NN_L + recall_NN_L)
precision_NN_H = NN_11/(NN_21 + NN_11 + NN_01) #actually H-mode out of total predicted H-mode
recall_NN_H = NN_11/(NN_12 + NN_11 + NN_10) #sensitivity
F1_score_NN_H = 2.*precision_NN_H*recall_NN_H/(precision_NN_H + recall_NN_H)
precision_NN_I = NN_22/(NN_22 + NN_12 + NN_02) #actually I-mode out of total predicted H-mode
recall_NN_I = NN_22/(NN_22 + NN_21 + NN_20) #sensitivity
F1_score_NN_I = 2.*precision_NN_I*recall_NN_I/(precision_NN_I + recall_NN_I)
precision_LR_L = LR_00/(LR_20 + LR_10 + LR_00) #actually H-mode out of total predicted H-mode
recall_LR_L = LR_00/(LR_02 + LR_01 + LR_00) #sensitivity
F1_score_LR_L = 2.*precision_LR_L*recall_LR_L/(precision_LR_L + recall_LR_L)
precision_LR_H = LR_11/(LR_21 + LR_11 + LR_01) #actually H-mode out of total predicted H-mode
recall_LR_H = LR_11/(LR_12 + LR_11 + LR_10) #sensitivity
F1_score_LR_H = 2.*precision_LR_H*recall_LR_H/(precision_LR_H + recall_LR_H)
precision_LR_I = LR_22/(LR_22 + LR_12 + LR_02) #actually I-mode out of total predicted H-mode
recall_LR_I = LR_22/(LR_22 + LR_21 + LR_20) #sensitivity
F1_score_LR_I = 2.*precision_LR_I*recall_LR_I/(precision_LR_I + recall_LR_I)

print('precision_RF_L:', np.mean(precision_RF_L),' +/- ',np.std(precision_RF_L))
print('recall_RF_L:', np.mean(recall_RF_L),' +/- ',np.std(recall_RF_L))
print('F1_score_RF_L:', np.mean(F1_score_RF_L),' +/- ',np.std(F1_score_RF_L)) 
print('precision_NB_L:', np.mean(precision_NB_L),' +/- ',np.std(precision_NB_L))
print('recall_NB_L:', np.mean(recall_NB_L),' +/- ',np.std(recall_NB_L))
print('F1_score_NB_L:', np.mean(F1_score_NB_L),' +/- ',np.std(F1_score_NB_L)) 
print('precision_NN_L:', np.mean(precision_NN_L),' +/- ',np.std(precision_NN_L))
print('recall_NN_L:', np.mean(recall_NN_L),' +/- ',np.std(recall_NN_L))
print('F1_score_NN_L:', np.mean(F1_score_NN_L),' +/- ',np.std(F1_score_NN_L))
print('precision_LR_L:', np.mean(precision_LR_L),' +/- ',np.std(precision_LR_L))
print('recall_LR_L:', np.mean(recall_LR_L),' +/- ',np.std(recall_LR_L))
print('F1_score_LR_L:', np.mean(F1_score_LR_L),' +/- ',np.std(F1_score_LR_L))
print('precision_RF_H:', np.mean(precision_RF_H),' +/- ',np.std(precision_RF_H))
print('recall_RF_H:', np.mean(recall_RF_H),' +/- ',np.std(recall_RF_H))
print('F1_score_RF_H:', np.mean(F1_score_RF_H),' +/- ',np.std(F1_score_RF_H)) 
print('precision_NB_H:', np.mean(precision_NB_H),' +/- ',np.std(precision_NB_H))
print('recall_NB_H:', np.mean(recall_NB_H),' +/- ',np.std(recall_NB_H))
print('F1_score_NB_H:', np.mean(F1_score_NB_H),' +/- ',np.std(F1_score_NB_H)) 
print('precision_NN_H:', np.mean(precision_NN_H),' +/- ',np.std(precision_NN_H))
print('recall_NN_H:', np.mean(recall_NN_H),' +/- ',np.std(recall_NN_H))
print('F1_score_NN_H:', np.mean(F1_score_NN_H),' +/- ',np.std(F1_score_NN_H))
print('precision_LR_H:', np.mean(precision_LR_H),' +/- ',np.std(precision_LR_H))
print('recall_LR_H:', np.mean(recall_LR_H),' +/- ',np.std(recall_LR_H))
print('F1_score_LR_H:', np.mean(F1_score_LR_H),' +/- ',np.std(F1_score_LR_H))
print('precision_RF_I:', np.mean(precision_RF_I),' +/- ',np.std(precision_RF_I))
print('recall_RF_I:', np.mean(recall_RF_I),' +/- ',np.std(recall_RF_I))
print('F1_score_RF_I:', np.mean(F1_score_RF_I),' +/- ',np.std(F1_score_RF_I)) 
print('precision_NB_I:', np.mean(precision_NB_I),' +/- ',np.std(precision_NB_I))
print('recall_NB_I:', np.mean(recall_NB_I),' +/- ',np.std(recall_NB_I))
print('F1_score_NB_I:', np.mean(F1_score_NB_I),' +/- ',np.std(F1_score_NB_I)) 
print('precision_NN_I:', np.mean(precision_NN_I),' +/- ',np.std(precision_NN_I))
print('recall_NN_I:', np.mean(recall_NN_I),' +/- ',np.std(recall_NN_I))
print('F1_score_NN_I:', np.mean(F1_score_NN_I),' +/- ',np.std(F1_score_NN_I))
print('precision_LR_I:', np.mean(precision_LR_I),' +/- ',np.std(precision_LR_I))
print('recall_LR_I:', np.mean(recall_LR_I),' +/- ',np.std(recall_LR_I))
print('F1_score_LR_I:', np.mean(F1_score_LR_I),' +/- ',np.std(F1_score_LR_I))

import pickle
#Saving created model
RF_LHI_pkl_filename = '/home/mathewsa/Desktop/RF_classifier_LHI.pkl'
RF_LHI_model_pkl = open(RF_LHI_pkl_filename, 'wb')
pickle.dump(rfc, RF_LHI_model_pkl)
RF_LHI_model_pkl.close()
LR_LHI_pkl_filename = '/home/mathewsa/Desktop/LR_classifier_LHI.pkl'
LR_LHI_model_pkl = open(LR_LHI_pkl_filename, 'wb')
pickle.dump(lr, LR_LHI_model_pkl)
LR_LHI_model_pkl.close()
NB_LHI_pkl_filename = '/home/mathewsa/Desktop/NB_classifier_LHI.pkl'
NB_LHI_model_pkl = open(NB_LHI_pkl_filename, 'wb')
pickle.dump(gnb, NB_LHI_model_pkl)
NB_LHI_model_pkl.close()
NN_LHI_pkl_filename = '/home/mathewsa/Desktop/NN_classifier_LHI.pkl'
NN_LHI_model_pkl = open(NN_LHI_pkl_filename, 'wb')
pickle.dump(mlp, NN_LHI_model_pkl)
NN_LHI_model_pkl.close()
scalerfile = '/home/mathewsa/Desktop/scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))

#Loading saved model
#RF_LHI_model_pkl = open(RF_LHI_pkl_filename, 'rb')
#RF_LHI_model = pickle.load(RF_LHI_model_pkl)
#print("Loaded model :: ", RF_LHI_model)
#scalerfile = '/home/mathewsa/Desktop/scaler.sav'
#scaler = pickle.load(open(scalerfile, 'rb')) 