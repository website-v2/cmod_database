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
import operator
import random
import sqlite3
from datetime import datetime
import numpy as np
import sys
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns
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
                            (values['P_ohm'])[i],(values['li'])[i],(values['rmag'])[i],(values['Halpha'])[i],(values['psurfa'])[i]]) #first element must be shot!
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
fraction_ = 0.80
train_valid_frac = 0.80
update_index = 0#(spectroscopy.getNode('\SPECTROSCOPY::z_ave')).units_of()
cycles = 100 #runs
while update_index < cycles:
    print('Fraction of total data for training + validation = ',train_valid_frac)
    print('Fraction of training + validation data used for training = ',fraction_)
    #use below 4 lines if randomizing shots AND time slices for train/validation set
    print("ML_testing_all_normalized_NN_100x100x100_layers_([(values['shot'])[i],(values['Wmhd'])[i],(values['nebar_efit'])[i],(values['beta_p'])[i],\
                            (values['P_ohm'])[i],(values['li'])[i],(values['rmag'])[i],(values['Halpha'])[i],(values['psurfa'])[i]]), cycles =",cycles,\
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

    for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
#                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest'),
                  (mlp, 'NeuralNet')]: 
        clf.fit(X_train_valid, y_train_valid)
    #   if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1] #probability of 1, or H-mode
        prob_pos_valid = clf.predict_proba(X_valid)[:, 1]
        prediction_prob[str(name)] = np.array([int(numeric_string) for numeric_string in prob_pos])
        prediction_prob_valid[str(name)] = np.array([int(numeric_string) for numeric_string in prob_pos_valid])
        prediction[str(name)] = np.array([int(numeric_string) for numeric_string in clf.predict(X_test)])
        prediction_valid[str(name)] = np.array([int(numeric_string) for numeric_string in clf.predict(X_valid)])
    
        c_matrix[str(name)] = array3x3(confusion_matrix(y_test_np, prediction[str(name)]))
        c_matrix_valid[str(name)] = array3x3(confusion_matrix(y_valid_np, prediction_valid[str(name)]))
        
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