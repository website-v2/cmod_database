#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 02:20:13 2018

@author: Abhilash
""" 
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools
import sqlite3
from datetime import datetime
import numpy as np
import sys
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
 
sqlite_file = '/Users/Abhilash/Desktop/am_transitions.db'
table_name = 'confinement_table'
table_name_transitions = 'transitions' 

conn = sqlite3.connect(sqlite_file)
cursor = conn.cursor() 
    
cursor.execute('select shot,id,present_mode,next_mode,time,time_at_transition,ip,\
               btor,p_lh,p_icrf,p_icrf_d,p_icrf_e,p_icrf_j3,p_icrf_j4,freq_icrf_d,\
               freq_icrf_e,freq_icrf_j,beta_N,beta_p,beta_t,kappa,triang_l,triang_u,\
               triang,li,areao,vout,aout,rout,zout,zmag,rmag,zsep_lower,zsep_upper,\
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
        'triang','li','areao','vout','aout','rout','zout','zmag','rmag','zsep_lower','zsep_upper',\
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

Y_data = []
X_data = []
total_x_data = []
bad_shot = 0 #initialize
i = 0 
while i < len(rows):
    for index in ['ip','btor','li','q95','Wmhd','p_icrf','beta_N','nebar_efit','cpasma']:
        while (values[index][i] == None) and ((i+1) < len(rows)):  
            i = i + 1 
        while values['present_mode'][i] == 'I': #not considering I-modes right now
            i = i + 1
        #the above for loop just ensures there is stored value for
        #those quantities being indexed, otherwise skip that column
    if (rows[i][4] > (transitions_start['{}'.format(rows[i][0])] + 0.2)):
        if (rows[i][4] < (transitions_end['{}'.format(rows[i][0])] - 0.2)): 
            if (((values['q95'])[i]) < 2.0) or (((values['li'])[i]) < 1.0) or (((values['e_bot_mks'])[i]) > 200.0):
                if rows[i][0] != bad_shot:
                    print(rows[i][0])
                    bad_shot = rows[i][0] 
            Y_data.append((values['present_mode'])[i])
            X_data.append([(values['ip'])[i],(values['btor'])[i],(values['li'])[i],
                      (values['q95'])[i],(values['Wmhd'])[i],(values['p_icrf'])[i],
                      (values['beta_p'])[i],(values['nebar_efit'])[i]])
            total_x_data.append([(values['ip'])[i],(values['btor'])[i],(values['li'])[i],
                      (values['q95'])[i],(values['Wmhd'])[i],(values['p_icrf'])[i],
                      (values['beta_N'])[i],(values['nebar_efit'])[i],(values['beta_p'])[i],
                      (values['beta_t'])[i],(values['kappa'])[i],(values['triang'])[i],
                      (values['areao'])[i],(values['vout'])[i],(values['aout'])[i],
                      (values['rout'])[i],(values['zout'])[i],
                      (values['zmag'])[i],(values['rmag'])[i],(values['zsep_lower'])[i],
                      (values['zsep_upper'])[i],(values['rsep_lower'])[i],(values['rsep_upper'])[i],
                      (values['zvsin'])[i],(values['rvsin'])[i],(values['zvsout'])[i],
                      (values['rvsout'])[i],(values['upper_gap'])[i],(values['lower_gap'])[i],
                      (values['qstar'])[i],(values['q95'])[i],(values['V_loop_efit'])[i],
                      (values['V_surf_efit'])[i],(values['cpasma'])[i],(values['ssep'])[i],
                      (values['P_ohm'])[i],(values['NL_04'])[i],(values['g_side_rat'])[i],
                      (values['e_bot_mks'])[i],(values['b_bot_mks'])[i]])
    i = i + 1

Y_data = np.where(np.array(Y_data) == 'L', 0, Y_data)
Y_data = np.where(Y_data == 'H', 1, Y_data)
class_names = ['L','H']

#use below 4 lines if randomizing shots AND time slices
#data = np.insert(X_data, len(X_data[0]), values=Y_data, axis=-1)
#random = (np.random.permutation(data))
#X_data = random[:,:-1]
#Y_data = random[:,-1]

#lb = preprocessing.OneHotEncoder()
#Y_data = lb.fit(Y_data)
X_train, y_train = X_data[:int(0.6*len(X_data))], Y_data[:int(0.6*len(X_data))]
X_valid, y_valid = X_data[int(0.6*len(X_data)):int(0.8*len(X_data))], Y_data[int(0.6*len(X_data)):int(0.8*len(X_data))]
X_train_valid, y_train_valid = X_data[:int(0.8*len(X_data))], Y_data[:int(0.8*len(X_data))]
X_test, y_test = X_data[int(0.8*len(X_data)):], Y_data[int(0.8*len(X_data)):]
y_test_np = np.array([int(numeric_string) for numeric_string in y_test])

q = 0
p = 0
while q < len(Y_data):
    if (Y_data[q] == '1') or (Y_data[q] == 1):
        p = p + 1
    q = q + 1
print('H-mode fraction to total time slices: ',p,'/',len(Y_data))
    
    
# Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)

# Plot calibration plots

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

prediction_prob = {}
prediction = {}
sum_array = {}
accuracy = {}
c_matrix = {} #confusion matrix
c_matrix1 = {}

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(X_train_valid, y_train_valid)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1] #probability of 1, or H-mode
        prediction_prob[str(name)] = np.array([int(numeric_string) for numeric_string in prob_pos])
        prediction[str(name)] = np.array([int(numeric_string) for numeric_string in clf.predict(X_test)])
        sum_array[str(name)] = y_test_np + prediction[str(name)]
        accuracy[str(name)] = (sum(i == 2 for i in sum_array[str(name)]) + sum(i == 0 for i in sum_array[str(name)]))/len(y_test) 
        c_matrix[str(name)] = confusion_matrix(y_test_np, prediction[str(name)])
        c_matrix1[str(name)+' predicted L-mode correctly'] = c_matrix[str(name)][0][0]/(c_matrix[str(name)][0][0] + (c_matrix[str(name)][1][0]))
        c_matrix1[str(name)+' predicted H-mode correctly'] = c_matrix[str(name)][1][1]/(c_matrix[str(name)][1][1] + (c_matrix[str(name)][0][1]))
#2 is H-mode correctly classified; 0 is L-mode correctly classified; 1 is misclassfication     
    else:  # use decision function
        prediction[str(name)] = np.array([int(numeric_string) for numeric_string in clf.predict(X_test)])
        sum_array[str(name)] = y_test_np + prediction[str(name)]
        accuracy[str(name)] = (sum(i == 2 for i in sum_array[str(name)]) + sum(i == 0 for i in sum_array[str(name)]))/len(y_test) 
        c_matrix[str(name)] = confusion_matrix(y_test_np, prediction[str(name)])
        c_matrix1[str(name)+' predicted L-mode correctly'] = c_matrix[str(name)][0][0]/(c_matrix[str(name)][0][0] + (c_matrix[str(name)][1][0]))
        c_matrix1[str(name)+' predicted H-mode correctly'] = c_matrix[str(name)][1][1]/(c_matrix[str(name)][1][1] + (c_matrix[str(name)][0][1]))
#2 is H-mode correctly classified; 0 is L-mode correctly classified; 1 is misclassfication
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=20)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=20, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()

## Train uncalibrated random forest classifier on whole train and validation
## data and evaluate on test data
#clf = RandomForestClassifier(n_estimators=25)
#clf.fit(X_train_valid, y_train_valid)
#clf_probs = clf.predict_proba(X_test)
#score = log_loss(y_test, clf_probs)
#
## Train random forest classifier, calibrate on validation data and evaluate
## on test data
#clf = RandomForestClassifier(n_estimators=25)
#clf.fit(X_train, y_train)
#clf_probs = clf.predict_proba(X_test)
#sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
#sig_clf.fit(X_valid, y_valid)
#sig_clf_probs = sig_clf.predict_proba(X_test)
#sig_score = log_loss(y_test, sig_clf_probs)

## Plot changes in predicted probabilities via arrows
#plt.figure(0)
#colors = ["r", "g", "b"]
#for i in range(clf_probs.shape[0]):
#    plt.arrow(clf_probs[i, 0], clf_probs[i, 1],
#              sig_clf_probs[i, 0] - clf_probs[i, 0],
#              sig_clf_probs[i, 1] - clf_probs[i, 1],
#              color=colors[y_test[i]], head_width=1e-2)
#
## Plot perfect predictions
#plt.plot([1.0], [0.0], 'ro', ms=20, label="Class 1")
#plt.plot([0.0], [1.0], 'go', ms=20, label="Class 2")
#plt.plot([0.0], [0.0], 'bo', ms=20, label="Class 3")
#
## Plot boundaries of unit simplex
#plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")
#
## Annotate points on the simplex
#plt.annotate(r'($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)',
#             xy=(1.0/3, 1.0/3), xytext=(1.0/3, .23), xycoords='data',
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             horizontalalignment='center', verticalalignment='center')
#plt.plot([1.0/3], [1.0/3], 'ko', ms=5)
#plt.annotate(r'($\frac{1}{2}$, $0$, $\frac{1}{2}$)',
#             xy=(.5, .0), xytext=(.5, .1), xycoords='data',
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             horizontalalignment='center', verticalalignment='center')
#plt.annotate(r'($0$, $\frac{1}{2}$, $\frac{1}{2}$)',
#             xy=(.0, .5), xytext=(.1, .5), xycoords='data',
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             horizontalalignment='center', verticalalignment='center')
#plt.annotate(r'($\frac{1}{2}$, $\frac{1}{2}$, $0$)',
#             xy=(.5, .5), xytext=(.6, .6), xycoords='data',
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             horizontalalignment='center', verticalalignment='center')
#plt.annotate(r'($0$, $0$, $1$)',
#             xy=(0, 0), xytext=(.1, .1), xycoords='data',
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             horizontalalignment='center', verticalalignment='center')
#plt.annotate(r'($1$, $0$, $0$)',
#             xy=(1, 0), xytext=(1, .1), xycoords='data',
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             horizontalalignment='center', verticalalignment='center')
#plt.annotate(r'($0$, $1$, $0$)',
#             xy=(0, 1), xytext=(.1, 1), xycoords='data',
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             horizontalalignment='center', verticalalignment='center')
## Add grid
#plt.grid("off")
#for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#    plt.plot([0, x], [x, 0], 'k', alpha=0.2)
#    plt.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
#    plt.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)
#
#plt.title("Change of predicted probabilities after sigmoid calibration")
#plt.xlabel("Probability class 1")
#plt.ylabel("Probability class 2")
#plt.xlim(-0.05, 1.05)
#plt.ylim(-0.05, 1.05)
#plt.legend(loc="best")
#
#print("Log-loss of")
#print(" * uncalibrated classifier trained on 80% datapoints: %.3f "
#      % score)
#print(" * classifier trained on 60% datapoints and calibrated on "
#      "20% datapoint: %.3f" % sig_score)
#
## Illustrate calibrator
#plt.figure(1)
## generate grid over 2-simplex
#p1d = np.linspace(0, 1, 20)
#p0, p1 = np.meshgrid(p1d, p1d)
#p2 = 1 - p0 - p1
#p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
#p = p[p[:, 2] >= 0]
#
#calibrated_classifier = sig_clf.calibrated_classifiers_[0]
#prediction = np.vstack([calibrator.predict(this_p)
#                        for calibrator, this_p in
#                        zip(calibrated_classifier.calibrators_, p.T)]).T
#prediction /= prediction.sum(axis=1)[:, None]
#
## Plot modifications of calibrator
#for i in range(prediction.shape[0]):
#    plt.arrow(p[i, 0], p[i, 1],
#              prediction[i, 0] - p[i, 0], prediction[i, 1] - p[i, 1],
#              head_width=1e-2, color=colors[np.argmax(p[i])])
## Plot boundaries of unit simplex
#plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")
#
#plt.grid("off")
#for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#    plt.plot([0, x], [x, 0], 'k', alpha=0.2)
#    plt.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
#    plt.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)
#
#plt.title("Illustration of sigmoid calibrator")
#plt.xlabel("Probability class 1")
#plt.ylabel("Probability class 2")
#plt.xlim(-0.05, 1.05)
#plt.ylim(-0.05, 1.05)
#
#plt.show()

print(accuracy)
print(c_matrix)
print('Test set has '+str(np.sum(y_test_np == 1))+' H-modes and '+str(np.sum(y_test_np == 0))+' L-modes')
print(c_matrix1)

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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(c_matrix['Logistic'], classes=class_names, normalize=True,
                      title='Normalized confusion matrix, Logistic')
plt.figure()
plot_confusion_matrix(c_matrix['Naive Bayes'], classes=class_names,
                      title='Confusion matrix, without normalization, NB')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(c_matrix['Naive Bayes'], classes=class_names, normalize=True,
                      title='Normalized confusion matrix, NB')
plt.figure()
plot_confusion_matrix(c_matrix['Support Vector Classification'], classes=class_names,
                      title='Confusion matrix, without normalization, SVC')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(c_matrix['Support Vector Classification'], classes=class_names, normalize=True,
                      title='Normalized confusion matrix, SVC')
plt.figure()
plot_confusion_matrix(c_matrix['Random Forest'], classes=class_names,
                      title='Confusion matrix, without normalization, RF')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(c_matrix['Random Forest'], classes=class_names, normalize=True,
                      title='Normalized confusion matrix, RF')
plt.show()


X_test = np.array(X_test)
X_train_valid = np.array(X_train_valid)
n, bins, patches = plt.hist(X_train_valid[:,0], 20, density=False, label = 'Ip, train', facecolor='g', alpha=0.75)
n, bins, patches = plt.hist(X_test[:,0], 20, density=False, label = 'Ip, test', facecolor='r', alpha=0.75)
plt.legend()
plt.show()
n, bins, patches = plt.hist(X_train_valid[:,1], 20, density=False, label = 'Btor, train', facecolor='g', alpha=0.75)
n, bins, patches = plt.hist(X_test[:,1], 20, density=False, label = 'Btor, test', facecolor='r', alpha=0.75)
plt.legend()
plt.show()
n, bins, patches = plt.hist(X_train_valid[:,2], 20, density=False, label = 'li, train', facecolor='g', alpha=0.75)
n, bins, patches = plt.hist(X_test[:,2], 20, density=False, label = 'li, test', facecolor='r', alpha=0.75)
plt.legend()
plt.show()
n, bins, patches = plt.hist(X_train_valid[:,3], 20, density=False, label = 'q95, train', facecolor='g', alpha=0.75)
n, bins, patches = plt.hist(X_test[:,3], 20, density=False, label = 'q95, test', facecolor='r', alpha=0.75)
plt.legend()
plt.show()
n, bins, patches = plt.hist(X_train_valid[:,4], 20, density=False, label = 'Wmhd, train', facecolor='g', alpha=0.75)
n, bins, patches = plt.hist(X_test[:,4], 20, density=False, label = 'Wmhd, test', facecolor='r', alpha=0.75)
plt.legend()
plt.show()
n, bins, patches = plt.hist(X_train_valid[:,5], 20, density=False, label = 'p_icrf, train', facecolor='g', alpha=0.75)
n, bins, patches = plt.hist(X_test[:,5], 20, density=False, label = 'p_icrf, test', facecolor='r', alpha=0.75)
plt.legend()
plt.show()
n, bins, patches = plt.hist(X_train_valid[:,6], 20, density=False, label = 'beta_N, train', facecolor='g', alpha=0.75)
n, bins, patches = plt.hist(X_test[:,6], 20, density=False, label = 'beta_N, test', facecolor='r', alpha=0.75)
plt.legend()
plt.show()
n, bins, patches = plt.hist(X_train_valid[:,7], 20, density=False, label = 'nebar_efit, train', facecolor='g', alpha=0.75)
n, bins, patches = plt.hist(X_test[:,7], 20, density=False, label = 'nebar_efit, test', facecolor='r', alpha=0.75)
plt.legend()
plt.show()


X_std = StandardScaler().fit_transform(total_x_data)
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
    
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp) 

plt.plot(range(0,len(total_x_data[0])),var_exp)
plt.plot(range(0,len(total_x_data[0])),cum_var_exp,label='cumulative explained variance')
plt.ylabel('Explained variance in percent')
plt.title('Explained variance by different principal components') 
plt.legend()
plt.show()


list_vector = []
names = ['ip','btor','li','q95','Wmhd','p_icrf','beta_N','nebar_efit','beta_p','beta_t','kappa','triang','areao','vout','aout','rout','zout','zmag','rmag','zsep_lower','zsep_upper','rsep_lower','rsep_upper','zvsin','rvsin','zvsout','rvsout','upper_gap','lower_gap','qstar','q95','V_loop_efit','V_surf_efit','cpasma','ssep','P_ohm','NL_04','g_side_rat','e_bot_mks','b_bot_mks']
k = 0 
while k < len(eig_vecs):
    j = 0
    for i in eig_vecs[k]:
        list_vector.append([k,i,j,names[j]]) #j and names[j] is essentially redundant here
        j = j + 1
    k = k + 1
    
k = 0
while k < len(eig_vecs):
    temp_sort = list_vector[(40*k):(39 + 40*k)]
    temp_sort = sorted(temp_sort, key=lambda tup: abs(tup[1]),reverse=True)
    print(temp_sort[0:20])
    k = k + 1