# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:50:18 2018

@author: mathewsa

Inserting a single column and populating it in an existing database;
This example is only adding column of psurfa and populating this particular column
"""

import sqlite3
from datetime import datetime
import numpy as np
import sys
import MDSplus 
from MDSplus import *
import numpy as np   
import os 
from os import getenv 
sys.path.append('/home/mathewsa/Desktop/confinement_table/codes+/')
import fast_efit06 
import idlpy
from idlpy import *

def smooth(y,box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y,box,mode='same')
    return y_smooth

def checkscalar(val):
    if np.isscalar(val[0]) == False:
	val = np.concatenate(val, axis=0)
    else:
	val = val
    return val

sqlite_old_file = '/home/mathewsa/Desktop/am_transitions_duplicate.db' #location of old database
sqlite_new_file = '/home/mathewsa/Desktop/am_transitions_duplicate.db' #location of new database
table_old_name = 'transitions'  #name of old table in old db
table_new_name = 'confinement_table' #name of new table in new db
column1 = 'shot'
column2 = 'id'
column3 = 'present_mode'
column4 = 'next_mode'
column5 = 'time'
column6 = 'time_at_transition'

# Connecting to the database file
conn = sqlite3.connect(sqlite_old_file)
cursor = conn.cursor() 
cursor.execute('select shot,id,present_mode,next_mode,time,time_at_transition from {old_table}'.format(old_table=table_old_name))
rows = cursor.fetchall() 
end = len(rows[:]) 
conn.commit()
conn.close() #found rows of old , now closing connection

start_times = [] 
end_times = []
list1 = [] #number of shots that start
list2 = [] #number of shots that end
x_old = 'start'
for x in rows:
    if x[0] != x_old:
        print('Start time for '+str(x[0])+' is '+str(x[4]))
        start_times.append(x[4])
        list1.append(x[0])
        x_old = x[0]
    else:
        continue
        
for x in rows:
    if x[3] == 'end':
        print('End time for '+str(x[0])+' is '+str(x[5]))
        end_times.append(x[5])
        list2.append(x[0])
        
i = 0  
timebases = []
if len(list1) == len(list2):
    while i < len(list1):
        timebases.append([list1[i],np.arange(round(start_times[i],3),round(end_times[i]+0.001,3),0.001)])
        i = i + 1
        
def returnNotMatches(a, b):
    output = [[x for x in a if x not in b], [x for x in b if x not in a]]
    if (((not output[0]) == False) or ((not output[1]) == False)):
        raise
    return output
returnNotMatches(list1,list2) #ensures same number of shots with start and end


#making table for training set
shots = list1#[130:] #[1080523020] # input for shots from confinement training table 
#if individual shots, store in ascending order by id 
i = 0
index_i = 0 #additional index if shots != timebase shots
while i < len(shots):
    shot = shots[i] 
    if shot == timebases[index_i][0]:
        pass
    else:
	flag = 1
	while (index_i < len(timebases)) and (flag == 1): 
	    if shot == timebases[index_i][0]:
	        flag = 0
	    else:
                index_i = index_i + 1
	if flag == 1:
	    print('This shot does not exist')
	    raise

    tree = Tree('cmod', shot)
    while True:  
        try: 
            start_time = timebases[index_i][1][0] #in seconds
            end_time = timebases[index_i][1][-1]
            timebase = np.arange(round(start_time,3),round(end_time,3),0.001) #using 1 millisecond constant interval
            tstart = 1000.*timebase[0] #milliseconds
            dt = 1.0 #milliseconds
            tend = 1000.*timebase[-1] #800.0 #milliseconds  
            
            fast_efit06.main(shot,tstart,tend,dt) 
            from fast_efit06 import path_shot
            
            tree = Tree('cmod', shot)
             
            NaN = (np.empty(len(timebase)) * np.nan)
            zeros = np.zeros(len(timebase))
            ones = np.ones(len(timebase))
            
            while True:
                try:  
                    path_shot = path_shot['{}'.format(shot)][0]
                    efit = MDSplus.Tree('{}'.format(path_shot), shot)
                    IDL.run("good_time_slices = efit_check(shot={},tree='{}')".format(shot,path_shot)) 
                    good_time_slices = getattr(IDL,"good_time_slices") #returns indexes for values passing efit_check test
                    time_efit = (efit.getNode('\efit_aeqdsk:time')).data()[good_time_slices]
                    psurfa = (efit.getNode('\efit_aeqdsk:psurfa')).data()[good_time_slices]; #surface area of lcfs
                    psurfa = np.interp(timebase,time_efit,psurfa,left=np.nan,right=np.nan)  
                    break
                except TreeNODATA:
                    time_efit = timebase
                    psurfa = NaN
                    print("No values stored for efit") 
                    print(shot)
                    break
                except:
                    print("Unexpected error for efit")
                    print(shot)
                    raise
            
            np.savez('/home/mathewsa/Desktop/single_shot_training_table_py.npz', psurfa=psurfa)
            np.savez('/home/mathewsa/Desktop/extra_variables.npz', timebase=timebase)

#acquiring data to put into table above
#and now putting data into table below

            sqlite_file = '/home/mathewsa/Desktop/am_transitions_duplicate.db'
            table_name = 'confinement_table'
            column1 = 'shot'
            column2 = 'id' # name of the PRIMARY KEY column
            column3 = 'present_mode'
            column4 = 'next_mode'
            column5 = 'time'
            column6 = 'time_at_transition'
            
            new_columns = [['psurfa','REAL']] 
            
            # Connecting to the database file
            conn = sqlite3.connect(sqlite_file)
            cursor = conn.cursor() 
            
            ii = 0
            while ii < len(new_columns):
                try:
                    cursor.execute("ALTER TABLE {tn} ADD COLUMN '{c}' {ct}"\
                            .format(tn=table_name, c=new_columns[ii][0], ct=new_columns[ii][1]))
                except:
                    print("Column {} already exists".format(new_columns[ii][0]))
                ii = ii + 1 
             
            conn.commit()  
            cursor.execute('select shot,id,present_mode,next_mode,time,time_at_transition from {}'.format(table_name))
            rows = cursor.fetchall()   
            data = np.load('/home/mathewsa/Desktop/single_shot_training_table_py.npz')
            extra_variables = np.load('/home/mathewsa/Desktop/extra_variables.npz')
            
            first_index = 0
            while shot != rows[first_index][0]:
                first_index = first_index + 1
            
            k = first_index  
            j = 0
            while (shot == rows[k][0]): 
                time = rows[k][4]   
                if round(((extra_variables['timebase'])[j]),3) == time: 
                    try:
                        for iii in data:   
                            conn.commit()  
                            cursor.execute((("UPDATE {} SET {}= ? WHERE id = ?").\
                            format(table_name,iii)),((data['{}'.format(iii)])[j],rows[k][1]))  
                        
                        current_time = datetime.now()    
                        current_time = str(datetime.now()) 
                        conn.commit()  
                        cursor.execute("UPDATE {} SET {} = ? WHERE id = ?".format(table_name,\
                        'update_time'),(current_time,rows[k][1]))
                             
                    except sqlite3.IntegrityError:
                        print('ERROR: ID already exists in PRIMARY KEY column {}'.format(column2))
                        #this exception only arises to tell us that we have added shots from the old table already, with "same ID"!
                k = k + 1 
                j = j + 1 
            
            conn.commit()
            conn.close()
 
            break 
        except:
            print("Unexpected error in shot")
            print(shot)
            if (shot == 1091014015) or (shot == 1160616018) or (shot == 1000511018) or (shot == 1050623010):
                pass
            else:
                raise       
    #can define an alternative timebase if desired, but using this
    #definition of > ~100kA as start/end condition currently for table
    i = i + 1