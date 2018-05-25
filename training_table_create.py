#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 23:44:07 2018

@author: Abhilash

This code utilizes scripts for running a fast efit (<5000 steps, 129x129), if
trees efit06 OR efit07 are available. It then calls upon a script to acquire MDSplus
data and then populates a newly created table with that information. It is initalized by
grabbing data from a TRAINING TABLE where every time slice is either L,H,I, or end,
and only time slices for transitions (i.e. mode changes) are included.
"""

import MDSplus 
from MDSplus import *
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
sys.path.append('/home/mathewsa/Desktop/confinement_table/codes+/')
import insert_columns_and_populate_training 
sys.path.append('/home/mathewsa/Desktop/confinement_table/codes+/')
import create_database
sys.path.append('/home/mathewsa/Desktop/confinement_table/codes+/')
import populate_new_database
import sqlite3

sqlite_old_file = '/home/mathewsa/Desktop/am_transitions.db' #location of old database
try:
    create_database.main() #creates confinement_table in database (old = new = same database here)
except:
    print('Table already exists')
populate_new_database.main() #inserts rows at every 1ms in confinement table of database
sqlite_new_file = '/home/mathewsa/Desktop/am_transitions.db' #location of new database
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
rows =cursor.fetchall() 
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
shots = list1[130:] #[1080523020] # input for shots from confinement training table 
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
            data_acquire_MDSplus.main(shot,timebase,path_shot)
            insert_columns_and_populate_training.main(shot) 
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

    
print('Shots to not include in tables except for training are:')
print(shots)
