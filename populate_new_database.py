#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:57:05 2018

@author: Abhilash
"""

"""This code inserts new rows into a database based on knowledge of another existing database
    consisting of only transitions and populates the database with data for every 1 millisecond"""

import sqlite3
import numpy as np

def main():
    sqlite_old_file = '/home/mathewsa/Desktop/am_transitions.db' #location of old database
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
    rows = cursor.fetchall() 
    end = len(rows[:]) 
    conn.commit()
    conn.close() #found rows of old , now closing connection
    
    conn = sqlite3.connect(sqlite_new_file) # now open new table
    cursor = conn.cursor() 
    cursor.execute('select shot,id,present_mode,next_mode,time,time_at_transition from {new_table}'.format(new_table=table_new_name))
    rows_new = cursor.fetchall()
    
    unique_id = str(213939) #str(rows[end-1][1] + 1)  #usually start at next index of old_table, but here I am adding to an existing table
    i = 538 #0 #usually start at 0, but here I am adding shots beginning at index i = 538 in rows[i]
    while i < end: 
        time = rows[i][4]
        try:
            print(time,rows[i][5])
            while (float(time) < (rows[i][5] - 0.0005)): #since only precision up to 1 millisecond used
            #ensures that at transition time, new mode is registered in database
                values = [rows[i][0],unique_id,rows[i][2],rows[i][3],time,rows[i][5]]
                cursor.execute("INSERT INTO {tn} ({c1},{c2},{c3},{c4},{c5},{c6}) VALUES ({shot_},{id_},'{mode1}','{mode2}',{t1},{t2})".\
                    format(tn=table_new_name,c1=column1,c2=column2,c3=column3,c4=column4,c5=column5,c6=column6,shot_=values[0],\
                           id_=values[1],mode1=values[2],mode2=values[3],t1=values[4],t2=values[5]))
                time = (float(time) + 0.001) 
                time = "{:.3f}".format(time) #makes time up to 3 decimal places (i.e. 0.001s) 
                unique_id = str(int(unique_id) + 1) 
        except sqlite3.IntegrityError:
            print('ERROR: ID already exists in PRIMARY KEY column {}'.format(column2))
            #this exception only arises to tell us that we have added shots from the old table already, with "same ID"!
        i = i + 1 
    
    conn.commit()
    conn.close()
    
     
    start_time = [] 
    end_time = []
    list1 = [] #number of shots that start
    list2 = [] #number of shots that end
    x_old = 'start'
    for x in rows: 
        if x[0] != x_old: 
            print('Start time for '+str(x[0])+' is '+str(x[4]))
            start_time.append(x[4])
            list1.append(x[0]) 
            x_old = x[0]
        else:
            continue
            
    for x in rows:
        if x[3] == 'end':
            print('End time for '+str(x[0])+' is '+str(x[5]))
            end_time.append(x[5])
            list2.append(x[0])
            
    i = 0 #use this for timebase if not based on 100kA condition of ip
    timebase = []
    if len(list1) == len(list2):
        while i < len(list1): 
            timebase.append([list1[i],np.arange(round(start_time[i],3),round(end_time[i],3),0.001)])
            i = i + 1
            
    def returnNotMatches(a, b):
        output = [[x for x in a if x not in b], [x for x in b if x not in a]]
        if (((not output[0]) == False) or ((not output[1]) == False)):
            raise
        return output
    returnNotMatches(list1,list2) #ensures same number of shots with start and end
    
if __name__ == "__main__":
    main()