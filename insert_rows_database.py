#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 00:45:27 2018

@author: Abhilash
"""

"""This code simply inserts new rows into an existing database"""

import sqlite3

sqlite_file = '/Users/Abhilash/Desktop/am_transitions_copy.db'
table_name = 'confinement_table'
column1 = 'shot'
column2 = 'id'
column3 = 'present_mode'
column4 = 'next_mode'
column5 = 'time'
column6 = 'time_at_transition'

# Connecting to the database file
conn = sqlite3.connect(sqlite_file)
cursor = conn.cursor() 
cursor.execute('select shot,id,present_mode,next_mode,time,time_at_transition from transitions_20171207')
rows =cursor.fetchall() 
end = len(rows[:]) 
i = 0
while i < end: 
    try:
        while (rows[i][4]!=rows[i][5]): #NEED TO SORT BY SHOT OR MAKE NEW DATABASE FROM OLD 
            time = str(rows[i][4] + 0.001)
            unique_id = str(rows[end-1][1] + 1) 
            values = [rows[i][0],unique_id,rows[i][2],rows[i][3],time,rows[i][5]]
            cursor.execute("INSERT INTO {tn} ({c1},{c2},{c3},{c4},{c5},{c6}) VALUES ({shot_},{id_},'{mode1}','{mode2}',{t1},{t2})".\
                format(tn=table_name,c1=column1,c2=column2,c3=column3,c4=column4,c5=column5,c6=column6,shot_=values[0],\
                       id_=values[1],mode1=values[2],mode2=values[3],t1=values[4],t2=values[5]))
    except sqlite3.IntegrityError:
        print('ERROR: ID already exists in PRIMARY KEY column {}'.format(column2))
        print(i)
    i = i + 1

## B) Tries to insert an ID (if it does not exist yet)
## with a specific value in a second column
#c.execute("INSERT OR IGNORE INTO {tn} ({idf}, {cn}) VALUES (123456, 'test')".\
#        format(tn=table_name, idf=id_column, cn=column_name))
#
## C) Updates the newly inserted or pre-existing entry            
#c.execute("UPDATE {tn} SET {cn}=('Hi World') WHERE {idf}=(123456)".\
#        format(tn=table_name, cn=column_name, idf=id_column))

conn.commit()
conn.close()