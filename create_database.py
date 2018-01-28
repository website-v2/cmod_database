#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:38:40 2018

@author: Abhilash
"""

"""This code simply creates a new database with specified columns"""

import sqlite3

sqlite_file = '/Users/Abhilash/Desktop/am_transitions_copy.db'    # name of the sqlite database file
table_name1 = 'confinement_table'  # name of the table to be created
field1 = 'shot' # name of the column
field1_type = 'INTEGER'  # column data type
field2 = 'id' 
field2_type = 'INTEGER'  
field3 = 'present_mode' 
field3_type = 'TEXT'  
field4 = 'next_mode' 
field4_type = 'TEXT'  
field5 = 'time' 
field5_type = 'REAL'  
field6 = 'time_at_transition' 
field6_type = 'REAL'  

conn = sqlite3.connect(sqlite_file)
cursor = conn.cursor()

# Creating a table with 1 column and set it as PRIMARY KEY
# note that PRIMARY KEY column must consist of unique values!
cursor.execute('CREATE TABLE {tn} ({f1} {f1t}, {f2} {f2t} PRIMARY KEY,\
               {f3} {f3t}, {f4} {f4t}, {f5} {f5t}, {f6} {f6t}) '\
        .format(tn=table_name1, f1=field1, f1t=field1_type, f2=field2, f2t=field2_type,\
        f3=field3, f3t=field3_type,f4=field4, f4t=field4_type, f5=field5, f5t=field5_type,\
        f6=field6, f6t=field6_type))