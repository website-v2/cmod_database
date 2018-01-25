#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:41:07 2018

@author: Abhilash
"""

"""This code simply inserts new columns into an existing table"""

import sqlite3

sqlite_file = '/Users/Abhilash/Desktop/am_transitions_copy.db'
table_name = 'confinement_table'
column1 = 'shot'
column2 = 'id' # name of the PRIMARY KEY column
column3 = 'present_mode'
column4 = 'next_mode'
column5 = 'time'
column6 = 'time_at_transition'

new_column1 = 'Ip'  # name of the new column
new_column2 = 'Bt'  # name of the second new column
column1_type = 'REAL' # E.g., INTEGER, TEXT, NULL, REAL, BLOB
column2_type = 'REAL'
default_val = 0.0 # a default value for the new column rows

# Connecting to the database file
conn = sqlite3.connect(sqlite_file)
cursor = conn.cursor() 
try:
    cursor.execute("ALTER TABLE {tn} ADD COLUMN '{c1}' {c1t}"\
            .format(tn=table_name, c1=new_column1, c1t=column1_type))
except:
    print("Column already exists")
conn.commit()
conn.close()

# Connecting to the database file
conn = sqlite3.connect(sqlite_file)
cursor = conn.cursor() 
try:
    cursor.execute("ALTER TABLE {tn} ADD COLUMN '{c2}' {c2t}"\
            .format(tn=table_name, c2=new_column2, c2t=column2_type))
except:
    print("Column already exists")
conn.commit()
conn.close()