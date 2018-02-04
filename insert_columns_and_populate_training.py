#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:41:07 2018

@author: Abhilash
"""

"""This code simply inserts new columns into an existing table"""

import sqlite3
from datetime import datetime
import numpy as np
import sys

def main(shot):
    sqlite_file = '/home/mathewsa/Desktop/am_transitions_copy.db'
    table_name = 'confinement_table'
    column1 = 'shot'
    column2 = 'id' # name of the PRIMARY KEY column
    column3 = 'present_mode'
    column4 = 'next_mode'
    column5 = 'time'
    column6 = 'time_at_transition'
    
    new_columns = [['ip','REAL'],['btor','REAL'],['p_lh','REAL'],['p_icrf','REAL'],
    ['p_icrf_d','REAL'],['p_icrf_e','REAL'],['p_icrf_j3','REAL'],['p_icrf_j4','REAL'],
    ['freq_icrf_d','REAL'],['freq_icrf_e','REAL'],['freq_icrf_j','REAL'],['beta_N','REAL'],
    ['beta_p','REAL'],['beta_t','REAL'],['kappa','REAL'],['triang_l','REAL'],['triang_u','REAL'],
    ['triang','REAL'],['li','REAL'],['areao','REAL'],['vout','REAL'],['aout','REAL'],
    ['rout','REAL'],['zout','REAL'],['zmag','REAL'],['rmag','REAL'],['zsep_lower','REAL'],['zsep_upper','REAL'],
    ['rsep_lower','REAL'],['rsep_upper','REAL'],['zvsin','REAL'],['rvsin','REAL'],['zvsout','REAL'],['rvsout','REAL'],
    ['upper_gap','REAL'],['lower_gap','REAL'],['q0','REAL'],['qstar','REAL'],['q95','REAL'],
    ['V_loop_efit','REAL'],['V_surf_efit','REAL'],['Wmhd','REAL'],['cpasma','REAL'],
    ['ssep','REAL'],['P_ohm','REAL'],['HoverHD','REAL'],['Halpha','REAL'],['Dalpha','REAL'],
    ['z_ave','REAL'],['p_rad','REAL'],['p_rad_core','REAL'],['nLave_04','REAL'],['NL_04','REAL'],
    ['nebar_efit','REAL'],['piezo_4_gas_input','REAL'],['g_side_rat','REAL'],['e_bot_mks','REAL'],
    ['b_bot_mks','REAL'],['update_time','TEXT']] 
    
    # Connecting to the database file
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor() 
    
    i = 0
    while i < len(new_columns):
        try:
            cursor.execute("ALTER TABLE {tn} ADD COLUMN '{c}' {ct}"\
                    .format(tn=table_name, c=new_columns[i][0], ct=new_columns[i][1]))
        except:
            print("Column {} already exists".format(new_columns[i][0]))
        i = i + 1 
     
    conn.commit()  
    cursor.execute('select shot,id,present_mode,next_mode,time,time_at_transition from {}'.format(table_name))
    rows = cursor.fetchall()   
    data = np.load('/home/mathewsa/Desktop/single_shot_training_table_py.npz')
    extra_variables = np.load('/home/mathewsa/Desktop/extra_variables.npz')
    
    first_index = 0
    while shot != rows[first_index][0]:
        first_index = first_index + 1
    
    k = first_index #+ rows[0][1] - 1 #since table does not necessarily start at id = 1 (and index starts at 0)
    j = 0
    while (shot == rows[k][0]): 
        time = rows[k][4]  
        if round(((extra_variables['timebase'])[j]),3) == time:
            try:  
                for i in data:   
                    conn.commit()  
                    cursor.execute((("UPDATE {} SET {}= ? WHERE id = ?").\
                    format(table_name,i)),((data['{}'.format(i)])[j],rows[k][1]))  
                    
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
        print(j,j/len((extra_variables['timebase'])))

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main(sys.argv[1])