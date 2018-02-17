# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:49:14 2018

@author: mathewsa
""" 

import sqlite3
from datetime import datetime
import numpy as np
import sys
import matplotlib.pyplot as plt 
 
sqlite_file = '/home/mathewsa/Desktop/am_transitions.db'
table_name = 'confinement_table'
table_name_transitions = 'transitions_20171207'
column1 = 'shot'
column2 = 'id' # name of the PRIMARY KEY column
column3 = 'present_mode'
column4 = 'next_mode'
column5 = 'time'

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
    

#below loop ensures only values where [tstart+0.5,tend-0.5] are included
values = {} #dictionary where all values for each column can be found 
i_column = 0
for column in columns:
    i = 0
    values[column] = []
    while i < len(rows):
        values[column].append(rows[i][i_column])
        i = i + 1  
    i_column = i_column + 1

a = 0.22#m for Alcator C-Mod minor radius
L_mode_ip = []
L_mode_btor = [] 
L_mode_p_icrf = [] 
L_mode_beta_N = [] 
L_mode_beta_p = []
L_mode_kappa = [] 
L_mode_triang = [] 
L_mode_li = [] 
L_mode_q95 = [] 
L_mode_Dalpha = []  
L_mode_p_rad = [] 
L_mode_p_rad_core = [] 
L_mode_nLave_04 = [] 
L_mode_nebar_efit = [] 
L_mode_e_bot_mks = [] 
L_mode_P_ohm = [] 
L_mode_Wmhd = [] 
L_mode_V_loop_efit = []  
L_mode_HoverHD = [] 
n_nG_L = []

H_mode_ip = []
H_mode_btor = [] 
H_mode_p_icrf = [] 
H_mode_beta_N = [] 
H_mode_beta_p = []
H_mode_kappa = [] 
H_mode_triang = [] 
H_mode_li = [] 
H_mode_q95 = [] 
H_mode_Dalpha = []  
H_mode_p_rad = [] 
H_mode_p_rad_core = [] 
H_mode_nLave_04 = [] 
H_mode_nebar_efit = [] 
H_mode_e_bot_mks = [] 
H_mode_P_ohm = [] 
H_mode_Wmhd = [] 
H_mode_V_loop_efit = []  
H_mode_HoverHD = [] 
n_nG_H = []

I_mode_ip = []
I_mode_btor = [] 
I_mode_p_icrf = [] 
I_mode_beta_N = [] 
I_mode_beta_p = []
I_mode_kappa = [] 
I_mode_triang = [] 
I_mode_li = [] 
I_mode_q95 = [] 
I_mode_Dalpha = []  
I_mode_p_rad = [] 
I_mode_p_rad_core = [] 
I_mode_nLave_04 = [] 
I_mode_nebar_efit = [] 
I_mode_e_bot_mks = [] 
I_mode_P_ohm = [] 
I_mode_Wmhd = [] 
I_mode_V_loop_efit = []  
I_mode_HoverHD = [] 
n_nG_I = []

i = 0 
while i < len(rows):
    for index in ['ip','btor','p_icrf','q95','beta_p','nLave_04']:
        while (values[index][i] == None) and ((i+1) < len(rows)):  
            i = i + 1 
        #the above for loop just ensures there is stored value for
        #those quantities being indexed, otherwise skip that column
    if (rows[i][4] > (transitions_start['{}'.format(rows[i][0])] + 0.5)):
        if (rows[i][4] < (transitions_end['{}'.format(rows[i][0])] - 0.5)):
            if (rows[i][0]) not in (1160803022, 1160803023, 1120824016, 1160805005, 1100902012, 1100902023, 1090914015):
                present_mode = (values['present_mode'])[i] 
                if (((values['q95'])[i]) < 2.0) or (((values['li'])[i]) < 1.0) or (((values['e_bot_mks'])[i]) > 200.0):
                    print(rows[i][0])
                if present_mode == 'L': 
                    L_mode_ip.append((values['ip'])[i]) 
                    L_mode_btor.append((values['btor'])[i])
                    L_mode_p_icrf.append((values['p_icrf'])[i])
                    L_mode_beta_N.append((values['beta_N'])[i])
                    L_mode_beta_p.append((values['beta_p'])[i])
                    L_mode_kappa.append((values['kappa'])[i])
                    L_mode_triang.append((values['triang'])[i])
                    L_mode_li.append((values['li'])[i])
                    L_mode_q95.append((values['q95'])[i])
                    L_mode_Dalpha.append((values['Dalpha'])[i])
                    L_mode_p_rad.append((values['p_rad'])[i])
                    L_mode_p_rad_core.append((values['p_rad_core'])[i])
                    L_mode_nLave_04.append((values['nLave_04'])[i])
                    L_mode_nebar_efit.append((values['nebar_efit'])[i])
                    L_mode_e_bot_mks.append((values['e_bot_mks'])[i])
                    L_mode_P_ohm.append((values['P_ohm'])[i])
                    L_mode_Wmhd.append((values['Wmhd'])[i])
                    L_mode_V_loop_efit.append((values['V_loop_efit'])[i])
                    L_mode_HoverHD.append((values['HoverHD'])[i])
                    nG = ((((values['ip'])[i])/(np.pi*a*a))/(10.**6.))*(10.**20.)
                    n_nG_L.append(((values['nLave_04'])[i])/nG)  
                elif present_mode == 'H': 
                    H_mode_ip.append((values['ip'])[i])
                    H_mode_btor.append((values['btor'])[i])
                    H_mode_p_icrf.append((values['p_icrf'])[i])
                    H_mode_beta_N.append((values['beta_N'])[i])
                    H_mode_beta_p.append((values['beta_p'])[i])
                    H_mode_kappa.append((values['kappa'])[i])
                    H_mode_triang.append((values['triang'])[i])
                    H_mode_li.append((values['li'])[i])
                    H_mode_q95.append((values['q95'])[i])
                    H_mode_Dalpha.append((values['Dalpha'])[i])
                    H_mode_p_rad.append((values['p_rad'])[i])
                    H_mode_p_rad_core.append((values['p_rad_core'])[i])
                    H_mode_nLave_04.append((values['nLave_04'])[i])
                    H_mode_nebar_efit.append((values['nebar_efit'])[i])
                    H_mode_e_bot_mks.append((values['e_bot_mks'])[i])
                    H_mode_P_ohm.append((values['P_ohm'])[i])
                    H_mode_Wmhd.append((values['Wmhd'])[i])
                    H_mode_V_loop_efit.append((values['V_loop_efit'])[i])
                    H_mode_HoverHD.append((values['HoverHD'])[i])
                    nG = ((((values['ip'])[i])/(np.pi*a*a))/(10.**6.))*(10.**20.)
                    n_nG_H.append(((values['nLave_04'])[i])/nG)  
                elif present_mode == 'I': 
                    I_mode_ip.append((values['ip'])[i])
                    I_mode_btor.append((values['btor'])[i])
                    I_mode_p_icrf.append((values['p_icrf'])[i])
                    I_mode_beta_N.append((values['beta_N'])[i])
                    I_mode_beta_p.append((values['beta_p'])[i])
                    I_mode_kappa.append((values['kappa'])[i])
                    I_mode_triang.append((values['triang'])[i])
                    I_mode_li.append((values['li'])[i])
                    I_mode_q95.append((values['q95'])[i])
                    I_mode_Dalpha.append((values['Dalpha'])[i])
                    I_mode_p_rad.append((values['p_rad'])[i])
                    I_mode_p_rad_core.append((values['p_rad_core'])[i])
                    I_mode_nLave_04.append((values['nLave_04'])[i])
                    I_mode_nebar_efit.append((values['nebar_efit'])[i])
                    I_mode_e_bot_mks.append((values['e_bot_mks'])[i])
                    I_mode_P_ohm.append((values['P_ohm'])[i])
                    I_mode_Wmhd.append((values['Wmhd'])[i])
                    I_mode_V_loop_efit.append((values['V_loop_efit'])[i])
                    I_mode_HoverHD.append((values['HoverHD'])[i])
                    nG = ((((values['ip'])[i])/(np.pi*a*a))/(10.**6.))*(10.**20.)
                    n_nG_I.append(((values['nLave_04'])[i])/nG)  
                else:
                    print('Error')
                    raise 
    i = i + 1
    
plt.figure() 
plt.scatter(n_nG_L[::1],L_mode_beta_p[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(n_nG_H[::1],H_mode_beta_p[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(n_nG_I[::1],I_mode_beta_p[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('n/nG')
plt.ylabel('beta poloidal')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('beta_v_nG.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(n_nG_L[::1],L_mode_beta_p[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(n_nG_H[::1],H_mode_beta_p[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(n_nG_I[::1],I_mode_beta_p[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('n/nG')
plt.ylabel('beta poloidal')
plt.xlim(0.0,1.2)
plt.legend(loc=2)
plt.savefig('beta_v_nG_01.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(n_nG_L[::1],L_mode_beta_p[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(n_nG_H[::1],H_mode_beta_p[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(n_nG_I[::1],I_mode_beta_p[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('n/nG')
plt.ylabel('beta poloidal')
#plt.xlim(-1.2,0.0)
plt.legend(loc=2)
plt.savefig('beta_v_nG_-01.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(n_nG_L[::1],L_mode_Wmhd[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(n_nG_H[::1],H_mode_Wmhd[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(n_nG_I[::1],I_mode_Wmhd[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('n/nG')
plt.ylabel('Wmhd')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('Wmhd_v_nG.png')
plt.draw()
plt.close()
 
plt.figure() 
plt.scatter(n_nG_L[::1],L_mode_q95[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(n_nG_H[::1],H_mode_q95[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(n_nG_I[::1],I_mode_q95[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('n/nG')
plt.ylabel('q95')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('q95_v_nG.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(n_nG_L[::1],L_mode_Dalpha[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(n_nG_H[::1],H_mode_Dalpha[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(n_nG_I[::1],I_mode_Dalpha[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('n/nG')
plt.ylabel('Dalpha')
#plt.ylim(0.0,100.)
plt.legend(loc=2)
plt.savefig('Dalpha_v_nG.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(n_nG_L[::1],L_mode_li[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(n_nG_H[::1],H_mode_li[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(n_nG_I[::1],I_mode_li[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('n/nG')
plt.ylabel('li')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('li_v_nG.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(n_nG_L[::1],L_mode_p_rad[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(n_nG_H[::1],H_mode_p_rad[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(n_nG_I[::1],I_mode_p_rad[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('n/nG')
plt.ylabel('p_rad')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('p_rad_v_nG.png')
plt.draw()
plt.close() 

plt.figure() 
plt.scatter(n_nG_L[::1],L_mode_e_bot_mks[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(n_nG_H[::1],H_mode_e_bot_mks[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(n_nG_I[::1],I_mode_e_bot_mks[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('n/nG')
plt.ylabel('e_bot_mks')
#plt.ylim(0.0,30.0)
plt.legend(loc=2)
plt.savefig('e_bot_mks_v_nG.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(n_nG_L[::1],L_mode_HoverHD[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(n_nG_H[::1],H_mode_HoverHD[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(n_nG_I[::1],I_mode_HoverHD[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('n/nG')
plt.ylabel('HoverHD')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('HoverHD_v_nG.png')
plt.draw()
plt.close()

 

 
plt.figure() 
plt.scatter(L_mode_Dalpha[::1],L_mode_p_icrf[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_Dalpha[::1],H_mode_p_icrf[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_Dalpha[::1],I_mode_p_icrf[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('Dalpha')
plt.ylabel('p_icrf')
#plt.xlim(0.0,100.)
plt.legend(loc=2)
plt.savefig('p_icrf_v_Dalpha.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(L_mode_beta_N[::1],L_mode_beta_p[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_beta_N[::1],H_mode_beta_p[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_beta_N[::1],I_mode_beta_p[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('beta_N')
plt.ylabel('beta_p')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('beta_p_v_beta_N.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(L_mode_li[::1],L_mode_q95[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_li[::1],H_mode_q95[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_li[::1],I_mode_q95[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('li')
plt.ylabel('q95')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('q95_v_li.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(L_mode_P_ohm[::1],L_mode_Wmhd[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_P_ohm[::1],H_mode_Wmhd[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_P_ohm[::1],I_mode_Wmhd[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('P_ohm')
plt.ylabel('Wmhd')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('Wmhd_v_P_ohm.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(L_mode_kappa[::1],L_mode_triang[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_kappa[::1],H_mode_triang[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_kappa[::1],I_mode_triang[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('kappa')
plt.ylabel('triang')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('triang_v_kappa.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(L_mode_P_ohm[::1],L_mode_Dalpha[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_P_ohm[::1],H_mode_Dalpha[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_P_ohm[::1],I_mode_Dalpha[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('P_ohm')
plt.ylabel('Dalpha')
#plt.ylim(0.0,100.)
plt.legend(loc=2)
plt.savefig('Dalpha_v_P_ohm.png')
plt.draw()
plt.close() 

plt.figure() 
plt.scatter(L_mode_P_ohm[::1],L_mode_q95[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_P_ohm[::1],H_mode_q95[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_P_ohm[::1],I_mode_q95[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('P_ohm')
plt.ylabel('q95')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('q95_v_Pohm.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(L_mode_p_rad_core[::1],L_mode_q95[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_p_rad_core[::1],H_mode_q95[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_p_rad_core[::1],I_mode_q95[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('p_rad_core')
plt.ylabel('q95')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('q95_v_p_rad_core.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(L_mode_e_bot_mks[::1],L_mode_q95[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_e_bot_mks[::1],H_mode_q95[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_e_bot_mks[::1],I_mode_q95[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('e_bot_mks')
plt.ylabel('q95')
#plt.xlim(0.0,30.)
plt.legend(loc=2)
plt.savefig('q95_v_e_bot_mks.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(L_mode_e_bot_mks[::1],L_mode_P_ohm[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_e_bot_mks[::1],H_mode_P_ohm[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_e_bot_mks[::1],I_mode_P_ohm[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('e_bot_mks')
plt.ylabel('P_ohm')
#plt.xlim(0.0,30.)
plt.legend(loc=2)
plt.savefig('P_ohm_v_e_bot_mks.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(L_mode_e_bot_mks[::1],L_mode_beta_p[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_e_bot_mks[::1],H_mode_beta_p[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_e_bot_mks[::1],I_mode_beta_p[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('e_bot_mks')
plt.ylabel('beta_p')
#plt.xlim(0.0,30.)
plt.legend(loc=2)
plt.savefig('beta_p_v_e_bot_mk.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(L_mode_q95[::1],L_mode_Dalpha[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_q95[::1],H_mode_Dalpha[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_q95[::1],I_mode_Dalpha[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('q95')
plt.ylabel('Dalpha')
#plt.ylim(0.0,100.)
plt.legend(loc=2)
plt.savefig('Dalpha_v_q95.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(L_mode_nLave_04[::1],L_mode_p_rad[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_nLave_04[::1],H_mode_p_rad[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_nLave_04[::1],I_mode_p_rad[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('nLave_04')
plt.ylabel('p_rad')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('p_rad_v_nLave_04.png')
plt.draw()
plt.close()

plt.figure() 
plt.scatter(L_mode_nebar_efit[::1],L_mode_p_rad[::1],c='b',s=70, alpha=0.03,label='L')
plt.scatter(H_mode_nebar_efit[::1],H_mode_p_rad[::1],c='g',s=70, alpha=0.03,label='H') 
plt.scatter(I_mode_nebar_efit[::1],I_mode_p_rad[::1],c='r',s=70, alpha=0.03,label='I') 
plt.xlabel('nebar_efit')
plt.ylabel('p_rad')
#plt.xlim(0.0,0.7)
plt.legend(loc=2)
plt.savefig('p_rad_v_nebar_efit.png')
plt.draw()
plt.close()