# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:47:03 2018

@author: mathewsa

Be careful that I want to actually run efit and possibly overwrite trees efit06 or efit07
"""

import MDSplus 
from MDSplus import *
import math
import os
import subprocess 


shot = 1120824016
tstart = 30.0
dt = 1.0
tend = 1680.0
#function to run efit in mode 10
#efit_ is the tree (e.g. efit06 or efit07)
#shot is the shot number, tstart, tend, and dt are in milliseconds 
filepath = "efit_input.txt"
with open("{}".format(filepath), 'w') as file_handler: 
    number = 1 #number of burst (<20)
    iterations = 0
    tree = Tree('efit07',-1)
    tree.createPulse(shot)
    file_handler.write('10\n') 
    file_handler.write('efit07\n') 
    file_handler.write('\n') 
    file_handler.write("{}, -{}\n".format(shot,number)) #the -{#} appends {#} overlapping bursts
    ntimes = 5000 #number of slices (<301 if not fast_efitdd)           
    while tstart < tend: #the numbers of rows written here must match {#}  
        if (tend - tstart) < dt*ntimes:
            ntimes = int(math.ceil((tend - tstart)/dt)) + 1 #ensure end time is included
        else:
            pass
        iterations = iterations + 1
        file_handler.write("{},{},{}\n".format(tstart,dt,ntimes))
        tstart = tstart + ntimes*dt 
    if iterations != number:
        print('ERROR')
        raise
    file_handler.close() 
    os.system("/usr/local/cmod/codes/efit/bin/fast_efitd129d < efit_input.txt > efit_output.txt")  
#        os.remove("efit_input.txt")
#        os.remove("efit_output.txt")
    
#filepath1 = "efit_input1.txt"
#with open("{}".format(filepath1), 'w') as file_handler: 
#    number = 1 #number of burst (<20)
#    iterations = 0
#    tree = Tree('efit07',-1)
#    tree.createPulse(shot)
#    file_handler.write('10\n') 
#    file_handler.write('efit07\n') 
#    file_handler.write('\n') 
#    file_handler.write("{}, -{}\n".format(shot,number)) #the -{#} appends {#} overlapping bursts
#    ntimes = 200 #number of slices (<301 if not fast_efitdd)           
#    while tstart < tend: #the numbers of rows written here must match {#}  
#        if (tend - tstart) < dt*ntimes:
#            ntimes = int(math.ceil((tend - tstart)/dt)) + 1 #ensure end time is included
#        else:
#            pass
#        iterations = iterations + 1
#        file_handler.write("{},{},{}\n".format(tstart,dt,ntimes))
#        tstart = tstart + ntimes*dt 
#    if iterations != number:
#        print('ERROR')
#        raise
#    file_handler.close() 
#    os.system("/usr/local/cmod/codes/efit/bin/efitd129d < efit_input1.txt > efit_output1.txt")  
##        os.remove("efit_input.txt")
##        os.remove("efit_output.txt")
    
tree = MDSplus.Tree('efit06', shot)   
date_created_efit06  = Data.execute("date_time($)",max(tree.getNodeWild('***').time_inserted))
tree = MDSplus.Tree('efit07', shot)   
date_created_efit07  = Data.execute("date_time($)",max(tree.getNodeWild('***').time_inserted))