# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:47:06 2018

@author: mathewsa
"""

#making table for shots in cmod logbook

import MDSplus 
from MDSplus import *
import numpy as np   
import os
import sys
from os import getenv
from datetime import datetime
sys.path.append('/home/mathewsa/Desktop/')
import fast_efit06
from fast_efit06 import path_shot
sys.path.append('/home/mathewsa/Desktop/')
import data_acquire_MDSplus 

shots = [1100210009,1100204004] #input for shots in cmod logbook we are considering
#shot = 1100204004 #1140724005 #1160930033 #1140221015  
i = 0
while i < len(shots):
    shot = shots[i] 
    tree = Tree('cmod', shot)
    while True: #plasma current is used as threshold for beginning and end of time series
        try:
            magnetics = MDSplus.Tree('magnetics', shot)
            ip = magnetics.getNode('\ip').data()
            time_ip = magnetics.getNode('\ip').dim_of().data() 
            index_begin = np.min(np.where(np.abs(ip) > 100000.)[0])
            index_end = np.max(np.where(np.abs(ip) > 100000.)[0])
            start_time = time_ip[index_begin] #in seconds
            end_time = time_ip[index_end]
            timebase = np.arange(round(start_time,3),round(end_time,3),0.001) #using 1 millisecond constant interval
            tstart = 1000.*timebase[0] #501.0 #milliseconds
            dt = 1.0 #milliseconds
            tend = 1000.*timebase[-1] #800.0 #milliseconds     
            fast_efit06.main(shot,tstart,tend,dt) 
            data_acquire_MDSplus.main(shot,timebase,path_shot)
            break
        except TreeNODATA: 
            print("No values stored for ip") 
            print(shot)
            raise
        except:
            print("Unexpected error for ip")
            print(shot)
            raise       
    #can define an alternative timebase if desired, but using this
    #definition of > 100kA as start/end condition currently for table
    i = i + 1