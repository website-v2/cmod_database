#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:57:46 2018

@author: Abhilash
"""

"""Read columns from a text file"""

file = "/Users/Abhilash/Desktop/test.txt"
f=open(file,"r")
lines=f.readlines()
column_time_ip =[]
column_ip=[]
column_time_btor=[]
column_btor=[]
for x in lines:
    column_time_ip.append(x.split(' ')[0])
    column_ip.append(x.split(' ')[1])
    column_time_btor.append(x.split(' ')[2])
    column_btor.append(x.split(' ')[3])
f.close()

#can now simply read these columns by index/time slice into my populated table