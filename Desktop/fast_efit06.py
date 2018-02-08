# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 00:05:37 2018

@author: mathewsa

This code takes shot, tstart, dt, and tend as input, and will write to trees 
efit06 or efit07 if they are available based on the 129x129 grid with double 
precision up to 5000 steps. Thus (tend-tstart)/dt < 5000 should be satisfied.
"""

import MDSplus 
from MDSplus import *
import math
import os
import subprocess 



#function to run efit in mode 10
#efit_ is the tree (e.g. efit06 or efit07)
#shot is the shot number, tstart, tend, and dt are in milliseconds
def run_efit(efit_,shot,tstart,tend,dt):     
    filepath = "efit_input.txt"
    with open("{}".format(filepath), 'w') as file_handler: 
        number = 1 #number of burst (<20)
        iterations = 0
        file_handler.write('10\n') 
        file_handler.write('{}\n'.format(efit_)) 
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
 


efits = ['efit06','efit07']
path_shot = {} #dictionary, with keys as shot; values are paths
#currently only used for efit path, but can be generalized for all measurements
        
#the below checks if efit06 tree has been created for shot recently
#if not, it will create the pulse and then run function to run efit
#if efit06 occupied, then try with efit07
#if efit07 occupied, raise an error

def main(shot,tstart,tend,dt):
    while True:
        try:
            efit_ = efits[0]
            tree = MDSplus.Tree('{}'.format(efit_), shot)   
            date_created  = Data.execute("date_time($)",max(tree.getNodeWild('***').time_inserted))
    #  the time if statements below ensures the timebase in the tree is at least covering the range and dt specified        
            time = (tree.getNode('\efit_aeqdsk:time')).data()        
            if (round(1000.*time[0]) <= round(tstart)) and (round(1000.*time[-1]) \
            >= round(tend)) and (int(1000.*(time[1]-time[0])) <= int(dt)):            
                print('{} is done in {}'.format(shot,efit_))   
                path_shot['{}'.format(shot)] = efit_,date_created.split(' ')[0]
                break
    # the date if statement will allow overwrite if last updated in 2018
            else:  
#                if ((date_created.split(" ")[0].split("-")[2]) == '2018') and (((date_created.split(" ")[0].split("-"))[1]) == 'JAN'):  
#                    efit_ = efits[0]
#                    print('Writing {} to {}'.format(shot,efit_))
#                    path_shot['{}'.format(shot)] = efit_,date_created.split(' ')[0]
#                    tree = Tree('{}'.format(efit_),-1)
#                    tree.createPulse(shot)
#                    run_efit(efit_,shot,tstart,tend,dt)
#                    break
#                else:
#                    print('{} is occupied for {}'.format(efit_,shot))
#above flag for data is currently commented out as certain input dates have variable structure
                try:
                    efit_ = efits[1]
                    tree = MDSplus.Tree('{}'.format(efit_), shot)   
                    date_created  = Data.execute("date_time($)",max(tree.getNodeWild('***').time_inserted))
                    time = (tree.getNode('\efit_aeqdsk:time')).data()        
                    if (round(1000.*time[0]) <= round(tstart)) and (round(1000.*time[-1]) \
                    >= round(tend)) and (int(1000.*(time[1]-time[0])) <= int(dt)):                       
                        print('{} is done in {}'.format(shot,efit_))  
                        path_shot['{}'.format(shot)] = efit_,date_created.split(' ')[0]
                        break
                    else:
#                        if ((date_created.split(" ")[0].split("-"))[2]) == '2018':
#                            efit_ = efits[1]
#                            print('Writing {} to {}'.format(shot,efit_))  
#                            path_shot['{}'.format(shot)] = efit_,date_created.split(' ')[0]
#                            tree = Tree('{}'.format(efit_),-1)
#                            tree.createPulse(shot) 
#                            run_efit(efit_,shot,tstart,tend,dt)
#                            break
#                        else:
                        print('{} and {} are occupied for {}'.format(efits[0],efits[1],shot))
                        raise
                except (TreeFOPENR,TreeNODATA):
                    efit_ = efits[1]
                    print('Writing {} to {}'.format(shot,efit_))  
                    path_shot['{}'.format(shot)] = efit_,date_created.split(' ')[0]
                    tree = Tree('{}'.format(efit_),-1)
                    tree.createPulse(shot) 
                    run_efit(efit_,shot,tstart,tend,dt)
                    break
        except (TreeFOPENR,TreeNODATA):
            efit_ = efits[0]
            print('Writing {} to {}'.format(shot,efit_))
            tree = Tree('{}'.format(efit_),-1)
            tree.createPulse(shot)
            run_efit(efit_,shot,tstart,tend,dt)
            date_created  = Data.execute("date_time($)",max(tree.getNodeWild('***').time_inserted))
            path_shot['{}'.format(shot)] = efit_,date_created.split(' ')[0]
            break
    print path_shot 

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])