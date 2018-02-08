# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:45:03 2018

@author: mathewsa

This code acquires all relevant data from MDSplus trees and saves select variables
as arrays. The correct path for the efit tree for that shot (i.e. efit06 or efit07
currently) must be passed. The desired timebase for interpolation must also be
provided as all the data is linearly interpolated over this particular timebase. 
If data is absent during segment(s) of the desired timebase, then dependent on the
particular variable, either 0 or NaN is inserted as a placeholder. Units for all
quantities of interest are stored in word document for the generated table.
"""

import MDSplus 
from MDSplus import *
import numpy as np   
import os
import sys
from os import getenv 
import idlpy
from idlpy import *

def smooth(y,box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y,box,mode='same')
    return y_smooth

def main(shot,timebase,path_shot):
    tree = Tree('cmod', shot)
     
    NaN = (np.empty(len(timebase)) * np.nan)
    zeros = np.zeros(len(timebase))
    ones = np.ones(len(timebase))
    
    while True:
        try:
            magnetics = MDSplus.Tree('magnetics', shot)
            ip = magnetics.getNode('\ip').data()
            time_ip = magnetics.getNode('\ip').dim_of().data() 
            ip = np.interp(timebase,time_ip,ip,left=np.nan,right=np.nan)
            btor=tree.getNode('\\BTOR') 
            btor_data=btor.record 
            Y = btor_data.data()
            time_btor = btor.getNode('\BTOR').dim_of().data()
            btor = Y
            btor = np.interp(timebase,time_btor,btor,left=np.nan,right=np.nan)
            break
        except TreeNODATA:
            btor = NaN
            time_btor = timebase
            print("No values stored for btor") 
            print(shot)
            break
        except:
            print("Unexpected error for btor")
            print(shot)
            raise
    
    
    while True:
        try:
            p_lh = MDSplus.Tree('LH', shot)
            time_p_lh = (p_lh.getNode('\LH::TOP.RESULTS:NETPOW')).dim_of().data() #[s]
            p_lh = 1000.*(p_lh.getNode('\LH::TOP.RESULTS:NETPOW')).data() #[W]
            p_lh = np.interp(timebase,time_p_lh,p_lh,left=np.nan,right=np.nan)
	    break
        except (TreeNODATA,TreeFOPENR): #not available if LH turned off
            p_lh = zeros
            time_p_lh = timebase
            print("No values stored for p_lh") 
            print(shot)
            break
        except:
	    if p_lh == 0.0:
                p_lh = zeros
                time_p_lh = timebase
		print("p_lh was just element 0")
		break
	    else:
                print("Unexpected error for p_lh")
                print(shot)
                raise 
    
    while True:
        try:
            p_icrf_tree = MDSplus.Tree('rf', shot)
            time_p_icrf = (p_icrf_tree.getNode('\RF::RF_POWER_NET')).dim_of().data()
            p_icrf = 1000000.*(p_icrf_tree.getNode('\RF::RF_POWER_NET')).data(); #[W] net icrf power
            p_icrf = np.interp(timebase,time_p_icrf,p_icrf,left=np.nan,right=np.nan)
            time_p_icrf_d = (p_icrf_tree.getNode('\RF::RF_POWER_D')).dim_of().data()
            p_icrf_d = 1000000.*(p_icrf_tree.getNode('\RF::RF_POWER_D')).data()
            p_icrf_d = np.interp(timebase,time_p_icrf_d,p_icrf_d,left=np.nan,right=np.nan)
            freq_icrf_d = (p_icrf_tree.getNode('\RF::TOP.ANTENNA.DATA.D_PORT:FREQ')).data()
            freq_icrf_d = 1000000.*ones*freq_icrf_d
            time_p_icrf_e = (p_icrf_tree.getNode('\RF::RF_POWER_E')).dim_of().data()
            p_icrf_e = 1000000.*(p_icrf_tree.getNode('\RF::RF_POWER_E')).data()
            p_icrf_e = np.interp(timebase,time_p_icrf_e,p_icrf_e,left=np.nan,right=np.nan)
            freq_icrf_e = (p_icrf_tree.getNode('\RF::TOP.ANTENNA.DATA.E_PORT:FREQ')).data()
            freq_icrf_e = 1000000.*ones*freq_icrf_e
            time_p_icrf_j3 = (p_icrf_tree.getNode('\RF::RF_POWER_J3')).dim_of().data()
            p_icrf_j3 = 1000000.*(p_icrf_tree.getNode('\RF::RF_POWER_J3')).data()
            p_icrf_j3 = np.interp(timebase,time_p_icrf_j3,p_icrf_j3,left=np.nan,right=np.nan)
            freq_icrf_j = (p_icrf_tree.getNode('\RF::TOP.ANTENNA.DATA.J_PORT:FREQ')).data() #for J3 and J4
            freq_icrf_j = 1000000.*ones*freq_icrf_j #J3 and J4 are equal
            time_p_icrf_j4 = (p_icrf_tree.getNode('\RF::RF_POWER_J4')).dim_of().data()
            p_icrf_j4 = 1000000.*(p_icrf_tree.getNode('\RF::RF_POWER_J4')).data()
            p_icrf_j4 = np.interp(timebase,time_p_icrf_j4,p_icrf_j4,left=np.nan,right=np.nan)
            break
        except TreeNODATA:
            p_icrf = p_icrf_d = p_icrf_e = p_icrf_j3 = p_icrf_j4 = NaN
            time_p_icrf = time_p_icrf_d = time_p_icrf_e = time_p_icrf_j3 = time_p_icrf_j4 = timebase
            print("No values stored for p_icrf") 
            print(shot)
            break
        except:
            print("Unexpected error for p_icrf")
            print(shot)
            raise


    while True:
        try:  
            path_shot = path_shot['{}'.format(shot)][0]
            efit = MDSplus.Tree('{}'.format(path_shot), shot)
            IDL.run("good_time_slices = efit_check(shot={},tree='{}')".format(shot,path_shot)) 
            good_time_slices = getattr(IDL,"good_time_slices") #returns indexes for values passing efit_check test
            time_efit = (efit.getNode('\efit_aeqdsk:time')).data()[good_time_slices]
            beta_N = (efit.getNode('\efit_aeqdsk:betan')).data()[good_time_slices]; #normalized beta
            beta_N = np.interp(timebase,time_efit,beta_N,left=np.nan,right=np.nan)
            beta_p = (efit.getNode('\efit_aeqdsk:betap')).data()[good_time_slices]; #beta_poloidal
            beta_p = np.interp(timebase,time_efit,beta_p,left=np.nan,right=np.nan)
            beta_t = (efit.getNode('\efit_aeqdsk:betat')).data()[good_time_slices]; #beta_toroidal
            beta_t = np.interp(timebase,time_efit,beta_t,left=np.nan,right=np.nan)
            kappa = (efit.getNode('\efit_aeqdsk:eout')).data()[good_time_slices]; #elongation - vertical
            kappa = np.interp(timebase,time_efit,kappa,left=np.nan,right=np.nan)
            triang_l = (efit.getNode('\efit_aeqdsk:doutl')).data()[good_time_slices]; #lower triangularity of lcfs
            triang_l = np.interp(timebase,time_efit,triang_l,left=np.nan,right=np.nan)
            triang_u = (efit.getNode('\efit_aeqdsk:doutu')).data()[good_time_slices]; #upper triangularity of lcfs
            triang_u = np.interp(timebase,time_efit,triang_u,left=np.nan,right=np.nan)
            triang = (triang_u + triang_l)/2. #overall triangularity - horizontal (major radius)
            li = (efit.getNode('\efit_aeqdsk:li')).data()[good_time_slices]; #internal inductance
            li = np.interp(timebase,time_efit,li,left=np.nan,right=np.nan)
            areao = (efit.getNode('\efit_aeqdsk:areao')).data()[good_time_slices]/(100.*100.); #area of lcfs
            areao = np.interp(timebase,time_efit,areao,left=np.nan,right=np.nan)
            vout = (efit.getNode('\efit_aeqdsk:vout')).data()[good_time_slices]/(100.*100.*100.); #volume of lcfs
            vout = np.interp(timebase,time_efit,vout,left=np.nan,right=np.nan)
            aout = (efit.getNode('\efit_aeqdsk:aout')).data()[good_time_slices]/(100.); #minor radius of lcfs
            aout = np.interp(timebase,time_efit,aout,left=np.nan,right=np.nan)
            rout = (efit.getNode('\efit_aeqdsk:rout')).data()[good_time_slices]/(100.); #major radius of geometric center
            rout = np.interp(timebase,time_efit,rout,left=np.nan,right=np.nan)
            zout = (efit.getNode('\efit_aeqdsk:zout')).data()[good_time_slices]/(100.); #z of lcfs (constructed)
            zout = np.interp(timebase,time_efit,zout,left=np.nan,right=np.nan)
            rmag = (efit.getNode('\efit_aeqdsk:rmagx')).data()[good_time_slices]/(100.)
            rmag = np.interp(timebase,time_efit,rmag,left=np.nan,right=np.nan)
            zmag = (efit.getNode('\efit_aeqdsk:zmagx')).data()[good_time_slices]/(100.); #z of magnetic axis
            zmag = np.interp(timebase,time_efit,zmag,left=np.nan,right=np.nan)
            rseps = (efit.getNode('\efit_aeqdsk:rseps')).data()[good_time_slices]; #r of upper and lower xpts
            rsep_lower = rseps[:,0]/(100.)
            rsep_upper = rseps[:,1]/(100.)   
            rsep_lower = np.interp(timebase,time_efit,rsep_lower,left=np.nan,right=np.nan)
            rsep_upper = np.interp(timebase,time_efit,rsep_upper,left=np.nan,right=np.nan)
            zseps = (efit.getNode('\efit_aeqdsk:zseps')).data()[good_time_slices]; #z of upper and lower xpts
            zsep_lower = zseps[:,0]/(100.)
            zsep_upper = zseps[:,1] /(100.) 
            zsep_lower = np.interp(timebase,time_efit,zsep_lower,left=np.nan,right=np.nan)
            zsep_upper = np.interp(timebase,time_efit,zsep_upper,left=np.nan,right=np.nan)
            rvsin = (efit.getNode('\efit_aeqdsk:rvsin')).data()[good_time_slices]/(100.); #r of inner strike point
            rvsin = np.interp(timebase,time_efit,rvsin,left=np.nan,right=np.nan)
            zvsin = (efit.getNode('\efit_aeqdsk:zvsin')).data()[good_time_slices]/(100.); #z of inner strike point
            zvsin = np.interp(timebase,time_efit,zvsin,left=np.nan,right=np.nan)
            rvsout = (efit.getNode('\efit_aeqdsk:rvsout')).data()[good_time_slices]/(100.); #r of outer strike point
            rvsout = np.interp(timebase,time_efit,rvsout,left=np.nan,right=np.nan)
            zvsout = (efit.getNode('\efit_aeqdsk:zvsout')).data()[good_time_slices]/(100.); #z of outer strike point
            zvsout = np.interp(timebase,time_efit,zvsout,left=np.nan,right=np.nan)
            upper_gap = (efit.getNode('\efit_aeqdsk:otop')).data()[good_time_slices]/100.; # meters
            upper_gap = np.interp(timebase,time_efit,upper_gap,left=np.nan,right=np.nan)
            lower_gap = (efit.getNode('\efit_aeqdsk:obott')).data()[good_time_slices]/100.; # meters
            lower_gap = np.interp(timebase,time_efit,lower_gap,left=np.nan,right=np.nan)
            q0 = (efit.getNode('\efit_aeqdsk:q0')).data()[good_time_slices]; #safety factor at center
            q0 = np.interp(timebase,time_efit,q0,left=np.nan,right=np.nan)
            qstar = (efit.getNode('\efit_aeqdsk:qstar')).data()[good_time_slices]; #cylindrical safety factor
            qstar = np.interp(timebase,time_efit,qstar,left=np.nan,right=np.nan)
            q95 = (efit.getNode('\efit_aeqdsk:q95')).data()[good_time_slices]; #edge safety factor
            q95 = np.interp(timebase,time_efit,q95,left=np.nan,right=np.nan)
            qout = (efit.getNode('\efit_aeqdsk:qout')).data()[good_time_slices]
            qout = np.interp(timebase,time_efit,qout,left=np.nan,right=np.nan)
            cpasma = (efit.getNode('\efit_aeqdsk:cpasma')).data()[good_time_slices] #calculated plasma current
            cpasma = np.interp(timebase,time_efit,cpasma,left=np.nan,right=np.nan)
            BtVac = (efit.getNode('\efit_aeqdsk:btaxv')).data()[good_time_slices] #on-axis plasma toroidal field 
            BtVac = np.interp(timebase,time_efit,BtVac,left=np.nan,right=np.nan)
            BtPlasma = (efit.getNode('\efit_aeqdsk:btaxp')).data()[good_time_slices] #on-axis plasma toroidal field
            BtPlasma = np.interp(timebase,time_efit,BtPlasma,left=np.nan,right=np.nan)
            BpAvg = (efit.getNode('\efit_aeqdsk:bpolav')).data()[good_time_slices] #average poloidal field
            BpAvg = np.interp(timebase,time_efit,BpAvg,left=np.nan,right=np.nan)
            V_loop_efit = (efit.getNode('\efit_aeqdsk:vloopt')).data()[good_time_slices]; #loop voltage
            V_loop_efit = np.interp(timebase,time_efit,V_loop_efit,left=np.nan,right=np.nan)
            V_surf_efit = (efit.getNode('\efit_aeqdsk:vsurfa')).data()[good_time_slices]; #surface voltage
            V_surf_efit = np.interp(timebase,time_efit,V_surf_efit,left=np.nan,right=np.nan)
            Wmhd = (efit.getNode('\efit_aeqdsk:wplasm')).data()[good_time_slices]; #diamagnetic/stored energy, [J]
            Wmhd = np.interp(timebase,time_efit,Wmhd,left=np.nan,right=np.nan)
            ssep = (efit.getNode('\efit_aeqdsk:ssep')).data()[good_time_slices]/100.; # distance on midplane between 1st and 2nd separatrices [m]
            ssep = np.interp(timebase,time_efit,ssep,left=np.nan,right=np.nan)
            n_over_ncrit = (efit.getNode('\efit_aeqdsk:xnnc')).data()[good_time_slices]; #vertical stability criterion (EFIT name: xnnc)
            n_over_ncrit = np.interp(timebase,time_efit,n_over_ncrit,left=np.nan,right=np.nan)  
            inductance = 4.*np.pi*1.E-7 * 0.68 * li/2.; # For simplicity, we use R0 = 0.68 m, but we could use \efit_aeqdsk:rmagx
            dipdt = np.gradient(ip,timebase)
            dipdt_smoothed = smooth(dipdt,11) #11-point smoothing (moving average box/by convolution)
            V_inductive = inductance*dipdt_smoothed
            V_resistive = V_loop_efit - V_inductive
            P_ohm = ip*V_resistive
            
            pcurrt = (efit.getNode('\efit_g_eqdsk:pcurrt')).data() #current density, Jp
            refit = (efit.getNode('\efit_g_eqdsk:pcurrt')).dim_of().data()  
            Bcentre = (efit.getNode('\efit_a_eqdsk:BCentr')).data()
            Rcentre = (efit.getNode('\efit_a_eqdsk:RCENCM')).data() #radial position where Bcent is calculated
            rGrid = (efit.getNode('\efit_g_eqdsk:rGrid')).data()
            zGrid = (efit.getNode('\efit_g_eqdsk:zGrid')).data()
            psiRZ = (efit.getNode('\efit_g_eqdsk:psiRZ')).data() #psi not normalized
            psiAxis = (efit.getNode('\efit_aeqdsk:simagx')).data() #psi on magnetic axis
            psiLCFS = (efit.getNode('\efit_aeqdsk:sibdry')).data() #psi at separatrix
            rhovn = ((efit.getNode('\efit_g_eqdsk:rhovn')).data()) #normalized sqrt(volume) mapped to normalized poloidal flux [second index is time slice]
            chord_v_len = (efit.getNode('\efit_aeqdsk:rco2v')).data() #vertical chord lengths for CO2 (TCI)
            i = 0  
            chord_4_v_len = []
            while i < len(chord_v_len):
                chord_4_v_len.append(chord_v_len[i][3])
                i = i + 1  
            chord_4_v_len = (np.array(chord_4_v_len))[good_time_slices]/100. #retrieves length of chord 4 in metres for all time slices
            chord_4_v_len = np.interp(timebase,time_efit,chord_4_v_len,left=np.nan,right=np.nan)  
            efit_rmid = (efit.getNode('\efit_fitout:rpres')).data() #maximum major radius of each flux surface
            volp = (efit.getNode('\efit_fitout:volp')).data() #array of volume within flux surface
            fpol = (efit.getNode('\efit_g_eqdsk:fpol')).data() #should be multipled by -1*sign(current)
            pres_flux = (efit.getNode('\efit_g_eqdsk:pres')).data() #array of pressure on flux surface psi
            ffprime = (efit.getNode('\efit_g_eqdsk:ffprim')).data()
            pprime = (efit.getNode('\efit_g_eqdsk:pprime')).data() #plasma pressure gradient as function of psi        
            qpsi = (efit.getNode('\efit_g_eqdsk:qpsi')).data() #array of q, safety factor, on flux surface psi       
            rlcfs = (efit.getNode('\efit_g_eqdsk:rbbbs')).data()
            zlcfs = (efit.getNode('\efit_g_eqdsk:zbbbs')).data() 
            break
        except TreeNODATA:
            time_efit = timebase
            beta_N = beta_p = beta_t = kappa = triang_l = triang_u =\
            triang = li = areao = vout = aout = rout = zmag =\
            zout = zseps = zvsin = zvsout = upper_gap = lower_gap =\
            q0 = qstar = q95 = qout = BtVac = BtPlasma = BpAvg = V_loop_efit =\
            V_surf_efit = Wmhd = ssep = n_over_ncrit = P_ohm = chord_4_v_len = NaN
            print("No values stored for efit") 
            print(shot)
            break
        except:
            print("Unexpected error for efit")
            print(shot)
            raise
    
    
    while True:
        try:
            electrons = MDSplus.Tree('ELECTRONS', shot) #NEW CORE only valid for SHOT>1020000000
            time_core = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_RZ')).dim_of().data()
            dens_core = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_RZ')).data(); #density (m^-3)
            dens_core_err = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_ERR')).data(); #error (m^-3)
            temp_core = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:TE_RZ')).data()*1000.; #temperature (eV)
            temp_core_err = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:TE_ERR')).data()*1000.; #error (eV)
            midR_core = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:R_MID_T')).data(); #mapped midplane R (m) coreTS
            z_core = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:Z_SORTED')).data(); # z-position (m)
            R_core = (electrons.getNode('\ELECTRONS::TOP.YAG.RESULTS.PARAM:R')).data(); # v R-position (m)
            dens_core_ = []
            temp_core_ = []
            i = 0
            while i < len(z_core):
                dens_core_.append(np.interp(timebase,time_core,dens_core[i,:],left=np.nan,right=np.nan))
                temp_core_.append(np.interp(timebase,time_core,temp_core[i,:],left=np.nan,right=np.nan))
                i = i + 1
            break
        except TreeNODATA:
            dens_core_ = dens_core_err = temp_core_ = temp_core_err =\
            midR_core = z_core = R_core = NaN   
            print("No values stored for core") 
            print(shot)
            break
        except:
            print("Unexpected error for core")
            print(shot)
            raise
    
    
    while True:
        try:
            electrons = MDSplus.Tree('ELECTRONS', shot) #NEW EDGE only valid for SHOT>1000000000
            time_edge = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE')).dim_of().data()
            dens_edge = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE')).data(); #density (m^-3)
            dens_edge_err = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE:ERROR')).data(); #error (m^-3)
            temp_edge = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:TE')).data(); #temperature (eV)
            temp_edge_err = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:TE:ERROR')).data(); #error (eV)
            midR_edge = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:RMID')).data(); #mapped midplane R (m) coreTS
            z_edge = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.DATA:FIBER_Z')).data(); # z-position (m)
            z_midplane = (electrons.getNode('\electrons::top.yag_edgets.data.fiber_z:holder_z')).data() #edge TS fibers are mounted in 
            #a movable block, and can be shifted vertically to position the scattering volumes as desired relative to the LCFS. Edge fiber position is recorded as a measurement in mm below the midplane
            R_edge = (electrons.getNode('\ELECTRONS::TOP.YAG.RESULTS.PARAM:R')).data(); # v R-position (m)
            dens_edge_ = []
            temp_edge_ = []
            i = 0
            while i < len(z_edge):
                dens_edge_.append(np.interp(timebase,time_edge,dens_edge[i,:],left=np.nan,right=np.nan))
                temp_edge_.append(np.interp(timebase,time_edge,temp_edge[i,:],left=np.nan,right=np.nan))
                i = i + 1        
            break
        except TreeNODATA:
            dens_edge_ = dens_edge_err = temp_edge_ = temp_edge_err =\
            NaN 
            print("No values stored for edge") 
            print(shot)
            break
        except:
            print("Unexpected error for edge")
            print(shot)
            raise
    
    
    while True:
        try:
            spectroscopy = MDSplus.Tree('SPECTROSCOPY', shot)
            HoverHD = (spectroscopy.getNode('\SPECTROSCOPY::BALMER_H_TO_D')).data(); #H/(H+D)
            time_HoverHD = spectroscopy.getNode('\SPECTROSCOPY::BALMER_H_TO_D').dim_of().data()
            HoverHD = np.interp(timebase,time_HoverHD,HoverHD,left=np.nan,right=np.nan)
            Halpha = (spectroscopy.getNode('\SPECTROSCOPY::HA_2_BRIGHT')).data(); #H-Alpha at H Port
            time_Halpha = (spectroscopy.getNode('\SPECTROSCOPY::HA_2_BRIGHT')).dim_of().data();
            Halpha = np.interp(timebase,time_Halpha,Halpha,left=np.nan,right=np.nan)
            Dalpha = (spectroscopy.getNode('\SPECTROSCOPY::TOP.VUV.VIS_SIGNALS:MCP_VIS_SIG1')).data() # D-alpha (W/m^2/st)
            time_Dalpha = (spectroscopy.getNode('\SPECTROSCOPY::TOP.VUV.VIS_SIGNALS:MCP_VIS_SIG1')).dim_of().data()
            Dalpha = np.interp(timebase,time_Dalpha,Dalpha,left=np.nan,right=np.nan)
            z_ave = (spectroscopy.getNode('\SPECTROSCOPY::z_ave')).data() 
            time_z_ave = (spectroscopy.getNode('\SPECTROSCOPY::z_ave')).dim_of().data() 
            z_ave = np.interp(timebase,time_z_ave,z_ave,left=np.nan,right=np.nan)
            p_rad_tree = MDSplus.Tree('SPECTROSCOPY', shot) #[W]
            time_p_rad = (p_rad_tree.getNode('\TWOPI_FOIL')).dim_of().data()
            p_rad = (p_rad_tree.getNode('\TWOPI_FOIL')).data()
            p_rad = np.interp(timebase,time_p_rad,p_rad,left=np.nan,right=np.nan)
            time_p_rad_core = (p_rad_tree.getNode('\\top.bolometer.results.foil:main_power')).dim_of().data()
            p_rad_core = (p_rad_tree.getNode('\\top.bolometer.results.foil:main_power')).data()*1000000.
            p_rad_core = np.interp(timebase,time_p_rad_core,p_rad_core,left=np.nan,right=np.nan)
    #use twopi_diode instead as in Granetz code if avoiding non-causal filtering
    #rad_fraction = p_rad/p_input (if p_input==0 then NaN/0)
            break
        except TreeNODATA:
            HoverHD = Halpha = Dalpha = z_ave = p_rad = p_rad_core = NaN
            time_HoverHD = time_Halpha = time_Dalpha = time_z_ave = time_p_rad = time_p_rad_core = timebase    
            print("No values stored for spectroscopy") 
            print(shot)
            break
        except:
            print("Unexpected error for spectroscopy")
            print(shot)
            raise
    
    
#    while True:
#        try:
#            cxrs = MDSplus.Tree('DNB', shot)
#           Vpol = 1000.*((cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.POLOIDAL:VEL')).data()) #poloidal velocity [m/s]
#            time_Vpol = (cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.POLOIDAL:VEL')).dim_of().data()
#            dVpol = 1000.*((cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.POLOIDAL:VEL_SIGMA')).data()) #poloidal velocity sigma [m/s]
#            Vpol = np.interp(timebase,time_Vpol,Vpol,left=np.nan,right=np.nan)
#            Vtor = 1000.*((cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.TOR_OUT:VEL')).data()) #toroidal velocity [m/s]$
#           time_Vtor = (cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.TOR_OUT:VEL')).dim_of().data() 
#            dVtor = 1000.*((cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.TOR_OUT:VEL_SIGMA')).data()) #toroidal velocity sigma [m/s]
#            Vtor = np.interp(timebase,time_Vtor,Vtor,left=np.nan,right=np.nan)
#            break
#        except TreeNODATA:
#            Vpol = dVpol = Vtor = dVtor = NaN
#            time_Vpol = time_Vtor = timebase       
#            print("No values stored for cxrs") 
#            print(shot)
#            break
#        except:
#            print("Unexpected error for cxrs")
#            print(shot)
#            raise
    
    while True:
        try:
            TCI = MDSplus.Tree('ELECTRONS', shot)
            NL_01 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_01')).data()
            NL_02 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_02')).data()
            NL_03 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_03')).data()
            NL_04 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')).data() #Units are meters^(-2)
            NL_05 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_05')).data()
            NL_06 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_06')).data()
            NL_07 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_07')).data()
            NL_08 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_08')).data()
            NL_09 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_09')).data()
            NL_10 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_10')).data()
            r_NL = (TCI.getNode('\electrons::top.tci.results:rad')).data() #major radius of each of 10 chords in metres
            time_NL_04 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')).dim_of().data() #same time for each chord
            NL_04 = np.interp(timebase,time_NL_04,NL_04,left=np.nan,right=np.nan)
            nLave_04 = NL_04/chord_4_v_len   
            nebar_efit = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS.INVERSION:NEBAR_EFIT')).data()
            time_nebar_efit = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS.INVERSION:NEBAR_EFIT')).dim_of().data() #NeBar_EFIT (TCI)
            nebar_efit = np.interp(timebase,time_nebar_efit,nebar_efit,left=np.nan,right=np.nan)
            break
        except TreeNODATA:
            NL_01 = NL_02 = NL_03 = NL_04 = NL_05 = NL_06 = NL_07 =\
            NL_08 = NL_09 = NL_10 = nLave_04 = nebar_efit = NaN
            time_NL_04 = time_nebar_efit = timebase  
            print("No values stored for tci") 
            print(shot)
            break
        except:
            print("Unexpected error for tci")
            print(shot)
            raise
            
            
#    while True:
#        try:
#            refl = MDSplus.Tree('ELECTRONS', shot)
#            refl_50GHz_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_01')).data()
#            time_refl_50GHz_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_01')).dim_of().data()
#            refl_50GHz_cos = np.interp(timebase,time_refl_50GHz_cos,refl_50GHz_cos,left=np.nan,right=np.nan)             
#            refl_50GHz_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_02')).data()
#            time_refl_50GHz_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_02')).dim_of().data()
#            refl_50GHz_sin = np.interp(timebase,time_refl_50GHz_sin,refl_50GHz_sin,left=np.nan,right=np.nan)             
#            refl_60GHz_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_03')).data()
#            time_refl_60GHz_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_03')).dim_of().data()
#            refl_60GHz_cos = np.interp(timebase,time_refl_60GHz_cos,refl_60GHz_cos,left=np.nan,right=np.nan)              
#            refl_60GHz_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_04')).data()
#            time_refl_60GHz_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_04')).dim_of().data()
#            refl_60GHz_sin = np.interp(timebase,time_refl_60GHz_sin,refl_60GHz_sin,left=np.nan,right=np.nan)              
#            refl_75GHz_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_05')).data()
#            time_refl_75GHz_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_05')).dim_of().data()
#            refl_75GHz_cos = np.interp(timebase,time_refl_75GHz_cos,refl_75GHz_cos,left=np.nan,right=np.nan)             
#            refl_75GHz_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_06')).data()
#            time_refl_75GHz_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_06')).dim_of().data()
#            refl_75GHz_sin = np.interp(timebase,time_refl_75GHz_sin,refl_75GHz_sin,left=np.nan,right=np.nan)            
#            refl_88GHz_lower_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_07')).data()
#            time_refl_88GHz_lower_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_07')).dim_of().data()
#            refl_88GHz_lower_cos = np.interp(timebase,time_refl_88GHz_lower_cos,refl_88GHz_lower_cos,left=np.nan,right=np.nan)             
#            refl_88GHz_lower_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_08')).data()
#            time_refl_88GHz_lower_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_08')).dim_of().data()
#            refl_88GHz_lower_sin = np.interp(timebase,time_refl_88GHz_lower_sin,refl_88GHz_lower_sin,left=np.nan,right=np.nan)            
#            refl_88GHz_upper_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_09')).data()
#            time_refl_88GHz_upper_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_09')).dim_of().data()
#            refl_88GHz_upper_cos = np.interp(timebase,time_refl_88GHz_upper_cos,refl_88GHz_upper_cos,left=np.nan,right=np.nan)             
#            refl_88GHz_upper_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_10')).data()
#            time_refl_88GHz_upper_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_10')).dim_of().data()
#            refl_88GHz_upper_sin = np.interp(timebase,time_refl_88GHz_upper_sin,refl_88GHz_upper_sin,left=np.nan,right=np.nan)             
#            refl_110GHz_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_11')).data()
#            time_refl_110GHz_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_11')).dim_of().data()
#            refl_110GHz_cos = np.interp(timebase,time_refl_110GHz_cos,refl_110GHz_cos,left=np.nan,right=np.nan)            
#            refl_110GHz_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_12')).data()
#            time_refl_110GHz_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_12')).dim_of().data()
#            refl_110GHz_sin = np.interp(timebase,time_refl_110GHz_sin,refl_110GHz_sin,left=np.nan,right=np.nan)               
#            refl_132GHz_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_13')).data()
#            time_refl_132GHz_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_13')).dim_of().data()
#            refl_132GHz_cos = np.interp(timebase,time_refl_132GHz_cos,refl_132GHz_cos,left=np.nan,right=np.nan)             
#            refl_132GHz_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_14')).data()
#            time_refl_132GHz_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_14')).dim_of().data()
#            refl_132GHz_sin = np.interp(timebase,time_refl_132GHz_sin,refl_132GHz_sin,left=np.nan,right=np.nan)              
#            refl_140GHz_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_15')).data()
#            time_refl_140GHz_cos = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_15')).dim_of().data()
#            refl_140GHz_cos = np.interp(timebase,time_refl_140GHz_cos,refl_140GHz_cos,left=np.nan,right=np.nan)             
#            refl_140GHz_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_16')).data()
#            time_refl_140GHz_sin = (refl.getNode('\electrons::top.reflect.cpci:dt132_1:input_16')).dim_of().data()
#            refl_140GHz_sin = np.interp(timebase,time_refl_140GHz_sin,refl_140GHz_sin,left=np.nan,right=np.nan)            
#            refl_50GHz = refl_50GHz_cos + refl_50GHz_sin
#            refl_60GHz = refl_60GHz_cos + refl_60GHz_sin
#            refl_75GHz = refl_75GHz_cos + refl_75GHz_sin
#            refl_88GHz_low = refl_88GHz_lower_cos + refl_88GHz_lower_sin
#            refl_88GHz_hi = refl_88GHz_upper_cos + refl_88GHz_upper_sin
#            refl_110GHz = refl_110GHz_cos + refl_110GHz_sin
#            refl_132GHz = refl_132GHz_cos + refl_132GHz_sin
#            refl_140GHz = refl_140GHz_cos + refl_140GHz_sin
#            
##            var_freq_refl = (refl.getNode('\electrons::TOP:REFLECT.results:var_freq')).data()
##            freq_refl = (refl.getNode('\ELECTRONS::TOP.REFLECT.RESULTS:FREQ')).data() 
##            freq_base_refl = (refl.getNode('\ELECTRONS::TOP.REFLECT.RESULTS:FREQ_BASE')).data() #?
##            dens_refl = (refl.getNode('\ELECTRONS::TOP.REFLECT.RESULTS:DENSITY')).data() #?
##            n95_refl = (refl.getNode('\ELECTRONS::TOP.REFLECT.RESULTS:N95')).data() #?
##            nlcfs_refl = (refl.getNode('\ELECTRONS::TOP.REFLECT.RESULTS:NLCFS')).data() #? 
##            rad_refl = (refl.getNode('\ELECTRONS::TOP.REFLECT.RESULTS:RAD')).data() #?
##            refl110 = (refl.getNode('\ELECTRONS::TOP.REFLECT.RESULTS:OPDIST110')).data() #?
##            refl50 = (refl.getNode('\ELECTRONS::TOP.REFLECT.RESULTS:OPDIST50')).data() #?
##            refl60 = (refl.getNode('\ELECTRONS::TOP.REFLECT.RESULTS:OPDIST60')).data() #? 
##            refl75 = (refl.getNode('\ELECTRONS::TOP.REFLECT.RESULTS:OPDIST75')).data() #?
##            refl88 = (refl.getNode('\ELECTRONS::TOP.REFLECT.RESULTS:OPDIST88')).data() #?
##            time_refl = (refl.getNode('\ELECTRONS::TOP.REFLECT.RESULTS:TIMEBASE')).data() #?
#            #WHY NO DATA AVAILABLE FOR ABOVE?
#            #PROFILE NOT OPENING...? 
#            break
#        except TreeNODATA: 
#            refl_50GHz_cos = refl_50GHz_sin = refl_60GHz_cos = refl_60GHz_sin =\
#            refl_75GHz_cos = refl_75GHz_sin = refl_88GHz_lower_cos = refl_88GHz_lower_sin =\
#            refl_88GHz_upper_cos = refl_88GHz_upper_sin = refl_110GHz_cos = refl_110GHz_sin =\
#            refl_132GHz_cos = refl_132GHz_sin = refl_140GHz_cos = refl_140GHz_sin =\
#            refl_50GHz = refl_60GHz = refl_75GHz = refl_88GHz_low =\
#            refl_88GHz_hi = refl_110GHz = refl_132GHz = refl_140GHz = NaN
#            time_refl_50GHz_cos = time_refl_50GHz_sin = time_refl_60GHz_cos =\
#            time_refl_60GHz_sin = time_refl_75GHz_cos = time_refl_75GHz_sin =\
#            time_refl_88GHz_lower_cos = time_refl_88GHz_lower_sin =\
#            time_refl_88GHz_upper_cos = time_refl_88GHz_upper_sin =\
#            time_refl_110GHz_cos = time_refl_110GHz_sin = time_refl_132GHz_cos =\
#            time_refl_132GHz_sin = time_refl_140GHz_cos = time_refl_140GHz_sin = timebase  
#            print("No values stored for reflectometry") 
#            print(shot)
#            break
#        except:
#            print("Unexpected error for reflectometry")
#            print(shot)
#            raise

    
    while True:
        try:
            electrons = MDSplus.Tree('ELECTRONS', shot)
            ne_t = (electrons.getNode('\THOM_MIDPLN:NE_T')).data() #Thomson Ne(+1.5) midplane
            time_thomson_ne_t = (electrons.getNode('\THOM_MIDPLN:NE_T')).dim_of().data()
            ne_t = np.interp(timebase,time_thomson_ne_t,ne_t,left=np.nan,right=np.nan)
            te_t = (electrons.getNode('\THOM_MIDPLN:TE_T')).data()*1000. #Thomson Te(+1.5)
            time_thomson_te_t = (electrons.getNode('\THOM_MIDPLN:TE_T')).dim_of().data()
            te_t = np.interp(timebase,time_thomson_te_t,te_t,left=np.nan,right=np.nan)
            break
        except TreeNODATA:
            ne_t = te_t = NaN
            time_thomson_ne_t = time_thomson_te_t = timebase  
            print("No values stored for thomson") 
            print(shot)
            break
        except:
            print("Unexpected error for thomson")
            print(shot)
            raise
    
    
    while True:
        try:
            gpc2_te0 = (electrons.getNode('\ELECTRONS::gpc2_te0')).data() #ECE GPC_T0
            time_gpc2_te0 = (electrons.getNode('\ELECTRONS::gpc2_te0')).dim_of().data()
            gpc2_te0 = np.interp(timebase,time_gpc2_te0,gpc2_te0,left=np.nan,right=np.nan)
            gpc_te8 = 1000.*((electrons.getNode('\ELECTRONS::gpc_te8')).data()) 
            time_gpc_te8 = (electrons.getNode('\ELECTRONS::gpc_te8')).dim_of().data()
            gpc_te8 = np.interp(timebase,time_gpc_te8,gpc_te8,left=np.nan,right=np.nan)
    	#gpc has 9 channels and gpc2 has 19 channels
            break
        except TreeNODATA:
            gpc2_te0 = gpc_te8 = NaN
            time_gpc2_te0 = time_gpc_te8 = timebase  
            print("No values stored for gpc") 
            print(shot)
            break
        except:
            print("Unexpected error for gpc")
            print(shot)
            raise
    
    while True:
        try:
            engineering = MDSplus.Tree('ENGINEERING', shot)
            piezo_4_gas_input = (engineering.getNode('\ENGINEERING::TOP.TORVAC.GAS.PVALVE_4:WAVEFORM')).data()
            time_gas_input = (engineering.getNode('\ENGINEERING::TOP.TORVAC.GAS.PVALVE_4:WAVEFORM')).dim_of().data()
            piezo_4_gas_input = np.interp(timebase,time_gas_input,piezo_4_gas_input,left=np.nan,right=np.nan)
            break
        except TreeNODATA:
            piezo_4_gas_input = NaN
            time_gas_input = timebase
            print("No values stored for engineering") 
            print(shot)
            break
        except:
            print("Unexpected error for engineering")
            print(shot)
            raise
            
            
    while True:
        try:
            pressure = MDSplus.Tree('edge', shot)
            g_side_rat = pressure.getNode('\g_side_rat').data() #[mtorr]
            time_g_side_rat = pressure.getNode(('\g_side_rat')).dim_of().data()
            g_side_rat = np.interp(timebase,time_g_side_rat,g_side_rat,left=np.nan,right=np.nan)
            e_bot_mks = (pressure.getNode('\e_bot_mks').data())[0] #[mtorr]
            time_e_bot_mks = pressure.getNode(('\e_bot_mks')).dim_of().data()
            e_bot_mks = np.interp(timebase,time_e_bot_mks,e_bot_mks,left=np.nan,right=np.nan)
            b_bot_mks = pressure.getNode('\\b_bot_mks').data()[0] #[mtorr]
            time_b_bot_mks = pressure.getNode(('\\b_bot_mks')).dim_of().data()
            b_bot_mks = np.interp(timebase,time_b_bot_mks,b_bot_mks,left=np.nan,right=np.nan)
            break
        except TreeNODATA:
            g_side_rat = e_bot_mks = b_bot_mks = NaN
            time_g_side_rat = time_e_bot_mks = time_b_bot_mks = timebase
            print("No values stored for edge pressures") 
            print(shot)
            break
        except:
            print("Unexpected error for edge pressures")
            print(shot)
            raise 
     
    np.savez('/home/mathewsa/Desktop/single_shot_training_table_py.npz', ip=ip,
    btor=btor,p_lh=p_lh,p_icrf=p_icrf,p_icrf_d=p_icrf_d,p_icrf_e=p_icrf_e,p_icrf_j3=p_icrf_j3,
    p_icrf_j4=p_icrf_j4,freq_icrf_d=freq_icrf_d,freq_icrf_e=freq_icrf_e,freq_icrf_j=freq_icrf_j,
    beta_N=beta_N,beta_p=beta_p,beta_t=beta_t,kappa=kappa,triang_l=triang_l,triang_u=triang_u,
    triang=triang,li=li,areao=areao,vout=vout,aout=aout,rout=rout,zout=zout,zmag=zmag,
    rmag=rmag,zsep_lower=zsep_lower,zsep_upper=zsep_upper,rsep_lower=rsep_lower,
    rsep_upper=rsep_upper,zvsin=zvsin,rvsin=rvsin,zvsout=zvsout,rvsout=rvsout,
    upper_gap=upper_gap,lower_gap=lower_gap,q0=q0,qstar=qstar,q95=q95,V_loop_efit=V_loop_efit,
    V_surf_efit=V_surf_efit,Wmhd=Wmhd,cpasma=cpasma,ssep=ssep,P_ohm=P_ohm,HoverHD=HoverHD,
    Halpha=Halpha,Dalpha=Dalpha,z_ave=z_ave,p_rad=p_rad,p_rad_core=p_rad_core,nLave_04=nLave_04,
    NL_04=NL_04,nebar_efit=nebar_efit,piezo_4_gas_input=piezo_4_gas_input,g_side_rat=g_side_rat,
    e_bot_mks=e_bot_mks,b_bot_mks=b_bot_mks)
    
    np.savez('/home/mathewsa/Desktop/extra_variables.npz', timebase=timebase)
     
    #omitted edge and core (which are not yet interpolated), engineering,QUICKFIT data, 
    #60/70/80/90/95,NL01-NL10, and cxrs data make certain arrays 0 if not called/unavailable    
    #perhaps include condition where if certain columns for a particular row are nan, 
     #or if error associated is too high, or if nonsensical (e.g. beta_p < 0), then do not save those
     #rows and remove those particular rows from the populated database
     
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    
