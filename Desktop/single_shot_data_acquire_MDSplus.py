# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:38:18 2018

@author: mathewsa
"""

#retrieving data for a single shot

import MDSplus 
from MDSplus import *
import numpy as np   
import os
import sys
from os import getenv
from datetime import datetime

def smooth(y,box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y,box,mode='same')
    return y_smooth

shot = 1160930033 #1100204004 #1140724005 #1140221015  
tree = Tree('cmod', shot)

while True: #plasma current is used as threshold for beginning and end of time series
    try:
        magnetics = MDSplus.Tree('magnetics', shot)
        ip = magnetics.getNode('\ip').data()
        time_ip = magnetics.getNode('\ip').dim_of().data()  
        break
    except TreeNODATA: 
        print("No values stored for ip") 
        print(shot)
        raise
    except:
        print("Unexpected error for ip")
        print(shot)
        raise  
index_begin = np.min(np.where(np.abs(ip) > 100000.)[0])
index_end = np.max(np.where(np.abs(ip) > 100000.)[0])
start_time = time_ip[index_begin]
end_time = time_ip[index_end]

timebase = []
timebase = np.arange(round(start_time,3),round(end_time,3),0.001)
NaN = (np.empty(len(timebase)) * np.nan)
zeros = np.zeros(len(timebase))
#can define an alternative timebase if desired, but using this
#definition of > 100kA as start/end condition currently for table

while True:
    try:
        btor=tree.getNode('\\BTOR') 
        btor_data=btor.record 
        Y = btor_data.data()
        time_btor = btor.getNode('\BTOR').dim_of().data()
        btor = Y
        btor = np.interp(timebase,time_btor,btor,left=np.nan,right=np.nan)
        ip = np.interp(timebase,time_ip,ip,left=np.nan,right=np.nan)
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
	print("Unexpected error for p_lh")
        print(shot)
        raise 


while True:
    try:
        p_icrf = MDSplus.Tree('rf', shot)
        time_p_icrf = (p_icrf.getNode('\RF::RF_POWER_NET')).dim_of().data()
        p_icrf = 1000000.*(p_icrf.getNode('\RF::RF_POWER_NET')).data(); #[W]
        p_icrf = np.interp(timebase,time_p_icrf,p_icrf,left=np.nan,right=np.nan)
	break
    except TreeNODATA:
        p_icrf = NaN
        time_p_icrf = timebase
        print("No values stored for p_icrf") 
        print(shot)
        break
    except:
        print("Unexpected error for p_icrf")
        print(shot)
        raise

while True:
    try:  
        efit = MDSplus.Tree('analysis', shot)
        time_efit = (efit.getNode('\efit_aeqdsk:time')).data()
        beta_N = (efit.getNode('\efit_aeqdsk:betan')).data(); #normalized beta
        beta_N = np.interp(timebase,time_efit,beta_N,left=np.nan,right=np.nan)
        beta_p = (efit.getNode('\efit_aeqdsk:betap')).data(); #beta_poloidal
        beta_p = np.interp(timebase,time_efit,beta_p,left=np.nan,right=np.nan)
        beta_t = (efit.getNode('\efit_aeqdsk:betat')).data(); #beta_toroidal
        beta_t = np.interp(timebase,time_efit,beta_t,left=np.nan,right=np.nan)
        kappa = (efit.getNode('\efit_aeqdsk:eout')).data(); #elongation - vertical
        kappa = np.interp(timebase,time_efit,kappa,left=np.nan,right=np.nan)
        triang_l = (efit.getNode('\efit_aeqdsk:doutl')).data(); #lower triangularity of lcfs
        triang_l = np.interp(timebase,time_efit,triang_l,left=np.nan,right=np.nan)
        triang_u = (efit.getNode('\efit_aeqdsk:doutu')).data(); #upper triangularity of lcfs
        triang_u = np.interp(timebase,time_efit,triang_u,left=np.nan,right=np.nan)
        triang = (triang_u + triang_l)/2. #overall triangularity - horizontal (major radius)
        li = (efit.getNode('\efit_aeqdsk:li')).data(); #internal inductance
        li = np.interp(timebase,time_efit,li,left=np.nan,right=np.nan)
        areao = (efit.getNode('\efit_aeqdsk:areao')).data(); #area of lcfs
        areao = np.interp(timebase,time_efit,areao,left=np.nan,right=np.nan)
        vout = (efit.getNode('\efit_aeqdsk:vout')).data(); #volume of lcfs
        vout = np.interp(timebase,time_efit,vout,left=np.nan,right=np.nan)
        aout = (efit.getNode('\efit_aeqdsk:aout')).data(); #minor radius of lcfs
        aout = np.interp(timebase,time_efit,aout,left=np.nan,right=np.nan)
        rout = (efit.getNode('\efit_aeqdsk:rout')).data(); #major radius of geometric center
        rout = np.interp(timebase,time_efit,rout,left=np.nan,right=np.nan)
        zmag = (efit.getNode('\efit_aeqdsk:zmagx')).data(); #z of magnetic axis
        zmag = np.interp(timebase,time_efit,zmag,left=np.nan,right=np.nan)
        zout = (efit.getNode('\efit_aeqdsk:zout')).data(); #z of lcfs (constructed)
        zout = np.interp(timebase,time_efit,zout,left=np.nan,right=np.nan)
        zseps = (efit.getNode('\efit_aeqdsk:zseps')).data(); #z of upper and lower xpts
        zsep_lower = zseps[:,0]
        zsep_upper = zseps[:,1]  
        zsep_lower = np.interp(timebase,time_efit,zsep_lower,left=np.nan,right=np.nan)
        zsep_upper = np.interp(timebase,time_efit,zsep_upper,left=np.nan,right=np.nan)
        zvsin = (efit.getNode('\efit_aeqdsk:zvsin')).data(); #z of inner strike point
        zvsin = np.interp(timebase,time_efit,zvsin,left=np.nan,right=np.nan)
        zvsout = (efit.getNode('\efit_aeqdsk:zvsout')).data(); #z of outer strike point
        zvsout = np.interp(timebase,time_efit,zvsout,left=np.nan,right=np.nan)
        upper_gap = (efit.getNode('\efit_aeqdsk:otop')).data()/100.; # meters
        upper_gap = np.interp(timebase,time_efit,upper_gap,left=np.nan,right=np.nan)
        lower_gap = (efit.getNode('\efit_aeqdsk:obott')).data()/100.; # meters
        lower_gap = np.interp(timebase,time_efit,lower_gap,left=np.nan,right=np.nan)
        q0 = (efit.getNode('\efit_aeqdsk:q0')).data(); #safety factor at center
        q0 = np.interp(timebase,time_efit,q0,left=np.nan,right=np.nan)
        qstar = (efit.getNode('\efit_aeqdsk:qstar')).data(); #cylindrical safety factor
        qstar = np.interp(timebase,time_efit,qstar,left=np.nan,right=np.nan)
        q95 = (efit.getNode('\efit_aeqdsk:q95')).data(); #edge safety factor
        q95 = np.interp(timebase,time_efit,q95,left=np.nan,right=np.nan)
        V_loop_efit = (efit.getNode('\efit_aeqdsk:vloopt')).data(); #loop voltage
        V_loop_efit = np.interp(timebase,time_efit,V_loop_efit,left=np.nan,right=np.nan)
        V_surf_efit = (efit.getNode('\efit_aeqdsk:vsurfa')).data(); #surface voltage
        V_surf_efit = np.interp(timebase,time_efit,V_surf_efit,left=np.nan,right=np.nan)
        Wmhd = (efit.getNode('\efit_aeqdsk:wplasm')).data(); #diamagnetic/stored energy, [J]
        Wmhd = np.interp(timebase,time_efit,Wmhd,left=np.nan,right=np.nan)
        ssep = (efit.getNode('\efit_aeqdsk:ssep')).data()/100.; # distance on midplane between 1st and 2nd separatrices [m]
        ssep = np.interp(timebase,time_efit,ssep,left=np.nan,right=np.nan)
        n_over_ncrit = (efit.getNode('\efit_aeqdsk:xnnc')).data(); #vertical stability criterion (EFIT name: xnnc)
        n_over_ncrit = np.interp(timebase,time_efit,n_over_ncrit,left=np.nan,right=np.nan)  
        inductance = 4.*np.pi*1.E-7 * 0.68 * li/2.; # For simplicity, we use R0 = 0.68 m, but we could use \efit_aeqdsk:rmagx
        dipdt = np.gradient(ip,timebase)
        dipdt_smoothed = smooth(dipdt,11) #11-point smoothing (moving average box/by convolution)
        V_inductive = inductance*dipdt_smoothed
        V_resistive = V_loop_efit - V_inductive
        P_ohm = ip*V_resistive
        
        pcurrt = (efit.getNode('\efit_g_eqdsk:pcurrt')).data()
        refit = (efit.getNode('\efit_g_eqdsk:pcurrt')).dim_of().data()
        efit_rmid = (efit.getNode('\\analysis::top.efit.results:fitout:rpres')).data()
        volp = (efit.getNode('\\analysis::top.efit.results:fitout:volp')).data() #array of volume within flux surface
        fpol = (efit.getNode('\\analysis::top.efit.results:fitout:fpol')).data()
        pres_flux = (efit.getNode('\\analysis::top.efit.results:fitout:pres')).data() #array of pressure on flux surface psi
        ffprime = (efit.getNode('\\analysis::top.efit.results:fitout:ffprim')).data()
        pprime = (efit.getNode('\\analysis::top.efit.results:fitout:pprime')).data()#plasma pressure gradient as function of psi        
        qpsi = (efit.getNode('\\analysis::top.efit.results:fitout:qpsi')).data() #array of q, safety factor, on flux surface psi        
#        rlcfs = (efit.getNode('\\analysis::top.efit.results:fitout:_RLCFS')).data() 
#possibly include aspect ratio ~ rout/aout
#can find time derivatives of quantities like beta, li, Wmhd 
        break
    except TreeNODATA:
        time_efit = timebase
        beta_N = beta_p = beta_t = kappa = triang_l = triang_u =\
        triang = li = areao = vout = aout = rout = zmag =\
        zout = zseps = zvsin = zvsout = upper_gap = lower_gap =\
        q0 = qstar = q95 = V_loop_efit = V_surf_efit = Wmhd = ssep =\
        n_over_ncrit = P_ohm = NaN
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
        temp_core = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:TE_RZ')).data(); #temperature (keV)
        temp_core_err = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:TE_ERR')).data(); #error (keV)
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
        H_HandD = (spectroscopy.getNode('\SPECTROSCOPY::BALMER_H_TO_D')).data(); #H/(H+D)
        time_H_HandD = spectroscopy.getNode('\SPECTROSCOPY::BALMER_H_TO_D').dim_of().data()
        H_HandD = np.interp(timebase,time_H_HandD,H_HandD,left=np.nan,right=np.nan)
        Halpha = (spectroscopy.getNode('\SPECTROSCOPY::HA_2_BRIGHT')).data(); #H-Alpha at H Port
        time_Halpha = (spectroscopy.getNode('\SPECTROSCOPY::HA_2_BRIGHT')).dim_of().data();
        Halpha = np.interp(timebase,time_Halpha,Halpha,left=np.nan,right=np.nan)
        Dalpha = (spectroscopy.getNode('\SPECTROSCOPY::TOP.VUV.VIS_SIGNALS:MCP_VIS_SIG1')).data() # D-alpha (W/m^2/st)
        time_Dalpha = (spectroscopy.getNode('\SPECTROSCOPY::TOP.VUV.VIS_SIGNALS:MCP_VIS_SIG1')).dim_of().data()
        Dalpha = np.interp(timebase,time_Dalpha,Dalpha,left=np.nan,right=np.nan)
        z_ave = (spectroscopy.getNode('\SPECTROSCOPY::z_ave')).data() 
        time_z_ave = (spectroscopy.getNode('\SPECTROSCOPY::z_ave')).dim_of().data() 
        z_ave = np.interp(timebase,time_z_ave,z_ave,left=np.nan,right=np.nan)
        p_rad = MDSplus.Tree('SPECTROSCOPY', shot) #[W]
        time_p_rad = (p_rad.getNode('\TWOPI_FOIL')).dim_of().data()
        p_rad = (p_rad.getNode('\TWOPI_FOIL')).data()
        p_rad = np.interp(timebase,time_p_rad,p_rad,left=np.nan,right=np.nan)
#use twopi_diode instead as in Granetz code if avoiding non-causal filtering
#rad_fraction = p_rad/p_input (if p_input==0 then NaN/0)
        break
    except TreeNODATA:
        H_HandD = Halpha = Dalpha = z_ave = p_rad = NaN
        time_H_HandD = time_Halpha = time_Dalpha = time_z_ave = time_p_rad = timebase    
        print("No values stored for spectroscopy") 
        print(shot)
        break
    except:
        print("Unexpected error for spectroscopy")
        print(shot)
        raise


while True:
    try:
        cxrs = MDSplus.Tree('DNB', shot)
        Vpol = 1000.*((cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.POLOIDAL:VEL')).data()) #poloidal velocity [m/s]
        time_Vpol = (cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.POLOIDAL:VEL')).dim_of().data()
        dVpol = 1000.*((cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.POLOIDAL:VEL_SIGMA')).data()) #poloidal velocity sigma [m/s]
        Vpol = np.interp(timebase,time_Vpol,Vpol,left=np.nan,right=np.nan)
        Vtor = 1000.*((cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.TOR_OUT:VEL')).data()) #toroidal velocity [m/s]$
        time_Vtor = (cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.TOR_OUT:VEL')).dim_of().data() 
        dVtor = 1000.*((cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.TOR_OUT:VEL_SIGMA')).data()) #toroidal velocity sigma [m/s]
        Vtor = np.interp(timebase,time_Vtor,Vtor,left=np.nan,right=np.nan)
	break
    except TreeNODATA:
        Vpol = dVpol = Vtor = dVtor = NaN
        time_Vpol = time_Vtor = timebase       
        print("No values stored for cxrs") 
        print(shot)
        break
    except:
        print("Unexpected error for cxrs")
        print(shot)
        raise

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
        nebar_efit = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS.INVERSION:NEBAR_EFIT')).data()
        time_nebar_efit = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS.INVERSION:NEBAR_EFIT')).dim_of().data() #NeBar_EFIT (TCI)
        nebar_efit = np.interp(timebase,time_nebar_efit,nebar_efit,left=np.nan,right=np.nan)
	break
    except TreeNODATA:
        NL_01 = NL_02 = NL_03 = NL_04 = NL_05 = NL_06 = NL_07 =\
        NL_08 = NL_09 = NL_10 = nebar_efit = NaN
        time_NL_04 = time_nebar_efit = timebase  
        print("No values stored for tci") 
        print(shot)
        break
    except:
        print("Unexpected error for tci")
        print(shot)
        raise


while True:
    try:
        electrons = MDSplus.Tree('ELECTRONS', shot)
        ne_t = (electrons.getNode('\THOM_MIDPLN:NE_T')).data() #Thomson Ne(+1.5) midplane
        time_thomson_ne_t = (electrons.getNode('\THOM_MIDPLN:NE_T')).dim_of().data()
        ne_t = np.interp(timebase,time_thomson_ne_t,ne_t,left=np.nan,right=np.nan)
        te_t = (electrons.getNode('\THOM_MIDPLN:TE_T')).data() #Thomson Te(+1.5)
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
        b_bot_mks = pressure.getNode('\\b_bot_mks').data() #[mtorr]
        time_b_bot_mks = pressure.getNode(('\\b_bot_mks')).dim_of().data()
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
        

update_time = datetime.now()    
update_time = str(datetime.now()) 
update_time = [update_time]*len(timebase) 

shot = np.asarray([shot]*len(timebase),int) 