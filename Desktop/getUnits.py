# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:50:14 2018

@author: mathewsa
"""

magnetics = MDSplus.Tree('magnetics', shot)
magnetics.getNode('\ip').units_of() #ampere
(magnetics.getNode('\ip').dim_of()).units_of() # N/A
magnetics.getNode('\\BTOR').units_of() #tesla 
(magnetics.getNode('\\BTOR').dim_of()).units_of() # N/A

p_lh = MDSplus.Tree('LH', shot)
p_lh.getNode('\LH::TOP.RESULTS:NETPOW').units_of() #kW, *1000 = W
p_lh.getNode('\LH::TOP.RESULTS:NETPOW').dim_of().units_of() #seconds

p_icrf = MDSplus.Tree('rf', shot)
p_icrf.getNode('\RF::RF_POWER_NET').units_of() #MW, *1000000 = W
(p_icrf.getNode('\RF::RF_POWER_NET')).dim_of().getUnits() #seconds

efit = MDSplus.Tree('{}'.format((path_shot['{}'.format(shot)])[0]), shot)
(efit.getNode('\efit_aeqdsk:time')).units_of() #seconds
beta_N = (efit.getNode('\efit_aeqdsk:betan')).units_of() #'%/a*cm*T'
beta_p = (efit.getNode('\efit_aeqdsk:betap')).units_of() # N/A
beta_t = (efit.getNode('\efit_aeqdsk:betat')).units_of() # N/A
kappa = (efit.getNode('\efit_aeqdsk:eout')).units_of() # N/A
triang_l = (efit.getNode('\efit_aeqdsk:doutl')).units_of() # N/A
triang_u = (efit.getNode('\efit_aeqdsk:doutu')).units_of() # N/A
triang = (triang_u + triang_l)/2.  
li = (efit.getNode('\efit_aeqdsk:li')).units_of() # N/A ???
areao = (efit.getNode('\efit_aeqdsk:areao')).units_of() #cm^2
vout = (efit.getNode('\efit_aeqdsk:vout')).units_of() #cm^3
aout = (efit.getNode('\efit_aeqdsk:aout')).units_of() #cm
rout = (efit.getNode('\efit_aeqdsk:rout')).units_of() #cm
zmag = (efit.getNode('\efit_aeqdsk:zmagx')).units_of() #cm
zout = (efit.getNode('\efit_aeqdsk:zout')).units_of() #cm
zseps = (efit.getNode('\efit_aeqdsk:zseps')).units_of() #cm
zsep_lower = zseps[:,0]
zsep_upper = zseps[:,1]    
zvsin = (efit.getNode('\efit_aeqdsk:zvsin')).units_of() #cm
zvsout = (efit.getNode('\efit_aeqdsk:zvsout')).units_of()  #cm
upper_gap = (efit.getNode('\efit_aeqdsk:otop')).units_of() #cm, /100. = m; 
lower_gap = (efit.getNode('\efit_aeqdsk:obott')).units_of() #cm, /100. = m;  
q0 = (efit.getNode('\efit_aeqdsk:q0')).units_of() #N/A
qstar = (efit.getNode('\efit_aeqdsk:qstar')).units_of() #N/A
q95 = (efit.getNode('\efit_aeqdsk:q95')).units_of() #N/A
V_loop_efit = (efit.getNode('\efit_aeqdsk:vloopt')).units_of() #V
V_surf_efit = (efit.getNode('\efit_aeqdsk:vsurfa')).units_of() #V
Wmhd = (efit.getNode('\efit_aeqdsk:wplasm')).units_of() #J
ssep = (efit.getNode('\efit_aeqdsk:ssep')).units_of() #cm, /100. = m;  
n_over_ncrit = (efit.getNode('\efit_aeqdsk:xnnc')).units_of() #N/A

electrons = MDSplus.Tree('ELECTRONS', shot) #NEW CORE only valid for SHOT>1020000000
time_core = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_RZ')).dim_of().units_of() #seconds
dens_core = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_RZ')).units_of(); #density (m^-3)
dens_core_err = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_ERR')).units_of(); #error (m^-3)
temp_core = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:TE_RZ')).units_of(); #temperature (keV)
temp_core_err = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:TE_ERR')).units_of(); #error (keV)
midR_core = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:R_MID_T')).units_of(); #mapped midplane R (m)
z_core = (electrons.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:Z_SORTED')).units_of(); # N/A? I believe z-position (m) 

electrons = MDSplus.Tree('ELECTRONS', shot) #NEW EDGE only valid for SHOT>1000000000
time_edge = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE')).dim_of().units_of() #seconds
dens_edge = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE')).units_of(); #density (m^-3)
dens_edge_err = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE:ERROR')).units_of(); #error (m^-3)
temp_edge = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:TE')).units_of(); #temperature (eV)
temp_edge_err = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:TE:ERROR')).units_of(); #error (eV)
midR_edge = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:RMID')).units_of(); #mapped midplane R (m) coreTS
z_edge = (electrons.getNode('\ELECTRONS::TOP.YAG_EDGETS.DATA:FIBER_Z')).units_of(); # N/A ??? but I believe z-position (m)
z_midplane = (electrons.getNode('\electrons::top.yag_edgets.data.fiber_z:holder_z')).units_of() #N/A ??? but I believe edge TS fibers are
#a movable block, and can be shifted vertically to position the scattering volumes as desired relative to the LCFS. 
#Edge fiber position is recorded as a measurement in mm below the midplane
R_edge = (electrons.getNode('\ELECTRONS::TOP.YAG.RESULTS.PARAM:R')).units_of() #N/A but I believe v R-position (m)

spectroscopy = MDSplus.Tree('SPECTROSCOPY', shot)
HoverHD = (spectroscopy.getNode('\SPECTROSCOPY::BALMER_H_TO_D')).units_of() ; #N/A H/(H+D)
time_HoverHD = spectroscopy.getNode('\SPECTROSCOPY::BALMER_H_TO_D').dim_of().units_of() #N/A?? (I believe seconds)
Halpha = (spectroscopy.getNode('\SPECTROSCOPY::HA_2_BRIGHT')).units_of() ; #mW/cm^2/ster    H-Alpha at H Port
time_Halpha = (spectroscopy.getNode('\SPECTROSCOPY::HA_2_BRIGHT')).dim_of().units_of() ; #seconds 
Dalpha = (spectroscopy.getNode('\SPECTROSCOPY::TOP.VUV.VIS_SIGNALS:MCP_VIS_SIG1')).units_of()  # 'W/m^2/st' D-alpha (W/m^2/st)
time_Dalpha = (spectroscopy.getNode('\SPECTROSCOPY::TOP.VUV.VIS_SIGNALS:MCP_VIS_SIG1')).dim_of().units_of() #seconds
z_ave = (spectroscopy.getNode('\SPECTROSCOPY::z_ave')).units_of() # N/A ??
time_z_ave = (spectroscopy.getNode('\SPECTROSCOPY::z_ave')).dim_of().units_of() #N/A
p_rad = MDSplus.Tree('SPECTROSCOPY', shot)  
time_p_rad = (p_rad.getNode('\TWOPI_FOIL')).dim_of().units_of() #seconds
p_rad = (p_rad.getNode('\TWOPI_FOIL')).units_of() #W
p_rad_core = (p_rad.getNode('\\top.bolometer.results.foil:main_power')).units_of() #MW


cxrs = MDSplus.Tree('DNB', shot)
Vpol = 1000.*((cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.POLOIDAL:VEL')).units_of()) #cxrs unavailable?
time_Vpol = (cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.POLOIDAL:VEL')).dim_of().units_of() #cxrs unavailable?
dVpol = 1000.*((cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.POLOIDAL:VEL_SIGMA')).units_of()) #cxrs unavailable?
Vtor = 1000.*((cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.TOR_OUT:VEL')).units_of()) #cxrs unavailable?
time_Vtor = (cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.TOR_OUT:VEL')).dim_of().units_of() #cxrs unavailable?
dVtor = 1000.*((cxrs.getNode('\DNB::TOP.MIT_CXRS.RESULTS.ACTIVE.TOR_OUT:VEL_SIGMA')).units_of()) #cxrs unavailable?
 
TCI = MDSplus.Tree('ELECTRONS', shot)
NL_01 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_01')).units_of() #N/A for NL_01 - NL10 ??
NL_02 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_02')).units_of() #although I believe units are meters^(-2)
NL_03 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_03')).units_of()
NL_04 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')).units_of()
NL_05 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_05')).units_of()
NL_06 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_06')).units_of()
NL_07 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_07')).units_of()
NL_08 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_08')).units_of()
NL_09 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_09')).units_of()
NL_10 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_10')).units_of()
r_NL = (TCI.getNode('\electrons::top.tci.results:rad')).units_of() #N/A ?? I believe units are m
time_NL_04 = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')).dim_of().units_of() #seconds; same time for each chord
nebar_efit = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS.INVERSION:NEBAR_EFIT')).units_of()
time_nebar_efit = (TCI.getNode('\ELECTRONS::TOP.TCI.RESULTS.INVERSION:NEBAR_EFIT')).dim_of().units_of() 

electrons = MDSplus.Tree('ELECTRONS', shot)
ne_t = (electrons.getNode('\THOM_MIDPLN:NE_T')).units_of() #m^-3
time_thomson_ne_t = (electrons.getNode('\THOM_MIDPLN:NE_T')).dim_of().units_of() #seconds 
te_t = (electrons.getNode('\THOM_MIDPLN:TE_T')).units_of() #keV
time_thomson_te_t = (electrons.getNode('\THOM_MIDPLN:TE_T')).dim_of().units_of() #seconds

gpc2_te0 = (electrons.getNode('\ELECTRONS::gpc2_te0')).units_of() #N/A ???
time_gpc2_te0 = (electrons.getNode('\ELECTRONS::gpc2_te0')).dim_of().units_of() #N/A ???
gpc_te8 = 1000.*((electrons.getNode('\ELECTRONS::gpc_te8')).units_of()) #keV
time_gpc_te8 = (electrons.getNode('\ELECTRONS::gpc_te8')).dim_of().units_of() #seconds

engineering = MDSplus.Tree('ENGINEERING', shot)
piezo_4_gas_input = (engineering.getNode('\ENGINEERING::TOP.TORVAC.GAS.PVALVE_4:WAVEFORM')).units_of() #engineering unavailable?
time_gas_input = (engineering.getNode('\ENGINEERING::TOP.TORVAC.GAS.PVALVE_4:WAVEFORM')).dim_of().units_of() #engineering unavailable?

pressure = MDSplus.Tree('edge', shot)
g_side_rat = pressure.getNode('\g_side_rat').units_of() #[mtorr]
time_g_side_rat = pressure.getNode(('\g_side_rat')).dim_of().units_of() #seconds
e_bot_mks = (pressure.getNode('\e_bot_mks').units_of())[0] #[mtorr]
time_e_bot_mks = pressure.getNode(('\e_bot_mks')).dim_of().units_of() #seconds