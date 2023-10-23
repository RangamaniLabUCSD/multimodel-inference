#!/bin/bash

# Kholodenko 2000
python inference_process.py -model kholodenko_2000 -free_params K8,v10,v9,K7,K9,KI,MAPK_total,K10 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 200 -ncores 10 -savedir ../../../results/MAPK/param_est/ -t1 4800 -input_state Input -ERK_states MAPK_PP

# LEVCHENKO 2000
python inference_process.py -model levchenko_2000 -free_params kOff1,kOn1,d8,RAFPase,k2,k6,k10,MEKPase,a2,kOn2,k3,d7,d9,d6,a10,kOff3,a9,a8,a3,a5,d3,d5,a6,k7,kOff4,d2,d10,a1,a4,k9,k5,k8,k4,d1,kOff2,a7,MAPKPase,d4,k1 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 200 -ncores 10 -savedir ../../../results/MAPK/param_est/ -t1 1000 -input_state RAFact -EGF_conversion_factor 0.001 -ERK_states MAPKstarstar

# Hatakeyama 2003
python inference_process.py -model hatakeyama_2003 -free_params k21,kf9,kb2,k19,kf3 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 200 -ncores 10 -savedir ../../../results/MAPK/param_est/ -t1 9600 -input_state HRG -ERK_states ERKPP

# Orton 2009
python inference_process.py -model orton_2009 -free_params km_Erk_Activation,k1_C3G_Deactivation,km_Erk_Deactivation,k1_P90Rsk_Deactivation -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 200 -ncores 10 -savedir ../../../results/MAPK/param_est/ -t1 360 -input_state EGF -EGF_conversion_factor 602214 -ERK_states ErkActive

# von Kriegsheim 2009
python inference_process.py -model vonKriegsheim_2009 -free_params k42,k4,k30,k68,k29,k32,k34 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 200 -ncores 10 -savedir ../../../results/MAPK/param_est/ -t1 3600 -input_state EGF -EGF_conversion_factor 6.048 -ERK_states ppERK,ppERK_15,ppERKn

# # Shin 2014
# python inference_process.py -model shin_2014 -free_params kc47,kc43,kd39,kc45,ERK_tot,ki39,kc41 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 200 -savedir ../../../results/MAPK/param_est/ -input_stat EGF -EGF_conversion_factor 0.001 -ERK_states pp_ERK -t1 540

# Ryu 2015
python inference_process.py -model ryu_2015 -free_params D2,T_dusp,K_dusp,K2,dusp_ind -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 200 -ncores 10 -savedir ../../../results/MAPK/param_est/ -input_stat EGF -EGF_conversion_factor 6.048 -ERK_states ERK_star -t1 540

# Kochańczyk 2017
python inference_process.py -model kochanczyk_2017 -free_params q1,q2,q6,d2,q4 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 200 -ncores 10 -savedir ../../../results/MAPK/param_est/ -input_stat EGF -EGF_conversion_factor 6048 -ERK_states ERKSPP -t1 7200

# Birghtman and Fell 2000
python inference_process.py -model brightman_fell_2000 -free_params kn14,K_24,kn16,V_26,V_24,kn12,k15,k_13,K_25,K_26,k17 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 200 -ncores 10 -savedir ../../../results/MAPK/param_est/ -t1 360 -input_state L -EGF_conversion_factor 1e-7 -ERK_states ERKPP

# Hornberg 2005
python inference_process.py -model hornberg_2005 -free_params k42,k18,kd57,kd42,kd55,kd19,k44,k4 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 200 -ncores 10 -savedir ../../../results/MAPK/param_est/ -t1 12000 -input_state c1 -EGF_conversion_factor 0.000000001 -ERK_states c59,c83

# BIRTWISTLE 2007
python inference_process.py -model birtwistle_2007 -free_params Kmf52,koff57,koff91,koff88,Kmr52,kon91,kcat94,kon93,koff58,kcat96,kcat92,koff46,kcon49,koff93,Vmaxr52,koff74,koff44,kf48,kon89,kon95 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 200 -ncores 10 -savedir ../../../results/MAPK/param_est/ -t1 5400 -input_state E -ERK_states ERKstar,ERKstar_ERKpase
