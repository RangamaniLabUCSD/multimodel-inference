#!/bin/bash

# Kholodenko 2000
python inference_process_dose_response.py -model kholodenko_2000 -free_params K8,v10,v9,KI,MAPK_total,K10  -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -ncores 1 -savedir ../../../results/MAPK/param_est/ -input_state Input -ERK_states MAPK_PP

# LEVCHENKO 2000
python inference_process_dose_response.py -model levchenko_2000 -free_params a2,k10,RAFPase,k4,k6,kOn2,total_scaffold,a10,kOff3,k5,a9,a8,a6,d2,a7,d10,a1,k9,k2,d1,k3,MEKPase,kOff2,MAPKPase,k1 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -ncores 1 -savedir ../../../results/MAPK/param_est/ -input_state RAFact -EGF_conversion_factor 0.001 -ERK_states MAPKstarstar

# Birghtman and Fell 2000
# FIXME: - investigate further, model is problematic
# python inference_process_dose_response.py -model brightman_fell_2000 -free_params FIXME:TBD -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -ncores 1 -savedir ../../../results/MAPK/param_est/ -input_state L -EGF_conversion_factor 602214 -ERK_states ERKPP

# Hatakeyama 2003
python inference_process_dose_response.py -model hatakeyama_2003 -free_params k21, k20,kf9,kb1,kb2,kb7,kf8,k19,k22,kb5,kf6,kf3,kb6 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -ncores 1 -savedir ../../../results/MAPK/param_est/ -input_state HRG -ERK_states ERKPP

Hornberg 2005
python inference_process_dose_response_jax_NUTS.py -model hornberg_2005 -free_params kd1,k6,k8,k18,k42,kd42,k43,k44,kd52,kd45,k47,kd47,kd48,k49,kd49,kd50,k52,kd44,k53,kd53,k55,kd55,k56,kd56,k57,kd57,k58,kd58 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -savedir ../../../results/MAPK/param_est/ -input_state c1 -EGF_conversion_factor 0.000000001 -ERK_states c59,c83

# Orton 2009
python inference_process_dose_response.py -model orton_2009 -free_params km_Erk_Activation, k1_C3G_Deactivation, km_Erk_Deactivation, k1_P90Rsk_Deactivation,k1_Sos_Deactivation -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -ncores 1 -savedir ../../../results/MAPK/param_est/ -input_state EGF -EGF_conversion_factor 602214 -ERK_states ErkActive

# von Kriegsheim 2009
python inference_process_dose_response.py -model vonKriegsheim_2009 -free_params  k42,k37,k4,k27,k30,k5,k28,k68,k29,k41,k25,k7,k13,k2,k40,k32,k10,k34 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -savedir ../../../results/MAPK/param_est/ -input_state EGF -EGF_conversion_factor 6.048 -ERK_states ppERK,ppERK_15,ppERKn

# Shin 2014
python inference_process_dose_response.py -model shin_2014 -free_params kc47,kc43,kd39,kc45,ERK_tot,ki39,kc41 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -savedir ../../../results/MAPK/param_est/ -input_stat EGF -EGF_conversion_factor 0.001 -ERK_states pp_ERK 

# Ryu 2015
python inference_process_dose_response.py -model ryu_2015 -free_params D2,T_dusp,K_dusp,K2,dusp_ind -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -ncores 1 -savedir ../../../results/MAPK/param_est/ -input_stat EGF -EGF_conversion_factor 6.048 -ERK_states ERK_star 

# Kochańczyk 2017
python inference_process_dose_response.py -model kochanczyk_2017 -free_params q1,q3,q2,u3,d1,q6,u2b,u1a,d2,u2a,q4 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -ncores 1 -savedir ../../../results/MAPK/param_est/ -input_stat EGF -EGF_conversion_factor 6048 -ERK_states ERKSPP 

# BIRTWISTLE 2007
python inference_process_dose_response.py -model birtwistle_2007 -free_params Kmf52,koff57,EGF_off,kcat90,koff91,koff40,koff88,koff95,kon91,kcat94,koff42,kon93,kcat92,koff46,kcon49,koff41,Vmaxr52,koff44,kon89,kon95 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -ncores 1 -savedir ../../../results/MAPK/param_est/ -input_state E -ERK_states ERKstar,ERKstar_ERKpase

