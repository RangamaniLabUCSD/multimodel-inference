#!/usr/bin/bash

# Huang Ferrell 1996
#CUDA_VISIBLE_DEVICES=0 python gsa_sample.py huang_ferrell_1996 -analyze_params MKK_tot,a3,k7,a4,d3,d10,d2,a2,d6,a8,a1,E2_tot,a7,a9,k5,a6,d1,d9,d5,k8,d8,k4,k6,k9,k3,d7,a10,MAPK_tot,k2,d4,a5,MKKK_tot,k10,MKKPase_tot,k1 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 3000 -input_param E1_tot

# Kholodenko 2000
#CUDA_VISIBLE_DEVICES=0 python gsa_sample.py kholodenko_2000 -analyze_params K8,v10,v9,K7,K9,KI,MAPK_total,K10 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 2400 -input_state Input --full_trajectory -ERK_state_indices -1

# Birghtman and Fell 2000
#CUDA_VISIBLE_DEVICES=0 python gsa_sample.py brightman_fell_2000 -analyze_params kn14,K_24,kn16,V_26,kn1,k3,V_24,kn12,k15,k_13,kn7,kn11,K_25,k2_4,K_26,k17,K_23,DT -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 360 -input_state L -input 6022

# LEVCHENKO 2000
# run on cortex
python gsa_sample.py levchenko_2000 -analyze_params kOff1,kOn1,d8,RAFPase,k2,k6,k10,MEKPase,a2,kOn2,k3,d7,d9,d6,a10,kOff3,a9,a8,a3,a5,d3,d5,a6,k7,kOff4,d2,d10,a1,a4,k9,k5,k8,k4,d1,kOff2,a7,MAPKPase,d4,k1 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 1000 -input_state RAFact -input 0.00001

# Hatakeyama 2003
#CUDA_VISIBLE_DEVICES=0 python gsa_sample.py hatakeyama_2003 -analyze_params k21,k20,kb29,kf9,kb24,kb3,kb23,kf25,kb1,kb2,kf24,kb7,kf8,k19,kf34,k22,kb5,kf6,kf3,kb6 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 3600 -input_state HRG --full_trajectory -ERK_state_indices 32

# Hornberg 2005
# this is run on cortex with 2 GPU cards (runs out of memory on a single RTX 2080 Ti)
XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py hornberg_2005 -analyze_params k42,k28,k52,kd50,k20,k6,kd45,k3,k18,k17,k25,k48,kd48,kd127,kd3,kd10,kd40,k61,kd5,k33,k16,kd22,kd4,kd34,kd44,k15,kd32,k10b,kd49,kd57,kd20,k21,k40,kd52,kd58,kd1,k8,kd53,kd35,k37,kd56,kd42,kd6,kd126,k35,kd23,kd33,kd47,kd55,kd25,kd18,kd19,k32,kd28,kd37,k44,kd8,kd17,k2,k19,k50,k41,k13,k34,kd21,kd41,k60,k126,k23,k29,kd29,kd2,k4,k58,k22,kd63,kd24,k56,k36 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 6000 -input_state c1 -input 1e-10 --full_trajectory -ERK_state_indices 58,82

# BIRTWISTLE 2007

# Orton 2009
#CUDA_VISIBLE_DEVICES=0 python gsa_sample.py orton_2009 -analyze_params km_Erk_Activation,k1_C3G_Deactivation,km_Erk_Deactivation,k1_Akt_Deactivation,k1_P90Rsk_Deactivation,k1_PI3K_Deactivation,k1_Sos_Deactivation -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 720 -input_state EGF -input 6022

# von Kriegsheim 2009
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py vonKriegsheim_2009 -analyze_params k42,k37,k4,k27,k45,k30,k43,k48,k5,k14,k28,k39,k46,k63,k68,k55,k29,k41,k25,k7,k13,k2,k40,k6,k18,k56,k32,k38,k10,k34 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 1800 -input_state EGF -input 0.06048 --full_trajectory -ERK_state_indices 26,28,29

# # Shin 2014
#CUDA_VISIBLE_DEVICES=0 python gsa_sample.py shin_2014 -analyze_params kc47,kc43,kd39,kc45,ERK_tot,ki39,kc41 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 540 -input_state EGF -input 1e-5

# # Ryu 2015
#CUDA_VISIBLE_DEVICES=0 python gsa_sample.py ryu_2015 -analyze_params D2,T_dusp,K_dusp,K2,dusp_ind -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 540 -input_state EGF -input 0.06048

# # Kochańczyk 2017
#CUDA_VISIBLE_DEVICES=0 python gsa_sample.py kochanczyk_2017 -analyze_params k3,q1,q3,q2,u3,d1,q6,u2b,u1a,d2,u2a,q5,u1b,q4 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 3600 -input_state EGF -input 60.84