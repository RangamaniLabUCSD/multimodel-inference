#!/usr/bin/bash

# Huang Ferrell 1996
# XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py huang_ferrell_1996 -analyze_params MKK_tot,a3,k7,a4,d3,d10,d2,a2,d6,a8,a1,E2_tot,a7,a9,k5,a6,d1,d9,d5,k8,d8,k4,k6,k9,k3,d7,a10,MAPK_tot,k2,d4,a5,MKKK_tot,k10,MKKPase_tot,k1 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 3000 -input_param E1_tot --full_trajectory -lower 1e-2 -upper 1e2 -ERK_state_indices 13,14

# Kholodenko 2000
# XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py kholodenko_2000 -analyze_params K8,v10,v9,K7,K9,KI,MAPK_total,K10 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 2400 -input_state Input --full_trajectory -ERK_state_indices -1 -lower 1e-2 -upper 1e2

# # Birghtman and Fell 2000
# # XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py brightman_fell_2000 -analyze_params kn14,K_24,kn16,V_26,kn1,k3,V_24,kn12,k15,k_13,kn7,kn11,K_25,k2_4,K_26,k17,K_23,DT -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 360 -input_state L -input 60221 --full_trajectory -lower 1e-2 -upper 1e2

# LEVCHENKO 2000
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py levchenko_2000 -analyze_params kOff1,a2,kOn1,k10,RAFPase,k4,k6,kOff4,kOn2,total_scaffold,d7,d9,d3,a3,a10,kOff3,k8,k5,a9,a8,a5,d8,d5,a6,k7,d2,a7,d10,a1,a4,k9,d6,k2,d1,k3,MEKPase,kOff2,MAPKPase,d4,k1 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 3000 -input_state RAFact -input 0.0001 -ERK_state_indices 16 --full_trajectory -lower 1e-2 -upper 1e2

# Hatakeyama 2003
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py hatakeyama_2003 -analyze_params k21,k20,kb29,kf9,kb24,kb3,kb23,kf25,kb1,kb2,kf24,kb7,kf8,k19,kf34,k22,kb5,kf6,kf3,kb6 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 7200 -input_state HRG --full_trajectory -ERK_state_indices 32 -lower 1e-2 -upper 1e2

# Hornberg 2005
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py hornberg_2005 -analyze_params kd1,k2,kd2,k3,kd3,k4,kd4,k5,kd5,k6,kd6,k8,kd8,k10b,kd10,k13,kd13,k15,kd15,k16,kd63,k17,kd17,k18,kd18,k19,kd19,k20,kd20,k21,kd21,k22,kd22,k23,kd23,kd24,k25,kd25,k28,kd28,k29,kd29,k32,kd32,k33,kd33,k34,kd34,k35,kd35,k36,kd36,k37,kd37,k40,kd40,k41,kd41,k42,kd42,k43,kd43,k44,kd52,k45,kd45,k47,kd47,k48,kd48,k49,kd49,k50,kd50,k52,kd44,k53,kd53,k55,kd55,k56,kd56,k57,kd57,k58,kd58,k60,kd60,k61,kd61,k126,kd126,k127,kd127 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 12000 -input_state c1 -input 1e-9 --full_trajectory -ERK_state_indices 58,82 -lower 1e-2 -upper 1e2

# Orton 2009
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py orton_2009 -analyze_params km_Erk_Activation,k1_C3G_Deactivation,km_Erk_Deactivation,k1_Akt_Deactivation,k1_P90Rsk_Deactivation,k1_PI3K_Deactivation,k1_Sos_Deactivation -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 1440 -input_state EGF -input 60221 --full_trajectory -lower 1e-2 -upper 1e2 -ERK_state_indices 15

# von Kriegsheim 2009
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py vonKriegsheim_2009 -analyze_params k42,k37,k4,k27,k45,k30,k43,k48,k5,k14,k28,k39,k46,k63,k68,k55,k29,k41,k25,k7,k13,k2,k40,k6,k18,k56,k32,k38,k10,k34 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 3600 -input_state EGF -input 0.6048 --full_trajectory -ERK_state_indices 26,28,29 --full_trajectory -lower 1e-2 -upper 1e2

# Shin 2014
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py shin_2014 -analyze_params kc47,kc43,kd39,kc45,ERK_tot,ki39,kc41 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 1080 -input_state EGF -input 1e-4 --full_trajectory -lower 1e-2 -upper 1e2 -ERK_state_indices -1

# Ryu 2015
# XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py ryu_2015 -analyze_params D2,T_dusp,K_dusp,K2,dusp_ind -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 540 -input_state EGF -input 0.6048 --full_trajectory -lower 1e-2 -upper 1e2 -ERK_state_indices 10

# Kochańczyk 2017
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py kochanczyk_2017 -analyze_params k3,q1,q3,q2,u3,d1,q6,u2b,u1a,d2,u2a,q5,u1b,q4 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 7200 -input_state EGF -input 604.8 --full_trajectory -lower 1e-2 -upper 1e2 -ERK_state_indices 24

# BIRTWISTLE 2007
# # We run this model without full trajectories, because we run out of GPU memory when trying to store all of the trajectories
# XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py birtwistle_2007 -analyze_params koff67,Kmf52,koff57,EGF_off,koff29,koff31,koff89,kcat90,koff91,koff40,koff61,koff77,koff21,koff45,koff68,koff24,kf12,b98,HRGoff_4,VeVc,koff4,koff78,koff26,koff88,koff22,a98,koff8,koff28,kf14,koff76,koff25,koff73,koff95,koff59,koff66,koff65,Kmr52,koff33,kf15,koff6,kon91,kf13,kcat94,koff30,koff42,kon93,koff70,koff58,kf11,koff19,kcat96,koff36,kcat92,koff17,HRGoff_3,koff46,koff71,koff34,koff20,koff72,kcon49,kf63,kdeg,koff93,koff35,koff5,koff18,koff41,koff32,Vmaxr52,koff74,koff75,koff27,koff43,koff62,koff23,koff37,koff44,koff80,koff60,kf48,koff69,koff16,kf64,koff9,kon89,kf10,koff79,kon95,koff7 -n_samples 256 -savedir ../../../results/MAPK/gsa/ -max_time 10000 -input_state E -ERK_state_indices 75,115 -lower 1e-2 -upper 1e2

# # BIRTWISTLE 2007 -- FEWER SAMPLES FOR PLOTTING!
# # We run this model without full trajectories, because we run out of GPU memory when trying to store all of the trajectories
# XLA_PYTHON_CLIENT_PREALLOCATE=false python gsa_sample.py birtwistle_2007 -analyze_params koff67,Kmf52,koff57,EGF_off,koff29,koff31,koff89,kcat90,koff91,koff40,koff61,koff77,koff21,koff45,koff68,koff24,kf12,b98,HRGoff_4,VeVc,koff4,koff78,koff26,koff88,koff22,a98,koff8,koff28,kf14,koff76,koff25,koff73,koff95,koff59,koff66,koff65,Kmr52,koff33,kf15,koff6,kon91,kf13,kcat94,koff30,koff42,kon93,koff70,koff58,kf11,koff19,kcat96,koff36,kcat92,koff17,HRGoff_3,koff46,koff71,koff34,koff20,koff72,kcon49,kf63,kdeg,koff93,koff35,koff5,koff18,koff41,koff32,Vmaxr52,koff74,koff75,koff27,koff43,koff62,koff23,koff37,koff44,koff80,koff60,kf48,koff69,koff16,kf64,koff9,kon89,kf10,koff79,kon95,koff7 -n_samples 16 -savedir ../../../results/MAPK/gsa/plotting_samples_ -max_time 10000 -input_state E -ERK_state_indices 75,115 -lower 1e-2 -upper 1e2 --full_trajectory
