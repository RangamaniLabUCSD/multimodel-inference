#!/bin/zsh

#!/usr/bin/bash

# Kholodenko 2000
python inference_process.py -model kholodenko_2000 -free_params  -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 2000 -savedir ../../../results/MAPK/param_est/ -t1 60 -input_state L -EGF_conversion_factor 602214 -ERK_state_indices 26

# Birghtman and Fell 2000
python inference_process.py -model brightman_fell_2000 -free_params kn14,K_24,kn16,V_26,V_24,kn12,k15,k_13,K_25,K_26,k17 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 2000 -savedir ../../../results/MAPK/param_est/ -t1 60 -input_state L -EGF_conversion_factor 602214 -ERK_state_indices 26

# LEVCHENKO 2000

# Hatakeyama 2003
python inference_process.py -model hatakeyama_2003 -free_params k21, kf9, kb2, k19, kf3 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 2000 -savedir ../../../results/MAPK/param_est/

# Hornberg 2005

# BIRTWISTLE 2007

# Orton 2009
python inference_process.py -model orton_2009 -free_params km_Erk_Activation,k1_C3G_Deactivation,km_Erk_Deactivation,k1_P90Rsk_Deactivation -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 2000 -savedir ../../../results/MAPK/param_est/

# von Kriegsheim 2009
python inference_process.py -model vonKriegsheim_2009 -free_params k42,k4,k30,k68,k29,k32,k34 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 2000 -savedir ../../../results/MAPK/param_est/

# Shin 2014
python inference_process.py -model shin_2014 -free_params kc47,kc43,kd39,kc45,ERK_tot,ki39,kc41 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 2000 -savedir ../../../results/MAPK/param_est/

# Ryu 2015

# Kochańczyk 2017
