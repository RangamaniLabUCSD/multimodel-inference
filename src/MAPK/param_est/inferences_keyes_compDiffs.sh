#!/bin/bash

# Orton 2009
# TODO: -
# all diff
python inference_process_location_diff_traj.py -model orton_2009 -pymc_model all_diff -free_params km_Erk_Activation,k1_C3G_Deactivation,km_Erk_Deactivation,k1_P90Rsk_Deactivation,k1_Sos_Deactivation -Rap1_state Rap1Inactive -savedir ../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/orton_2009/all_diff/all_diff_ -input_state EGF -EGF_conversion_factor 602214 -ERK_states ErkActive -prior_family "[['LogNormal(mu=13.822823751258499)',['sigma']],['LogNormal(mu=0.9162907318741551)',['sigma']],['LogNormal(mu=15.067270166119108)',['sigma']],['LogNormal(mu=-5.298317366548036)',['sigma']],['LogNormal(mu=0.9162907318741551)',['sigma']]]" -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4

# TODO: -
# Rap1 diff
python inference_process_location_diff_traj.py -model orton_2009 -pymc_model Orton_2009_Rap1_diff -free_params km_Erk_Activation,k1_C3G_Deactivation,km_Erk_Deactivation,k1_P90Rsk_Deactivation,k1_Sos_Deactivation -Rap1_state Rap1Inactive -savedir ../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/orton_2009/Rap1_diff/Rap1_diff_ -input_state EGF -EGF_conversion_factor 602214 -ERK_states ErkActive -prior_family "[['LogNormal(mu=13.822823751258499)',['sigma']],['LogNormal(mu=0.9162907318741551)',['sigma']],['LogNormal(mu=15.067270166119108)',['sigma']],['LogNormal(mu=-5.298317366548036)',['sigma']],['LogNormal(mu=0.9162907318741551)',['sigma']]]" -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4

# TODO: -
# p90RsKSoS diff
python inference_process_location_diff_traj.py -model orton_2009 -pymc_model Orton_2009_p90RsKSoS_diff -free_params km_Erk_Activation,k1_C3G_Deactivation,km_Erk_Deactivation,k1_P90Rsk_Deactivation,k1_Sos_Deactivation -Rap1_state Rap1Inactive -savedir ../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/orton_2009/p90RsKSoS_diff/p90RsKSoS_diff_ -input_state EGF -EGF_conversion_factor 602214 -ERK_states ErkActive -prior_family "[['LogNormal(mu=13.822823751258499)',['sigma']],['LogNormal(mu=0.9162907318741551)',['sigma']],['LogNormal(mu=15.067270166119108)',['sigma']],['LogNormal(mu=-5.298317366548036)',['sigma']],['LogNormal(mu=0.9162907318741551)',['sigma']]]" -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4

# TODO:
# p90RskSoS_Rap1 diff
python inference_process_location_diff_traj.py -model orton_2009 -pymc_model Orton_2009_p90RsKSoS_Rap1_diff -free_params km_Erk_Activation,k1_C3G_Deactivation,km_Erk_Deactivation,k1_P90Rsk_Deactivation,k1_Sos_Deactivation -Rap1_state Rap1Inactive -savedir ../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/orton_2009/p90RsKSoS_Rap1_diff/p90RsKSoS_Rap1_diff_ -input_state EGF -EGF_conversion_factor 602214 -ERK_states ErkActive -prior_family "[['LogNormal(mu=13.822823751258499)',['sigma']],['LogNormal(mu=0.9162907318741551)',['sigma']],['LogNormal(mu=15.067270166119108)',['sigma']],['LogNormal(mu=-5.298317366548036)',['sigma']],['LogNormal(mu=0.9162907318741551)',['sigma']]]" -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4

# Shin 2014
# TODO: -
# all diff
python inference_process_location_diff_traj.py -model shin_2014_Rap1 -pymc_model all_diff -free_params kc47,kc43,kd39,kc45,ERK_tot,ki39,kc41 -Rap1_state Rap1 -ERK_states pp_ERK -savedir ../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/shin_2014/all_diff/all_diff_ -input_state EGF -EGF_conversion_factor 0.001 -ERK_states pp_ERK -prior_family "[['LogNormal(mu=-0.9403273097023963)',['sigma']],['LogNormal(mu=1.539659017826184)',['sigma']],['LogNormal(mu=-1.822631132895142)',['sigma']],['LogNormal(mu=-2.677715001306137)',['sigma']],['LogNormal(mu=-1.487220279709851)',['sigma']],['LogNormal(mu=-7.1670442769327005)',['sigma']],['LogNormal(mu=3.9265174515785985)',['sigma']]]" -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4

TODO: -
# Rap1 diff
python inference_process_location_diff_traj.py -model shin_2014_Rap1 -pymc_model Shin_2014_Rap1_diff -free_params kc47,kc43,kd39,kc45,ERK_tot,ki39,kc41 -Rap1_state Rap1 -ERK_states pp_ERK -savedir ../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/shin_2014/Rap1_diff/Rap1_diff_ -input_state EGF -EGF_conversion_factor 0.001 -ERK_states pp_ERK -prior_family "[['LogNormal(mu=-0.9403273097023963)',['sigma']],['LogNormal(mu=1.539659017826184)',['sigma']],['LogNormal(mu=-1.822631132895142)',['sigma']],['LogNormal(mu=-2.677715001306137)',['sigma']],['LogNormal(mu=-1.487220279709851)',['sigma']],['LogNormal(mu=-7.1670442769327005)',['sigma']],['LogNormal(mu=3.9265174515785985)',['sigma']]]" -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4

# TODO: -
# Sos diff
python inference_process_location_diff_traj.py -model shin_2014_Rap1 -pymc_model Shin_2014_Sos_diff -free_params kc47,kc43,kd39,kc45,ERK_tot,ki39,kc41 -Rap1_state Rap1 -ERK_states pp_ERK -savedir ../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/shin_2014/Sos_diff/Sos_diff_ -input_state EGF -EGF_conversion_factor 0.001 -ERK_states pp_ERK -prior_family "[['LogNormal(mu=-0.9403273097023963)',['sigma']],['LogNormal(mu=1.539659017826184)',['sigma']],['LogNormal(mu=-1.822631132895142)',['sigma']],['LogNormal(mu=-2.677715001306137)',['sigma']],['LogNormal(mu=-1.487220279709851)',['sigma']],['LogNormal(mu=-7.1670442769327005)',['sigma']],['LogNormal(mu=3.9265174515785985)',['sigma']]]" -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4

# TODO: -
# Sos_Rap1 diff
python inference_process_location_diff_traj.py -model shin_2014_Rap1 -pymc_model Shin_2014_Sos_Rap1_diff -free_params kc47,kc43,kd39,kc45,ERK_tot,ki39,kc41 -Rap1_state Rap1 -ERK_states pp_ERK -savedir ../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/shin_2014/Sos_Rap1_diff/Sos_Rap1_diff_ -input_state EGF -EGF_conversion_factor 0.001 -ERK_states pp_ERK -prior_family "[['LogNormal(mu=-0.9403273097023963)',['sigma']],['LogNormal(mu=1.539659017826184)',['sigma']],['LogNormal(mu=-1.822631132895142)',['sigma']],['LogNormal(mu=-2.677715001306137)',['sigma']],['LogNormal(mu=-1.487220279709851)',['sigma']],['LogNormal(mu=-7.1670442769327005)',['sigma']],['LogNormal(mu=3.9265174515785985)',['sigma']]]" -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4

# Ryu 2015
# TODO: -
# all diff
python inference_process_location_diff_traj.py -model ryu_2015_Rap1 -pymc_model all_diff -free_params D2,T_dusp,K_dusp,K2,dusp_ind -Rap1_state Rap1 -savedir ../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/ryu_2015/all_diff/all_diff_ -input_state EGF -EGF_conversion_factor 6.048 -ERK_states ERK_star -prior_family "[['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=4.499809670330265)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=0.0)',['sigma']],['LogNormal(mu=1.791759469228055)',['sigma']]]" -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4

# TODO: -
# Rap1 diff
python inference_process_location_diff_traj.py -model ryu_2015_Rap1 -pymc_model Ryu_2015_Rap1_diff -free_params D2,T_dusp,K_dusp,K2,dusp_ind -Rap1_state Rap1 -savedir ../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/ryu_2015/Rap1_diff/Rap1_diff_ -input_state EGF -EGF_conversion_factor 6.048 -ERK_states ERK_star -prior_family "[['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=4.499809670330265)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=0.0)',['sigma']],['LogNormal(mu=1.791759469228055)',['sigma']]]"  -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4

# TODO: -
# DUSP diff
python inference_process_location_diff_traj.py -model ryu_2015_Rap1 -pymc_model Ryu_2015_DUSP_diff -free_params D2,T_dusp,K_dusp,K2,dusp_ind -Rap1_state Rap1 -savedir ../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/ryu_2015/DUSP_diff/DUSP_diff_ -input_state EGF -EGF_conversion_factor 6.048 -ERK_states ERK_star -prior_family "[['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=4.499809670330265)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=0.0)',['sigma']],['LogNormal(mu=1.791759469228055)',['sigma']]]"  -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4

# TODO: -
# DUSP_Rap1 diff
python inference_process_location_diff_traj.py -model ryu_2015_Rap1 -pymc_model Ryu_2015_DUSP_Rap1_diff -free_params D2,T_dusp,K_dusp,K2,dusp_ind -Rap1_state Rap1 -savedir ../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/ryu_2015/DUSP_Rap1_diff/DUSP_Rap1_diff_ -input_state EGF -EGF_conversion_factor 6.048 -ERK_states ERK_star -prior_family "[['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=4.499809670330265)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=0.0)',['sigma']],['LogNormal(mu=1.791759469228055)',['sigma']]]"  -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4