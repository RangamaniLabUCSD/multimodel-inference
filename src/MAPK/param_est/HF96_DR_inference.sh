#!/bin/bash

# Kholodenko 2000
# complete
python inference_process_dose_response.py -model kholodenko_2000 -free_params K8,v10,v9,KI,MAPK_total,K10  -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -ncores 1 -savedir ../../../results/MAPK/param_est/HF96_synthetic_data/kholodenko_2000/ -input_state Input -ERK_states MAPK_PP -prior_family "[['LogNormal(mu=2.70805020110221)',['sigma']],['LogNormal(mu=-0.6931471805599453)',['sigma']], ['LogNormal(mu=-0.6931471805599453)',['sigma']], ['LogNormal(mu=2.1972245773362196)',['sigma']], ['LogNormal(mu=5.703782474656201)',['sigma']], ['LogNormal(mu=2.70805020110221)',['sigma']]]" -ss_method newton -newton_event_atol 1e-6 -newton_event_rtol 1e-6 --skip_prior_sample

# LEVCHENKO 2000
# RUNNING: synapse
python inference_process_dose_response.py -model levchenko_2000 -free_params a2,k10,RAFPase,k4,k6,kOn2,total_scaffold,a10,kOff3,k5,a9,a8,a6,d2,a7,d10,a1,k9,k2,d1,k3,MEKPase,kOff2,MAPKPase,k1 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 200 -ncores 1 -savedir ../../../results/MAPK/param_est/HF96_synthetic_data/levchenko_2000/ -input_state RAFact -EGF_conversion_factor 0.001 -ERK_states MAPKstarstar -prior_family "[['LogNormal(mu=-0.6931471805599453)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=-1.2039728043259361)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=2.302585092994046)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=1.6094379124341003)',['sigma']],['LogNormal(mu=-2.995732273553991)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=2.995732273553991)',['sigma']],['LogNormal(mu=1.6094379124341003)',['sigma']],['LogNormal(mu=2.302585092994046)',['sigma']],['LogNormal(mu=-0.6931471805599453)',['sigma']],['LogNormal(mu=2.995732273553991)',['sigma']],['LogNormal(mu=-0.916290731874155)',['sigma']],['LogNormal(mu=0.0)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=-0.916290731874155)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=-1.6094379124341005)',['sigma']],['LogNormal(mu=-2.995732273553991)',['sigma']],['LogNormal(mu=-1.2039728043259361)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']]]" -ss_method ode -event_atol 1e-10 -event_rtol 1e-10 -t1 100000 --skip_prior_sample

# Hornberg 2005
# complete: synapse
python inference_process_dose_response.py -model hornberg_2005 -free_params kd1,k6,k8,k18,k42,kd42,k44,kd52,kd45,kd47,kd48,kd49,kd50,k52,kd44,kd53,kd55,k56,kd56,kd57,k58,kd58 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 100 -savedir ../../../results/MAPK/param_est/HF96_synthetic_data/hornberg_2005/ -input_state c1 -EGF_conversion_factor 0.000000001 -ERK_states c59,c83 -prior_family "[['LogNormal(mu=-5.562282912382502)',['sigma']],['LogNormal(mu=-7.600902459542082)',['sigma']],['LogNormal(mu=-13.308692955595822)',['sigma']],['LogNormal(mu=-10.596634733096073)',['sigma']],['LogNormal(mu=-9.04482593349861)',['sigma']],['LogNormal(mu=-1.6094379124341003)',['sigma']],['LogNormal(mu=-10.845096092394574)',['sigma']],['LogNormal(mu=-3.4112477175156566)',['sigma']],['LogNormal(mu=1.252762968495368)',['sigma']],['LogNormal(mu=1.0647107369924282)',['sigma']],['LogNormal(mu=-0.2231435513142097)',['sigma']],['LogNormal(mu=-2.8682189532550315)',['sigma']],['LogNormal(mu=-0.6931471805599453)',['sigma']],['LogNormal(mu=-9.32575122348751)',['sigma']],['LogNormal(mu=-4.000854219134761)',['sigma']],['LogNormal(mu=2.772588722239781)',['sigma']],['LogNormal(mu=1.7404661748405046)',['sigma']],['LogNormal(mu=-10.658510136814161)',['sigma']],['LogNormal(mu=-0.5108256237659907)',['sigma']],['LogNormal(mu=-1.4024237430497744)',['sigma']],['LogNormal(mu=-11.695647101785523)',['sigma']],['LogNormal(mu=-0.6931471805599453)',['sigma']]]" -t1 12000 -ncores 1 -ss_method ode -event_atol 1e-10 -event_rtol 1e-10 --skip_prior_sample --skip_ss_check_func

# BIRTWISTLE 2007
# complete: synapse
python inference_process_dose_response.py -model birtwistle_2007 -free_params Kmf52,koff57,EGF_off,kcat90,koff91,koff40,koff88,koff95,kon91,kcat94,koff42,kon93,kcat92,koff46,kcon49,koff41,Vmaxr52,koff44,kon89,kon95 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 100 -ncores 1 -savedir ../../../results/MAPK/param_est/HF96_synthetic_data/birtwistle_2007/ -input_state E -ERK_states ERKstar,ERKstar_ERKpase -prior_family "[['LogNormal(mu=6.3015942750942955)',['sigma']],['LogNormal(mu=-0.7927465457827001)',['sigma']],['LogNormal(mu=-4.045554398052669)',['sigma']], ['LogNormal(mu=2.995917256443601)',['sigma']],['LogNormal(mu=4.60515318584359)',['sigma']],['LogNormal(mu=1.1330459209859989)',['sigma']],['LogNormal(mu=1.367493731656173)',['sigma']],['LogNormal(mu=4.6051931857235955)',['sigma']],['LogNormal(mu=-1.626584071269071)',['sigma']],['LogNormal(mu=-0.003405793134832821)',['sigma']],['LogNormal(mu=1.2583189340660492)',['sigma']],['LogNormal(mu=-1.6079390363103645)',['sigma']],['LogNormal(mu=-1.6074399097714274)',['sigma']],['LogNormal(mu=-0.655080979753489)',['sigma']],['LogNormal(mu=2.3004127351323884)',['sigma']],['LogNormal(mu=1.9528432026578095)',['sigma']],['LogNormal(mu=5.294697322086547)',['sigma']], ['LogNormal(mu=-0.8521429120504033)',['sigma']],['LogNormal(mu=-1.6109390385603677)',['sigma']],['LogNormal(mu=-1.6129440517633882)',['sigma']]]" -t1 10000 --skip_prior_sample

# Orton 2009
# complete
python inference_process_dose_response.py -model orton_2009 -free_params km_Erk_Activation,k1_C3G_Deactivation,km_Erk_Deactivation,k1_P90Rsk_Deactivation,k1_Sos_Deactivation -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 250 -ncores 1 -savedir ../../../results/MAPK/param_est/HF96_synthetic_data/orton_2009/ -input_state EGF -EGF_conversion_factor 602214 -ERK_states ErkActive -prior_family "[['LogNormal(mu=13.822823751258499)',['sigma']],['LogNormal(mu=0.9162907318741551)',['sigma']],['LogNormal(mu=15.067270166119108)',['sigma']],['LogNormal(mu=-5.298317366548036)',['sigma']],['LogNormal(mu=0.9162907318741551)',['sigma']]]" -ss_method ode --skip_prior_sample -t1 1400


# von Kriegsheim 2009
# complete
python inference_process_dose_response.py -model vonKriegsheim_2009 -free_params  k42,k37,k4,k27,k30,k5,k28,k68,k29,k41,k25,k7,k13,k2,k40,k32,k10,k34 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 100 -savedir ../../../results/MAPK/param_est/HF96_synthetic_data/vonKriegsheim_2009/ -input_state EGF -EGF_conversion_factor 6.048 -ERK_states ppERK,ppERK_15,ppERKn -prior_family "[['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=0.0)',['sigma']],['LogNormal(mu=-4.615220521841593)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=-4.605170185988091)',['sigma']],['LogNormal(mu=-8.517193191416238)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=-7.600902459542082)',['sigma']],['LogNormal(mu=-1.760260802168684)',['sigma']],['LogNormal(mu=-4.605170185988091)',['sigma']],['LogNormal(mu=-3.218875824868201)',['sigma']],['LogNormal(mu=-2.6621212692138103)',['sigma']],['LogNormal(mu=-2.1386266695649785)',['sigma']],['LogNormal(mu=-4.731867839034049)',['sigma']],['LogNormal(mu=0.22298353851284428)',['sigma']],['LogNormal(mu=-0.4177907577988013)',['sigma']],['LogNormal(mu=-1.458479730351939)',['sigma']],['LogNormal(mu=-4.605170185988091)',['sigma']]]" -t1 5400 -ncores 1 -ss_method ode --skip_prior_sample

# Shin 2014
# complete: martini
python inference_process_dose_response.py -model shin_2014 -free_params kc47,kc43,kd39,kc45,ERK_tot,ki39,kc41 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 4000 -savedir ../../../results/MAPK/param_est/HF96_synthetic_data/shin_2014/ -input_state EGF -EGF_conversion_factor 0.001 -ERK_states pp_ERK -prior_family "[['LogNormal(mu=-0.9403273097023963)',['sigma']],['LogNormal(mu=1.539659017826184)',['sigma']],['LogNormal(mu=-1.822631132895142)',['sigma']],['LogNormal(mu=-2.677715001306137)',['sigma']],['LogNormal(mu=-1.487220279709851)',['sigma']],['LogNormal(mu=-7.1670442769327005)',['sigma']],['LogNormal(mu=3.9265174515785985)',['sigma']]]" -ss_method newton -newton_event_atol 1e-6 -newton_event_rtol 1e-6 --skip_prior_sample

# Ryu 2015
# complete: martini
python inference_process_dose_response.py -model ryu_2015 -free_params D2,T_dusp,K_dusp,K2,dusp_ind -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -ncores 1 -savedir ../../../results/MAPK/param_est/HF96_synthetic_data/ryu_2015/ -input_state EGF -EGF_conversion_factor 6.048 -ERK_states ERK_star -prior_family "[['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=4.499809670330265)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=0.0)',['sigma']],['LogNormal(mu=1.791759469228055)',['sigma']]]" -t1 540 -ss_method newton --skip_prior_sample

# Kochańczyk 2017
# Complete: martini
python inference_process_dose_response.py -model kochanczyk_2017 -free_params q1,q3,q2,u3,d1,q6,u2b,u1a,d2,u2a,q4 -data_file ../../../results/MAPK/HF_96_synthetic_data.csv -nsamples 1000 -ncores 1 -savedir ../../../results/MAPK/param_est/HF96_synthetic_data/kochanczyk_2017/ -input_state EGF -EGF_conversion_factor 6048 -ERK_states ERKSPP -prior_family "[['LogNormal(mu=-4.605170185988091)',['sigma']],['LogNormal(mu=-8.111728083308074)',['sigma']], ['LogNormal(mu=-4.605170185988091)',['sigma']], ['LogNormal(mu=-4.605170185988091)',['sigma']], ['LogNormal(mu=-4.605170185988091)',['sigma']], ['LogNormal(mu=-8.111728083308074)',['sigma']], ['LogNormal(mu=0.0)',['sigma']], ['LogNormal(mu=-4.605170185988091)',['sigma']], ['LogNormal(mu=-4.605170185988091)',['sigma']],['LogNormal(mu=0.0)',['sigma']],['LogNormal(mu=-8.111728083308074)',['sigma']]]" -t1 10800 -ss_method newton -newton_event_atol 1e-6 -newton_event_rtol 1e-6 --skip_prior_sample