import pdb
from os import environ
environ['OMP_NUM_THREADS'] = '1'
#environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import sys
import argparse
import pymc as pm
from pytensor.tensor import max as pt_max

sys.path.append("../models/")
from orton_2009 import *
# Rap1 models
from shin_2014_Rap1 import *
from ryu_2015_Rap1 import *
from vonKriegsheim_2009_Rap1 import *
# from kochanczyk_2017_rap1 import *

sys.path.append("../")
from utils import *
import os

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


# need a posterior predictive function that can deal with the fact that we now have multiple outputs in the llike function
def plot_trajectories(mapk_model_name, times, inputs, savedir, type,
    llike_CYTO, llike_PM, llike_CYTO_Rap1KD, llike_PM_Rap1KD, \
    data_CYTO, data_PM, data_CYTO_RAP1i, data_PM_RAP1i,
    data_std_CYTO, data_std_PM, data_std_CYTO_RAP1i, data_std_PM_RAP1i, 
    seed=np.random.default_rng(seed=123)):
    """ Creates prior predictive samples plot of the stimulus response curve.

    Type should be 'prior' or 'posterior'.
    """

    data = {'CYTO':{'llike':llike_CYTO, 'data':data_CYTO, 'data_std':data_std_CYTO},
            'PM':{'llike':llike_PM, 'data':data_PM, 'data_std':data_std_PM},
            'CYTO_Rap1KD':{'llike':llike_CYTO_Rap1KD, 'data':data_CYTO_RAP1i, 'data_std':data_std_CYTO_RAP1i},
            'PM_Rap1KD':{'llike':llike_PM_Rap1KD, 'data':data_PM_RAP1i, 'data_std':data_std_PM_RAP1i}}

    fig_ax = []
    for key in data.keys():
        samples = data[key]['llike']

        if samples.ndim == 3:
            nchains,nsamples,ntime=samples.shape
            samples = np.reshape(samples, (nchains*nsamples, ntime))
        else:
            nsamples,ntime=samples.shape
        
        fig, ax = get_sized_fig_ax(3.0, 1.5)

        # plot the samples
        tr_dict =  {'run':{}, 'timepoint':{}, 'ERK_act':{}}
        names = ['run'+str(i) for i in range(nsamples)]
        idxs = np.linspace(0, (nsamples*times.shape[0]-2), nsamples*times.shape[0]-1)
        cnt = 0
        for i in range(nsamples):
            for j in range(times.shape[0]-1):
                    tr_dict['run'][int(idxs[cnt])] = names[i]
                    tr_dict['timepoint'][int(idxs[cnt])] = times[j+1]
                    tr_dict['ERK_act'][int(idxs[cnt])] = samples[i,j]
                    cnt += 1
        tr_df = pd.DataFrame.from_dict(tr_dict)

        sns.lineplot(data=tr_df,
                    x='timepoint',
                    y='ERK_act',
                    color='c',
                    legend=True,
                    errorbar=('pi', 95), # percentile interval form 2.5th to 97.5th
                    ax=ax)

        # ax.set_ylim([0.0, 1.0])
        ax.set_xlim([0.0, max(times)])

        # plot data
        data_downsample = 3
        ax.errorbar(times[::data_downsample], np.squeeze(data[key]['data'])[::data_downsample], yerr=data[key]['data_std'][::data_downsample], fmt='o', linewidth=1.0, markersize=0.1, color='k')

        ax.set_xlabel('time (min)')
        ax.set_ylabel('ERK activity')

        # save the figure
        fig.savefig(savedir + mapk_model_name + '_' + key + '_' + type + '_predictive.pdf', 
                    bbox_inches='tight', transparent=True)

        # save the samples
        np.save(savedir + mapk_model_name + '_' + key + '_' + type + '_predictive_samples.npy', samples)

        fig_ax.append((fig, ax))
    
    return fig_ax

def build_pymc_model_local(prior_param_dict, data, y0, y0_Rap1KD,
                    output_states, max_time, model_dfrx_ode, model_func=None, 
                    simulator=ERK_stim_response, data_sigma=0.1):
    """ Builds a pymc model object for the MAPK models.

    Constructs priors for the model, and uses the ERK_stim_response function to 
    generate the stimulus response function and likelihood.
    
    If model is None, the function will use the default model. If a model is s
    pecified, it will use that model_func function to create a PyMC model.

    This is different than the build_pymc_model function in that it creates PyTensor Ops for
    the case with the Rap1 active and inactive which is defined by different initial conditions.
    """

    ####### SOL_OP for the model with Rap1 #######
    # Create jax functions to solve # 
    def sol_op_jax(*params):
        pred, _ = simulator(params, model_dfrx_ode, max_time, y0, output_states)
        return jnp.vstack((pred))

    # get the jitted versions
    sol_op_jax_jitted = jax.jit(sol_op_jax)

    
    # Create pytensor Op and register with jax # 
    class StimRespOp(Op):
        def make_node(self, *inputs):
            inputs = [pt.as_tensor_variable(inp) for inp in inputs]
            outputs = [pt.matrix()]
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            result = sol_op_jax_jitted(*inputs)
            if jnp.any(jnp.isnan(jnp.array(result))):
                print('Warning: NaNs in the result. Setting to zeros.')
                result = jnp.zeros_like(result)
            outputs[0][0] = np.asarray(result, dtype="float64")
        
        def grad(self, inputs, output_grads):
            raise NotImplementedError("PyTensor gradient of StimRespOp not implemented")


    # construct Ops and register with jax_funcify
    sol_op = StimRespOp()

    @jax_funcify.register(StimRespOp)
    def sol_op_jax_funcify(op, **kwargs):
        return sol_op_jax
    
    ####### SOL_OP for the model w/0 Rap1 #######
    # Create jax functions to solve # 
    def sol_op_jax_Rap1KD(*params):
        pred, _ = simulator(params, model_dfrx_ode, max_time, y0_Rap1KD, output_states)
        return jnp.vstack((pred))

    # get the jitted versions
    sol_op_jax_jitted_Rap1KD = jax.jit(sol_op_jax_Rap1KD)

    
    # Create pytensor Op and register with jax # 
    class StimRespOp_Rap1KD(Op):
        def make_node(self, *inputs):
            inputs = [pt.as_tensor_variable(inp) for inp in inputs]
            outputs = [pt.matrix()]
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            result = sol_op_jax_jitted_Rap1KD(*inputs)
            if jnp.any(jnp.isnan(jnp.array(result))):
                print('Warning: NaNs in the result. Setting to zeros.')
                result = jnp.zeros_like(result)
            outputs[0][0] = np.asarray(result, dtype="float64")
        
        def grad(self, inputs, output_grads):
            raise NotImplementedError("PyTensor gradient of StimRespOp not implemented")


    # construct Ops and register with jax_funcify
    sol_op_Rap1KD = StimRespOp_Rap1KD()

    @jax_funcify.register(StimRespOp_Rap1KD)
    def sol_op_Rap1KD_jax_funcify(op, **kwargs):
        return sol_op_jax_Rap1KD
    
    
    # Construct the PyMC model # 
    model = model_func(prior_param_dict, sol_op, sol_op_Rap1KD, data, data_sigma)

    return model

################################################################################
#### PyMC Model Functions ####
################################################################################
# all diff - works for all models
# Orton 2009 only Rap1 diff
# Orton 2009 on Sos Feedback diff
# Orton 2009 Rap1 and Sos Feedback diff

def all_diff(prior_param_dict, sol_op, sol_op_Rap1KD, data_CYTO, data_PM, data_std_CYTO, data_std_PM,
             data_CYTO_Rap1KD, data_PM_Rap1KD, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD):
    """ PyMC model for Orton 2009 model with all parameters different.
    """
    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors_CYTO = []
        priors_PM = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            # need to create diff priors for each compartment
            info_split = value.split('",')
            prior_CYTO = eval(info_split[0] + '_CYTO",' + info_split[1])
            prior_PM = eval(info_split[0] + '_PM",' + info_split[1])
            priors_CYTO.append(prior_CYTO)
            priors_PM.append(prior_PM)

        # predict full Rap1
        CYTO = pm.Deterministic("CYTO", sol_op(*priors_CYTO))
        PM = pm.Deterministic("PM", sol_op(*priors_PM))
        # Rap1 inhib
        CYTO_Rap1KD = pm.Deterministic("CYTO_Rap1KD", sol_op_Rap1KD(*priors_CYTO))
        PM_Rap1KD = pm.Deterministic("PM_Rap1KD", sol_op_Rap1KD(*priors_PM))
        # normalization factors
        CYTO_norm = pm.Deterministic("CYTO_norm", pt_max(CYTO))
        PM_norm = pm.Deterministic("PM_norm", pt_max(PM))

        # normalized predictions are the actual predictions divided by the max value
        prediction_CYTO = pm.Deterministic("prediction_CYTO", CYTO/CYTO_norm)
        prediction_PM = pm.Deterministic("prediction_PM", PM/PM_norm)
        prediction_CYTO_Rap1KD = pm.Deterministic("prediction_CYTO_Rap1KD", CYTO_Rap1KD/CYTO_norm)
        prediction_PM_Rap1KD = pm.Deterministic("prediction_PM_Rap1KD", PM_Rap1KD/PM_norm) 

        # assume a normal model for the data
        llike_CYTO = pm.Normal("llike_CYTO", mu=prediction_CYTO, sigma=data_std_CYTO, observed=data_CYTO)
        llike_PM = pm.Normal("llike_PM", mu=prediction_PM, sigma=data_std_PM, observed=data_PM)
        llike_CYTO_Rap1KD = pm.Normal("llike_CYTO_Rap1KD", mu=prediction_CYTO_Rap1KD, sigma=data_std_CYTO_Rap1KD, observed=data_CYTO_Rap1KD)
        llike_PM_Rap1KD = pm.Normal("llike_PM_Rap1KD", mu=prediction_PM_Rap1KD, sigma=data_std_PM_Rap1KD, observed=data_PM_Rap1KD)

    return model

def Orton_2009_Rap1_diff(prior_param_dict, sol_op, sol_op_Rap1KD, data_CYTO, data_PM, data_std_CYTO, data_std_PM,
             data_CYTO_Rap1KD, data_PM_Rap1KD, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD):
    # index of the parameter to change
    k1_C3G_Deact_idx = list(prior_param_dict.keys()).index('k1_C3G_Deactivation')
    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            if key != 'k1_C3G_Deactivation':
                prior = eval(value)
                priors.append(prior)
            else:
                priors.append(None)
                k1_C3G_Deactivation_CYTO = pm.LogNormal("k1_C3G_Deactivation_CYTO",sigma=3.5244297004803578, mu=0.9162907318741551)
                k1_C3G_Deactivation_PM = pm.LogNormal("k1_C3G_Deactivation_PM",sigma=3.5244297004803578, mu=0.9162907318741551)
    
        # predict dose response
        CYTO = sol_op(*priors[0:k1_C3G_Deact_idx], k1_C3G_Deactivation_CYTO, *priors[k1_C3G_Deact_idx+1:])
        PM = sol_op(*priors[0:k1_C3G_Deact_idx], k1_C3G_Deactivation_PM, *priors[k1_C3G_Deact_idx+1:])
        CYTO_Rap1KD = sol_op_Rap1KD(*priors[0:k1_C3G_Deact_idx], k1_C3G_Deactivation_CYTO, *priors[k1_C3G_Deact_idx+1:])
        PM_Rap1KD = sol_op_Rap1KD(*priors[0:k1_C3G_Deact_idx], k1_C3G_Deactivation_PM, *priors[k1_C3G_Deact_idx+1:])

         # normalization factors
        CYTO_norm = pm.Deterministic("CYTO_norm", pt_max(CYTO))
        PM_norm = pm.Deterministic("PM_norm", pt_max(PM))

        # normalized predictions are the actual predictions divided by the max value
        prediction_CYTO = pm.Deterministic("prediction_CYTO", CYTO/CYTO_norm)
        prediction_PM = pm.Deterministic("prediction_PM", PM/PM_norm)
        prediction_CYTO_Rap1KD = pm.Deterministic("prediction_CYTO_Rap1KD", CYTO_Rap1KD/CYTO_norm)
        prediction_PM_Rap1KD = pm.Deterministic("prediction_PM_Rap1KD", PM_Rap1KD/PM_norm) 

        # assume a normal model for the data
        llike_CYTO = pm.Normal("llike_CYTO", mu=prediction_CYTO, sigma=data_std_CYTO, observed=data_CYTO)
        llike_PM = pm.Normal("llike_PM", mu=prediction_PM, sigma=data_std_PM, observed=data_PM)
        llike_CYTO_Rap1KD = pm.Normal("llike_CYTO_Rap1KD", mu=prediction_CYTO_Rap1KD, sigma=data_std_CYTO_Rap1KD, observed=data_CYTO_Rap1KD)
        llike_PM_Rap1KD = pm.Normal("llike_PM_Rap1KD", mu=prediction_PM_Rap1KD, sigma=data_std_PM_Rap1KD, observed=data_PM_Rap1KD)

    return model

def Orton_2009_p90RsKSoS_diff(prior_param_dict, sol_op, sol_op_Rap1KD, data_CYTO, data_PM, data_std_CYTO, data_std_PM,
             data_CYTO_Rap1KD, data_PM_Rap1KD, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD):
    # index of the parameter to change
    k1_P90Rsk_idx = list(prior_param_dict.keys()).index('k1_P90Rsk_Deactivation')
    k1_Sos_idx = list(prior_param_dict.keys()).index('k1_Sos_Deactivation')
    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            if key not in ['k1_P90Rsk_Deactivation', 'k1_Sos_Deactivation']:
                prior = eval(value)
                priors.append(prior)
            else:
                print(key)
                priors.append(None)
            
        k1_P90Rsk_Deactivation_CYTO = pm.LogNormal("k1_P90Rsk_Deactivation_CYTO",sigma=3.5244297004803578, mu=-5.298317366548036)
        k1_P90Rsk_Deactivation_PM = pm.LogNormal("k1_P90Rsk_Deactivation_PM",sigma=3.5244297004803578, mu=-5.298317366548036)
        k1_Sos_Deactivation_CYTO = pm.LogNormal("k1_Sos_Deactivation_CYTO",sigma=3.5244297004803578, mu=0.9162907318741551)
        k1_Sos_Deactivation_PM = pm.LogNormal("k1_Sos_Deactivation_PM",sigma=3.5244297004803578, mu=0.9162907318741551)
    
        # predict dose response
        CYTO = sol_op(k1_Sos_Deactivation_CYTO, *priors[k1_Sos_idx+1:k1_P90Rsk_idx], k1_P90Rsk_Deactivation_CYTO, *priors[k1_P90Rsk_idx+1:])
        PM = sol_op(k1_Sos_Deactivation_PM, *priors[k1_Sos_idx+1:k1_P90Rsk_idx], k1_P90Rsk_Deactivation_PM, *priors[k1_P90Rsk_idx+1:])
        CYTO_Rap1KD = sol_op_Rap1KD(k1_Sos_Deactivation_CYTO, *priors[k1_Sos_idx+1:k1_P90Rsk_idx], k1_P90Rsk_Deactivation_CYTO, *priors[k1_P90Rsk_idx+1:])
        PM_Rap1KD = sol_op_Rap1KD(k1_Sos_Deactivation_PM, *priors[k1_Sos_idx+1:k1_P90Rsk_idx], k1_P90Rsk_Deactivation_PM, *priors[k1_P90Rsk_idx+1:])

        # normalization factors
        CYTO_norm = pm.Deterministic("CYTO_norm", pt_max(CYTO))
        PM_norm = pm.Deterministic("PM_norm", pt_max(PM))

        # normalized predictions are the actual predictions divided by the max value
        prediction_CYTO = pm.Deterministic("prediction_CYTO", CYTO/CYTO_norm)
        prediction_PM = pm.Deterministic("prediction_PM", PM/PM_norm)
        prediction_CYTO_Rap1KD = pm.Deterministic("prediction_CYTO_Rap1KD", CYTO_Rap1KD/CYTO_norm)
        prediction_PM_Rap1KD = pm.Deterministic("prediction_PM_Rap1KD", PM_Rap1KD/PM_norm) 

        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
        llike_CYTO = pm.Normal("llike_CYTO", mu=prediction_CYTO, sigma=data_std_CYTO, observed=data_CYTO)
        llike_PM = pm.Normal("llike_PM", mu=prediction_PM, sigma=data_std_PM, observed=data_PM)
        llike_CYTO_Rap1KD = pm.Normal("llike_CYTO_Rap1KD", mu=prediction_CYTO_Rap1KD, sigma=data_std_CYTO_Rap1KD, observed=data_CYTO_Rap1KD)
        llike_PM_Rap1KD = pm.Normal("llike_PM_Rap1KD", mu=prediction_PM_Rap1KD, sigma=data_std_PM_Rap1KD, observed=data_PM_Rap1KD)

    return model

def Orton_2009_p90RsKSoS_Rap1_diff(prior_param_dict, sol_op, sol_op_Rap1KD, data_CYTO, data_PM, data_std_CYTO, data_std_PM, data_CYTO_Rap1KD, data_PM_Rap1KD, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD):
    # index of the parameter to change
    k1_P90Rsk_idx = list(prior_param_dict.keys()).index('k1_P90Rsk_Deactivation')
    k1_Sos_idx = list(prior_param_dict.keys()).index('k1_Sos_Deactivation')
    k1_C3G_Deact_idx = list(prior_param_dict.keys()).index('k1_C3G_Deactivation')

    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            if key not in ['k1_P90Rsk_Deactivation', 'k1_Sos_Deactivation', 'k1_C3G_Deactivation']:
                prior = eval(value)
                priors.append(prior)
            else:
                priors.append(None)
            
        k1_C3G_Deactivation_CYTO = pm.LogNormal("k1_C3G_Deactivation_CYTO",sigma=3.5244297004803578, mu=0.9162907318741551)
        k1_C3G_Deactivation_PM = pm.LogNormal("k1_C3G_Deactivation_PM",sigma=3.5244297004803578, mu=0.9162907318741551)
        k1_P90Rsk_Deactivation_CYTO = pm.LogNormal("k1_P90Rsk_Deactivation_CYTO",sigma=3.5244297004803578, mu=-5.298317366548036)
        k1_P90Rsk_Deactivation_PM = pm.LogNormal("k1_P90Rsk_Deactivation_PM",sigma=3.5244297004803578, mu=-5.298317366548036)
        k1_Sos_Deactivation_CYTO = pm.LogNormal("k1_Sos_Deactivation_CYTO",sigma=3.5244297004803578, mu=0.9162907318741551)
        k1_Sos_Deactivation_PM = pm.LogNormal("k1_Sos_Deactivation_PM",sigma=3.5244297004803578, mu=0.9162907318741551)
    
        # predict dose response
        CYTO = sol_op(k1_Sos_Deactivation_CYTO, \
            *priors[k1_Sos_idx+1:k1_C3G_Deact_idx], \
            k1_C3G_Deactivation_CYTO, \
            *priors[k1_C3G_Deact_idx+1:k1_P90Rsk_idx], \
            k1_P90Rsk_Deactivation_CYTO, *priors[k1_P90Rsk_idx+1:])
        
        PM = sol_op(k1_Sos_Deactivation_PM, \
            *priors[k1_Sos_idx+1:k1_C3G_Deact_idx], \
            k1_C3G_Deactivation_PM, \
            *priors[k1_C3G_Deact_idx+1:k1_P90Rsk_idx], \
            k1_P90Rsk_Deactivation_PM, *priors[k1_P90Rsk_idx+1:])
        
        CYTO_Rap1KD = sol_op_Rap1KD(k1_Sos_Deactivation_CYTO, \
            *priors[k1_Sos_idx+1:k1_C3G_Deact_idx], \
            k1_C3G_Deactivation_CYTO, \
            *priors[k1_C3G_Deact_idx+1:k1_P90Rsk_idx], \
            k1_P90Rsk_Deactivation_CYTO, *priors[k1_P90Rsk_idx+1:])
        
        PM_Rap1KD = sol_op_Rap1KD(k1_Sos_Deactivation_PM, \
            *priors[k1_Sos_idx+1:k1_C3G_Deact_idx], \
            k1_C3G_Deactivation_PM, \
            *priors[k1_C3G_Deact_idx+1:k1_P90Rsk_idx], \
            k1_P90Rsk_Deactivation_PM, *priors[k1_P90Rsk_idx+1:])
        
        # normalization factors
        CYTO_norm = pm.Deterministic("CYTO_norm", pt_max(CYTO))
        PM_norm = pm.Deterministic("PM_norm", pt_max(PM))

        # normalized predictions are the actual predictions divided by the max value
        prediction_CYTO = pm.Deterministic("prediction_CYTO", CYTO/CYTO_norm)
        prediction_PM = pm.Deterministic("prediction_PM", PM/PM_norm)
        prediction_CYTO_Rap1KD = pm.Deterministic("prediction_CYTO_Rap1KD", CYTO_Rap1KD/CYTO_norm)
        prediction_PM_Rap1KD = pm.Deterministic("prediction_PM_Rap1KD", PM_Rap1KD/PM_norm) 

        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
        llike_CYTO = pm.Normal("llike_CYTO", mu=prediction_CYTO, sigma=data_std_CYTO, observed=data_CYTO)
        llike_PM = pm.Normal("llike_PM", mu=prediction_PM, sigma=data_std_PM, observed=data_PM)
        llike_CYTO_Rap1KD = pm.Normal("llike_CYTO_Rap1KD", mu=prediction_CYTO_Rap1KD, sigma=data_std_CYTO_Rap1KD, observed=data_CYTO_Rap1KD)
        llike_PM_Rap1KD = pm.Normal("llike_PM_Rap1KD", mu=prediction_PM_Rap1KD, sigma=data_std_PM_Rap1KD, observed=data_PM_Rap1KD)

    return model

def Shin_2014_Rap1_diff(prior_param_dict, sol_op, sol_op_Rap1KD, data_CYTO, data_PM, data_std_CYTO, data_std_PM,
             data_CYTO_Rap1KD, data_PM_Rap1KD, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD):
    # index of the parameter to change
    kRap1_RafAct = list(prior_param_dict.keys()).index('kRap1_RafAct')
    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            if key != 'kRap1_RafAct':
                prior = eval(value)
                priors.append(prior)

        kRap1_RafAct_CYTO = pm.LogNormal("kRap1_RafAct_CYTO",sigma=2.79, mu=0.0)
        kRap1_RafAct_PM = pm.LogNormal("kRap1_RafAct_PM",sigma=2.79, mu=0.0)
    
        # predict dose response
        CYTO = sol_op(*priors, kRap1_RafAct_CYTO)
        PM = sol_op(*priors, kRap1_RafAct_PM)
        CYTO_Rap1KD = sol_op_Rap1KD(*priors, kRap1_RafAct_CYTO)
        PM_Rap1KD = sol_op_Rap1KD(*priors, kRap1_RafAct_PM)

        # normalization factors
        CYTO_norm = pm.Deterministic("CYTO_norm", pt_max(CYTO))
        PM_norm = pm.Deterministic("PM_norm", pt_max(PM))

        # normalized predictions are the actual predictions divided by the max value
        prediction_CYTO = pm.Deterministic("prediction_CYTO", CYTO/CYTO_norm)
        prediction_PM = pm.Deterministic("prediction_PM", PM/PM_norm)
        prediction_CYTO_Rap1KD = pm.Deterministic("prediction_CYTO_Rap1KD", CYTO_Rap1KD/CYTO_norm)
        prediction_PM_Rap1KD = pm.Deterministic("prediction_PM_Rap1KD", PM_Rap1KD/PM_norm) 

        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
        llike_CYTO = pm.Normal("llike_CYTO", mu=prediction_CYTO, sigma=data_std_CYTO, observed=data_CYTO)
        llike_PM = pm.Normal("llike_PM", mu=prediction_PM, sigma=data_std_PM, observed=data_PM)
        llike_CYTO_Rap1KD = pm.Normal("llike_CYTO_Rap1KD", mu=prediction_CYTO_Rap1KD, sigma=data_std_CYTO_Rap1KD, observed=data_CYTO_Rap1KD)
        llike_PM_Rap1KD = pm.Normal("llike_PM_Rap1KD", mu=prediction_PM_Rap1KD, sigma=data_std_PM_Rap1KD, observed=data_PM_Rap1KD)

    return model

def Shin_2014_Sos_diff(prior_param_dict, sol_op, sol_op_Rap1KD, data_CYTO, data_PM, data_std_CYTO, data_std_PM,
             data_CYTO_Rap1KD, data_PM_Rap1KD, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD):
    # index of the parameter to change
    ki39_idx = list(prior_param_dict.keys()).index('ki39')
    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            if key != 'ki39':
                prior = eval(value)
                priors.append(prior)
            else:
                priors.append(None)
            
        ki39_CYTO = pm.LogNormal("ki39_CYTO",sigma=3.5244297004803578, mu=-7.1670442769327005)
        ki39_PM = pm.LogNormal("ki39_PM",sigma=3.5244297004803578, mu=-7.1670442769327005)
    
        # predict dose response
        CYTO = sol_op(*priors[0:ki39_idx], ki39_CYTO, *priors[ki39_idx+1:])
        PM = sol_op(*priors[0:ki39_idx], ki39_PM, *priors[ki39_idx+1:])
        CYTO_Rap1KD = sol_op_Rap1KD(*priors[0:ki39_idx], ki39_CYTO, *priors[ki39_idx+1:])
        PM_Rap1KD = sol_op_Rap1KD(*priors[0:ki39_idx], ki39_PM, *priors[ki39_idx+1:])
        
        # normalization factors
        CYTO_norm = pm.Deterministic("CYTO_norm", pt_max(CYTO))
        PM_norm = pm.Deterministic("PM_norm", pt_max(PM))

        # normalized predictions are the actual predictions divided by the max value
        prediction_CYTO = pm.Deterministic("prediction_CYTO", CYTO/CYTO_norm)
        prediction_PM = pm.Deterministic("prediction_PM", PM/PM_norm)
        prediction_CYTO_Rap1KD = pm.Deterministic("prediction_CYTO_Rap1KD", CYTO_Rap1KD/CYTO_norm)
        prediction_PM_Rap1KD = pm.Deterministic("prediction_PM_Rap1KD", PM_Rap1KD/PM_norm) 

        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
        llike_CYTO = pm.Normal("llike_CYTO", mu=prediction_CYTO, sigma=data_std_CYTO, observed=data_CYTO)
        llike_PM = pm.Normal("llike_PM", mu=prediction_PM, sigma=data_std_PM, observed=data_PM)
        llike_CYTO_Rap1KD = pm.Normal("llike_CYTO_Rap1KD", mu=prediction_CYTO_Rap1KD, sigma=data_std_CYTO_Rap1KD, observed=data_CYTO_Rap1KD)
        llike_PM_Rap1KD = pm.Normal("llike_PM_Rap1KD", mu=prediction_PM_Rap1KD, sigma=data_std_PM_Rap1KD, observed=data_PM_Rap1KD)

    return model

def Shin_2014_Sos_Rap1_diff(prior_param_dict, sol_op, sol_op_Rap1KD, data_CYTO, data_PM, data_std_CYTO, data_std_PM,
             data_CYTO_Rap1KD, data_PM_Rap1KD, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD):
    # index of the parameter to change
    ki39_idx = list(prior_param_dict.keys()).index('ki39')
    kRap1_RafAct = list(prior_param_dict.keys()).index('kRap1_RafAct')
    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            if key not in ['ki39', 'kRap1_RafAct']:
                prior = eval(value)
                priors.append(prior)
            elif key == 'ki39':
                priors.append(None)
            
        ki39_CYTO = pm.LogNormal("ki39_CYTO",sigma=3.5244297004803578, mu=-7.1670442769327005)
        ki39_PM = pm.LogNormal("ki39_PM",sigma=3.5244297004803578, mu=-7.1670442769327005)
        kRap1_RafAct_CYTO = pm.LogNormal("kRap1_RafAct_CYTO",sigma=2.79, mu=0.0)
        kRap1_RafAct_PM = pm.LogNormal("kRap1_RafAct_PM",sigma=2.79, mu=0.0)
    
        # predict dose response
        CYTO = sol_op(*priors[0:ki39_idx], ki39_CYTO, *priors[ki39_idx+1:], kRap1_RafAct_CYTO)
        PM = sol_op(*priors[0:ki39_idx], ki39_PM, *priors[ki39_idx+1:], kRap1_RafAct_PM)
        CYTO_Rap1KD = sol_op_Rap1KD(*priors[0:ki39_idx], ki39_CYTO, *priors[ki39_idx+1:], kRap1_RafAct_CYTO)
        PM_Rap1KD = sol_op_Rap1KD(*priors[0:ki39_idx], ki39_PM, *priors[ki39_idx+1:], kRap1_RafAct_PM)
        
        # normalization factors
        CYTO_norm = pm.Deterministic("CYTO_norm", pt_max(CYTO))
        PM_norm = pm.Deterministic("PM_norm", pt_max(PM))

        # normalized predictions are the actual predictions divided by the max value
        prediction_CYTO = pm.Deterministic("prediction_CYTO", CYTO/CYTO_norm)
        prediction_PM = pm.Deterministic("prediction_PM", PM/PM_norm)
        prediction_CYTO_Rap1KD = pm.Deterministic("prediction_CYTO_Rap1KD", CYTO_Rap1KD/CYTO_norm)
        prediction_PM_Rap1KD = pm.Deterministic("prediction_PM_Rap1KD", PM_Rap1KD/PM_norm) 

        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
        llike_CYTO = pm.Normal("llike_CYTO", mu=prediction_CYTO, sigma=data_std_CYTO, observed=data_CYTO)
        llike_PM = pm.Normal("llike_PM", mu=prediction_PM, sigma=data_std_PM, observed=data_PM)
        llike_CYTO_Rap1KD = pm.Normal("llike_CYTO_Rap1KD", mu=prediction_CYTO_Rap1KD, sigma=data_std_CYTO_Rap1KD, observed=data_CYTO_Rap1KD)
        llike_PM_Rap1KD = pm.Normal("llike_PM_Rap1KD", mu=prediction_PM_Rap1KD, sigma=data_std_PM_Rap1KD, observed=data_PM_Rap1KD)

    return model

def Ryu_2015_Rap1_diff(prior_param_dict, sol_op, sol_op_Rap1KD, data_CYTO, data_PM, data_std_CYTO, data_std_PM,
             data_CYTO_Rap1KD, data_PM_Rap1KD, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD):
    
    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            if key not in ['k_RafRap1', 'D_RafRap1']:
                prior = eval(value)
                priors.append(prior)
        
        k_RafRap1_CYTO = pm.LogNormal("k_RafRap1_CYTO",sigma=3.5244297004803578, mu=1.6094379124341003)
        k_RafRap1_PM = pm.LogNormal("k_RafRap1_PM",sigma=3.5244297004803578, mu=1.6094379124341003)
        D_RafRap1_CYTO = pm.LogNormal("D_RafRap1_CYTO",sigma=3.5244297004803578, mu=0.0)
        D_RafRap1_PM = pm.LogNormal("D_RafRap1_PM",sigma=3.5244297004803578, mu=0.0)
    
        # predict dose response
        CYTO = sol_op(*priors, k_RafRap1_CYTO, D_RafRap1_CYTO)
        PM = sol_op(*priors, k_RafRap1_PM, D_RafRap1_PM)
        CYTO_Rap1KD = sol_op_Rap1KD(*priors, k_RafRap1_CYTO, D_RafRap1_CYTO)
        PM_Rap1KD = sol_op_Rap1KD(*priors, k_RafRap1_PM, D_RafRap1_PM)

        # normalization factors
        CYTO_norm = pm.Deterministic("CYTO_norm", pt_max(CYTO))
        PM_norm = pm.Deterministic("PM_norm", pt_max(PM))

        # normalized predictions are the actual predictions divided by the max value
        prediction_CYTO = pm.Deterministic("prediction_CYTO", CYTO/CYTO_norm)
        prediction_PM = pm.Deterministic("prediction_PM", PM/PM_norm)
        prediction_CYTO_Rap1KD = pm.Deterministic("prediction_CYTO_Rap1KD", CYTO_Rap1KD/CYTO_norm)
        prediction_PM_Rap1KD = pm.Deterministic("prediction_PM_Rap1KD", PM_Rap1KD/PM_norm) 

        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
        llike_CYTO = pm.Normal("llike_CYTO", mu=prediction_CYTO, sigma=data_std_CYTO, observed=data_CYTO)
        llike_PM = pm.Normal("llike_PM", mu=prediction_PM, sigma=data_std_PM, observed=data_PM)
        llike_CYTO_Rap1KD = pm.Normal("llike_CYTO_Rap1KD", mu=prediction_CYTO_Rap1KD, sigma=data_std_CYTO_Rap1KD, observed=data_CYTO_Rap1KD)
        llike_PM_Rap1KD = pm.Normal("llike_PM_Rap1KD", mu=prediction_PM_Rap1KD, sigma=data_std_PM_Rap1KD, observed=data_PM_Rap1KD)

    return model

def Ryu_2015_DUSP_diff(prior_param_dict, sol_op, sol_op_Rap1KD, data_CYTO, data_PM, data_std_CYTO, data_std_PM,
             data_CYTO_Rap1KD, data_PM_Rap1KD, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD):
    # index of the parameter to change
    D2_idx = list(prior_param_dict.keys()).index('D2')
    T_dusp_idx = list(prior_param_dict.keys()).index('T_dusp')
    K_dusp_idx = list(prior_param_dict.keys()).index('K_dusp')
    dusp_ind_idx = list(prior_param_dict.keys()).index('dusp_ind')

    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            if key not in ['D2', 'T_dusp', 'K_dusp', 'dusp_ind']:
                prior = eval(value)
                priors.append(prior)
            else:
                priors.append(None)

        D2_CYTO = pm.LogNormal("D2_CYTO",sigma=0.3902194510262181, mu=-2.3025850929940455)
        D2_PM = pm.LogNormal("D2_PM",sigma=0.3902194510262181, mu=-2.3025850929940455)
        T_dusp_CYTO = pm.LogNormal("T_dusp_CYTO",sigma=0.3902194510262181, mu=4.499809670330265)
        T_dusp_PM = pm.LogNormal("T_dusp_PM",sigma=0.3902194510262181, mu=4.499809670330265)
        K_dusp_CYTO = pm.LogNormal("K_dusp_CYTO",sigma=0.3902194510262181, mu=-2.3025850929940455)
        K_dusp_PM = pm.LogNormal("K_dusp_PM",sigma=0.3902194510262181, mu=-2.3025850929940455)
        dusp_ind_CYTO = pm.LogNormal("dusp_ind_CYTO",sigma=0.3902194510262181, mu=1.791759469228055)
        dusp_ind_PM = pm.LogNormal("dusp_ind_PM",sigma=0.3902194510262181, mu=1.791759469228055)
    
        # predict dose response
        CYTO = sol_op(*priors[0:D2_idx], D2_CYTO, *priors[D2_idx+1:dusp_ind_idx], \
                dusp_ind_CYTO, K_dusp_CYTO, T_dusp_CYTO, *priors[T_dusp_idx+1:])
        PM = sol_op(*priors[0:D2_idx], D2_PM, *priors[D2_idx+1:dusp_ind_idx], \
                dusp_ind_PM, K_dusp_PM, T_dusp_PM, *priors[T_dusp_idx+1:])
        CYTO_Rap1KD = sol_op_Rap1KD(*priors[0:D2_idx], D2_CYTO, *priors[D2_idx+1:dusp_ind_idx], \
                dusp_ind_CYTO, K_dusp_CYTO, T_dusp_CYTO, *priors[T_dusp_idx+1:])
        PM_Rap1KD = sol_op_Rap1KD(*priors[0:D2_idx], D2_PM, *priors[D2_idx+1:dusp_ind_idx], \
                dusp_ind_PM, K_dusp_PM, T_dusp_PM, *priors[T_dusp_idx+1:])

        # normalization factors
        CYTO_norm = pm.Deterministic("CYTO_norm", pt_max(CYTO))
        PM_norm = pm.Deterministic("PM_norm", pt_max(PM))

        # normalized predictions are the actual predictions divided by the max value
        prediction_CYTO = pm.Deterministic("prediction_CYTO", CYTO/CYTO_norm)
        prediction_PM = pm.Deterministic("prediction_PM", PM/PM_norm)
        prediction_CYTO_Rap1KD = pm.Deterministic("prediction_CYTO_Rap1KD", CYTO_Rap1KD/CYTO_norm)
        prediction_PM_Rap1KD = pm.Deterministic("prediction_PM_Rap1KD", PM_Rap1KD/PM_norm) 

        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
        llike_CYTO = pm.Normal("llike_CYTO", mu=prediction_CYTO, sigma=data_std_CYTO, observed=data_CYTO)
        llike_PM = pm.Normal("llike_PM", mu=prediction_PM, sigma=data_std_PM, observed=data_PM)
        llike_CYTO_Rap1KD = pm.Normal("llike_CYTO_Rap1KD", mu=prediction_CYTO_Rap1KD, sigma=data_std_CYTO_Rap1KD, observed=data_CYTO_Rap1KD)
        llike_PM_Rap1KD = pm.Normal("llike_PM_Rap1KD", mu=prediction_PM_Rap1KD, sigma=data_std_PM_Rap1KD, observed=data_PM_Rap1KD)

    return model

def Ryu_2015_DUSP_Rap1_diff(prior_param_dict, sol_op, sol_op_Rap1KD, data_CYTO, data_PM, data_std_CYTO, data_std_PM,
             data_CYTO_Rap1KD, data_PM_Rap1KD, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD):
    # index of the parameter to change
    D2_idx = list(prior_param_dict.keys()).index('D2')
    T_dusp_idx = list(prior_param_dict.keys()).index('T_dusp')
    K_dusp_idx = list(prior_param_dict.keys()).index('K_dusp')
    dusp_ind_idx = list(prior_param_dict.keys()).index('dusp_ind')

    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            if key not in ['D2', 'T_dusp', 'K_dusp', 'dusp_ind', 'k_RafRap1', 'D_RafRap1']:
                prior = eval(value)
                priors.append(prior)
            elif key in ['D2', 'T_dusp', 'K_dusp', 'dusp_ind']:
                priors.append(None)

        D2_CYTO = pm.LogNormal("D2_CYTO",sigma=0.3902194510262181, mu=-2.3025850929940455)
        D2_PM = pm.LogNormal("D2_PM",sigma=0.3902194510262181, mu=-2.3025850929940455)
        T_dusp_CYTO = pm.LogNormal("T_dusp_CYTO",sigma=0.3902194510262181, mu=4.499809670330265)
        T_dusp_PM = pm.LogNormal("T_dusp_PM",sigma=0.3902194510262181, mu=4.499809670330265)
        K_dusp_CYTO = pm.LogNormal("K_dusp_CYTO",sigma=0.3902194510262181, mu=-2.3025850929940455)
        K_dusp_PM = pm.LogNormal("K_dusp_PM",sigma=0.3902194510262181, mu=-2.3025850929940455)
        dusp_ind_CYTO = pm.LogNormal("dusp_ind_CYTO",sigma=0.3902194510262181, mu=1.791759469228055)
        dusp_ind_PM = pm.LogNormal("dusp_ind_PM",sigma=0.3902194510262181, mu=1.791759469228055)
        k_RafRap1_CYTO = pm.LogNormal("k_RafRap1_CYTO",sigma=0.3902194510262181, mu=-0.6931471805599453)
        k_RafRap1_PM = pm.LogNormal("k_RafRap1_PM",sigma=0.3902194510262181, mu=-0.6931471805599453)
        D_RafRap1_CYTO = pm.LogNormal("D_RafRap1_CYTO",sigma=0.3902194510262181, mu=0.0)
        D_RafRap1_PM = pm.LogNormal("D_RafRap1_PM",sigma=0.3902194510262181, mu=0.0)
    

        # predict dose response
        CYTO = sol_op(*priors[0:D2_idx], D2_CYTO, *priors[D2_idx+1:dusp_ind_idx], \
                      dusp_ind_CYTO, K_dusp_CYTO, T_dusp_CYTO, *priors[T_dusp_idx+1:],\
                    k_RafRap1_CYTO, D_RafRap1_CYTO)
        PM = sol_op(*priors[0:D2_idx], D2_PM, *priors[D2_idx+1:dusp_ind_idx], \
                      dusp_ind_PM, K_dusp_PM, T_dusp_PM, *priors[T_dusp_idx+1:],\
                    k_RafRap1_PM, D_RafRap1_PM)
        CYTO_Rap1KD = sol_op_Rap1KD(*priors[0:D2_idx], D2_CYTO, *priors[D2_idx+1:dusp_ind_idx], \
                      dusp_ind_CYTO, K_dusp_CYTO, T_dusp_CYTO, *priors[T_dusp_idx+1:],\
                    k_RafRap1_CYTO, D_RafRap1_CYTO)
        PM_Rap1KD = sol_op_Rap1KD(*priors[0:D2_idx], D2_PM, *priors[D2_idx+1:dusp_ind_idx], \
                      dusp_ind_PM, K_dusp_PM, T_dusp_PM, *priors[T_dusp_idx+1:],\
                    k_RafRap1_PM, D_RafRap1_PM)

        # normalization factors
        CYTO_norm = pm.Deterministic("CYTO_norm", pt_max(CYTO))
        PM_norm = pm.Deterministic("PM_norm", pt_max(PM))

        # normalized predictions are the actual predictions divided by the max value
        prediction_CYTO = pm.Deterministic("prediction_CYTO", CYTO/CYTO_norm)
        prediction_PM = pm.Deterministic("prediction_PM", PM/PM_norm)
        prediction_CYTO_Rap1KD = pm.Deterministic("prediction_CYTO_Rap1KD", CYTO_Rap1KD/CYTO_norm)
        prediction_PM_Rap1KD = pm.Deterministic("prediction_PM_Rap1KD", PM_Rap1KD/PM_norm) 

        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
        llike_CYTO = pm.Normal("llike_CYTO", mu=prediction_CYTO, sigma=data_std_CYTO, observed=data_CYTO)
        llike_PM = pm.Normal("llike_PM", mu=prediction_PM, sigma=data_std_PM, observed=data_PM)
        llike_CYTO_Rap1KD = pm.Normal("llike_CYTO_Rap1KD", mu=prediction_CYTO_Rap1KD, sigma=data_std_CYTO_Rap1KD, observed=data_CYTO_Rap1KD)
        llike_PM_Rap1KD = pm.Normal("llike_PM_Rap1KD", mu=prediction_PM_Rap1KD, sigma=data_std_PM_Rap1KD, observed=data_PM_Rap1KD)

    return model

##############################
# def arg parsers to take inputs from the command line
##############################
def parse_args(raw_args=None):
    """ function to parse command line arguments
    """
    parser=argparse.ArgumentParser(description="Generate Morris samples for the specified model.")
    parser.add_argument("-model", type=str, help="model to process.")
    parser.add_argument("-pymc_model", type=str, help="Pymc model to use.")
    parser.add_argument("-free_params", type=str, help="parameters to estimate")
    parser.add_argument("-Rap1_state", type=str, help="Names of inactive Rap1 species.")
    parser.add_argument("-nsamples", type=int, default=1000, help="Number of samples to posterior samples to draw. Defaults to 1000.")
    parser.add_argument("-savedir", type=str, help="Path to save results. Defaults to current directory.")
    parser.add_argument("-input_state", type=str, default='EGF', help="Name of EGF input in the state vector. Defaults to EGF.")
    parser.add_argument("-EGF_conversion_factor", type=float, default=1.0, help="Conversion factor to convert EGF from nM to other units. Defaults to 1.")
    parser.add_argument("-ERK_states", type=str, default=None, help="Names of ERK species to use for inference. Defaults to None.")
    parser.add_argument("-time_conversion_factor", type=float, default=1.0, help="Conversion factor to convert from seconds by division. Default is 1. Mins would be 60")
    parser.add_argument("-prior_family", type=str, default="[['Gamma()',['alpha', 'beta']]]", help="Prior family to use. Defaults to uniform.")
    parser.add_argument("-ncores", type=int, default=1, help="Number of cores to use for multiprocessing. Defaults to None which will use all available cores.")
    parser.add_argument("-nchains", type=int, default=4, help="Number of chains to run. Defaults to 4.")
    parser.add_argument("--skip_sample", action='store_false',default=True)
    parser.add_argument("--skip_prior_sample", action='store_false',default=True)
    parser.add_argument("-rtol", type=float,default=1e-6)
    parser.add_argument("-atol", type=float,default=1e-6)
    parser.add_argument("-upper_prior_mult", type=float,default=1e3)
    parser.add_argument("-lower_prior_mult", type=float,default=1e-3)

    args=parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    """ main function to execute command line script functionality.
    """
    seed = np.random.default_rng(seed=123)
    args = parse_args(raw_args)

    print('Processing model {}.'.format(args.model))
    
    # try calling the model
    try:
        model = eval(args.model + '(transient=False)')
    except:
        print('Warning Model {} not found. Skipping this.'.format(args.model))

    # get parameter names and initial conditions
    p_dict, plist = model.get_nominal_params()
    y0_dict, y0 = model.get_initial_conditions()

    # add savedir if it does not exist
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    # load the data
    # load in the training data
    data_file = '../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-'

    inputs_CYTO, data_CYTO, data_std_CYTO, times_CYTO \
        = load_data_json(data_file+'CYTO_CYTOmax.json', data_std=True, time=True)
    _, data_PM, data_std_PM, times_PM \
        = load_data_json(data_file+'PM_PMmax.json', data_std=True, time=True)

    data_file = '../../../results/MAPK/Keyes_et_al_2020-fig3-data1-v2-'
    _, data_CYTO_RAP1i, data_std_CYTO_RAP1i, \
        times_CYTO_RAP1i = load_data_json(data_file+'CYTO_RAP1inhib_CYTOmax.json', \
        data_std=True, time=True)
    _, data_PM_RAP1i, data_std_PM_RAP1i, times_PM_RAP1i \
        = load_data_json(data_file+'PM_RAP1inhib_PMmax.json', \
        data_std=True, time=True)

    data_time_to_mins = 60

    # data in each compartment are sampled at slightly different times, so we need to interpolate to align them
    # use diffrax linear interpolation to get the MAPK activity at specific time point over 40mins
    data_CYTO_interp = diffrax.LinearInterpolation(times_CYTO, data_CYTO)
    data_std_CYTO_interp = diffrax.LinearInterpolation(times_CYTO, data_std_CYTO)
    data_PM_interp = diffrax.LinearInterpolation(times_PM, data_PM)
    data_std_PM_interp = diffrax.LinearInterpolation(times_PM, data_std_PM)

    data_CYTO_RAP1i_interp = diffrax.LinearInterpolation(times_CYTO_RAP1i, \
        data_CYTO_RAP1i)
    data_std_CYTO_RAP1i_interp = diffrax.LinearInterpolation(times_CYTO_RAP1i,\
        data_std_CYTO_RAP1i)
    data_PM_RAP1i_interp = diffrax.LinearInterpolation(times_PM_RAP1i, data_PM_RAP1i)
    data_std_PM_RAP1i_interp = diffrax.LinearInterpolation(times_PM_RAP1i, \
        data_std_PM_RAP1i)

    min_time = np.round(np.min([times_CYTO[-1], times_PM[-1], times_CYTO_RAP1i[-1], times_PM_RAP1i[-1]]))
    n_times = np.max([len(times_CYTO), len(times_PM), len(times_CYTO_RAP1i), len(times_PM_RAP1i)])
    times = np.linspace(0, min_time, n_times)

    # get data at standard times
    data_CYTO = data_CYTO_interp.evaluate(times)
    data_std_CYTO = data_std_CYTO_interp.evaluate(times)
    data_PM = data_PM_interp.evaluate(times)
    data_std_PM = data_std_PM_interp.evaluate(times)

    data_CYTO_RAP1i = data_CYTO_RAP1i_interp.evaluate(times)
    data_std_CYTO_RAP1i = data_std_CYTO_RAP1i_interp.evaluate(times)
    data_PM_RAP1i = data_PM_RAP1i_interp.evaluate(times)
    data_std_PM_RAP1i = data_std_PM_RAP1i_interp.evaluate(times)


    # convert EGF to required units
    # inputs are the same in both compartments, so just use CYTO
    inputs_native_units = inputs_CYTO * args.EGF_conversion_factor

    # get the params to sample
    analyze_params = args.free_params.split(',')
    free_param_idxs = [list(p_dict.keys()).index(p) for p in analyze_params]

    # get the EGF index and ERK indices
    state_names = list(y0_dict.keys())
    EGF_idx = state_names.index(args.input_state)
    ERK_indices = [state_names.index(s) for s in args.ERK_states.split(',')]

    # construct the strings to make priors and constants
    prior_param_dict = set_prior_params(args.model, list(p_dict.keys()), plist, free_param_idxs, upper_mult=args.upper_prior_mult, lower_mult=args.lower_prior_mult, prior_family=args.prior_family, savedir=args.savedir, saveplot=False)

    # make simulator lambda function that solves at correct times with the time conversion factor taken into account]
    # NOTE: use times[1:] to avoind issues associated with included t=0 point
    def ERK_stim_traj(p, model, max_time, y0, output_states):
        traj = solve_traj(model, y0, p, max_time, ERK_indices, times[1:]/args.time_conversion_factor, args.rtol, args.atol)

        return [traj], traj

    # make initial conditions that reflect the inputs
    y0_EGF_ins = construct_y0_EGF_inputs(inputs_native_units, np.array([y0]), EGF_idx)

    # constuct initial condition with Rap1 kockdown
    y0_Rap1_knockdown = y0_dict.copy()
    y0_Rap1_knockdown[args.Rap1_state] = 0.0
    y0_Rap1KD = tuple(y0_Rap1_knockdown.values())
    y0_EGF_ins_Rap1_KD = construct_y0_EGF_inputs(inputs_native_units, np.array([y0_Rap1KD]), EGF_idx)

    # construct the pymc model
    # Note: We do not use the build_pymc_model function, because we need to 
    #   build a model that runs the simulator three times for each input level
    # NOTE: use data_CYTO[1:], etc to avoid issues associated with included t=0 point
    try:
        model_func = lambda prior_param_dict, sol_op, sol_op_Rap1KD, data, \
            data_std: eval(args.pymc_model)(prior_param_dict, sol_op, \
            sol_op_Rap1KD, [data_CYTO[1:]], [data_PM[1:]], [data_std_CYTO[1:]], \
                [data_std_PM[1:]],  [data_CYTO_RAP1i[1:]], [data_PM_RAP1i[1:]], \
                [data_std_CYTO_RAP1i[1:]], [data_std_PM_RAP1i[1:]])
    except OSError as e:
        print('Warning Pymc model {} not found'.format(args.pymc_model))
        raise
        
    pymc_model = build_pymc_model_local(prior_param_dict, None, y0_EGF_ins[0], 
                    y0_EGF_ins_Rap1_KD[0], ERK_indices, 
                    np.max(times/args.time_conversion_factor), diffrax.ODETerm(model), 
                    simulator=ERK_stim_traj, data_sigma=None, model_func=model_func,)

    if args.skip_prior_sample:
        # sample from the posterior predictive
        with pymc_model:
            prior_predictive = pm.sample_prior_predictive(samples=500, random_seed=seed)
        
        prior_predictive.to_json(args.savedir + args.model + '_prior_samples.json')

        # extract llike values
        prior_llike_CYTO = np.squeeze(prior_predictive.prior_predictive['llike_CYTO'].values)
        prior_llike_PM = np.squeeze(prior_predictive.prior_predictive['llike_PM'].values)
        prior_llike_CYTO_Rap1KD = np.squeeze(prior_predictive.prior_predictive['llike_CYTO_Rap1KD'].values)
        prior_llike_PM_Rap1KD = np.squeeze(prior_predictive.prior_predictive['llike_PM_Rap1KD'].values)

        plot_trajectories(args.model, times/60, inputs_CYTO, args.savedir, 'prior',
        prior_llike_CYTO, prior_llike_PM, prior_llike_CYTO_Rap1KD, prior_llike_PM_Rap1KD, data_CYTO, 
        data_PM, data_CYTO_RAP1i, data_PM_RAP1i, data_std_CYTO, data_std_PM, 
        data_std_CYTO_RAP1i, data_std_PM_RAP1i)

    # SMC sampling
    if args.skip_sample:
        posterior_idata = smc_pymc(pymc_model, args.model, args.savedir, 
                    nsamples=args.nsamples, ncores=args.ncores, chains=args.nchains,seed=seed)
    else:
        posterior_idata, _ = load_smc_samples_to_idata(args.savedir + args.model + '_smc_samples.json')
    
    # trace plots and diagnostics
    plot_sampling_trace_diagnoses(posterior_idata, args.savedir, args.model)

    # posterior predictive samples
    with pymc_model:
        # sample from the posterior predictive
        posterior_predictive = pm.sample_posterior_predictive(posterior_idata,random_seed=seed)
        
    posterior_predictive.to_json(args.savedir + args.model + '_posterior_samples.json')
    
    # extract llike values
    llike_CYTO = np.squeeze(posterior_predictive.posterior_predictive['llike_CYTO'].values)
    llike_PM = np.squeeze(posterior_predictive.posterior_predictive['llike_PM'].values)
    llike_CYTO_Rap1KD = np.squeeze(posterior_predictive.posterior_predictive['llike_CYTO_Rap1KD'].values)
    llike_PM_Rap1KD = np.squeeze(posterior_predictive.posterior_predictive['llike_PM_Rap1KD'].values)

    # make the plots
    plot_trajectories(args.model, times/60, inputs_CYTO, args.savedir, 'posterior',
        llike_CYTO, llike_PM, llike_CYTO_Rap1KD, llike_PM_Rap1KD, data_CYTO, 
        data_PM, data_CYTO_RAP1i, data_PM_RAP1i, data_std_CYTO, data_std_PM, 
        data_std_CYTO_RAP1i, data_std_PM_RAP1i)
    
    print('Completed {}'.format(args.model))

if __name__ == '__main__':
    main()