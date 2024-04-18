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
# jax.config.update("jax_platform_name", "cpu")

# # print out device count
# n_devices = jax.local_device_count() 
# print(jax.devices())
# print('Using {} jax devices'.format(n_devices))

# need a posterior predictive function that can deal with the fact that we now have multiple outputs in the llike function
def create_posterior_predictive_local(model, posterior_idata, mapk_model_name, data_CYTO, data_PM, data_std_CYTO, data_std_PM, times, inputs, savedir, seed=np.random.default_rng(seed=123)):
    """ Creates prior predictive samples plot of the stimulus response curve.
    """

    # sample from the posterior predictive
    with model:
        posterior_predictive = pm.sample_posterior_predictive(posterior_idata, model=model, 
                                                              random_seed=seed)

    # extract llike values
    posterior_llike_CYTO = np.squeeze(posterior_predictive.posterior_predictive['llike_CYTO'].values)
    posterior_llike_PM = np.squeeze(posterior_predictive.posterior_predictive['llike_PM'].values)

    fig_ax = []
    for post, name, data, data_std in zip([posterior_llike_CYTO, posterior_llike_PM], ['_CYTO', '_PM'], [data_CYTO, data_PM],      
                          [data_std_CYTO, data_std_PM]):
        # generate the plot
        # reshape accordingly
        nchains,nsamples,ntime=post.shape
        posterior_llike = np.reshape(post, (nchains*nsamples, ntime))
        
        fig, ax = plot_trajectory_responses_oneAxis(posterior_llike, data, inputs, times,
                                            savedir+mapk_model_name+name+'_legend_posterior_predictive.pdf', data_std=data_std)

        # save the figure
        fig.savefig(savedir + mapk_model_name + name + '_posterior_predictive.pdf', 
                    bbox_inches='tight', transparent=True)

        # save the samples
        np.save(savedir + mapk_model_name + name + '_posterior_predictive_samples.npy', posterior_llike)

        fig_ax.append((fig, ax))
    
    return fig_ax

def create_prior_predictive_local(model, mapk_model_name, data_CYTO, data_PM, data_std_CYTO, data_std_PM, data_CYTO_Rap1KD, data_PM_Rap1KD, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD, times, inputs, savedir, seed=np.random.default_rng(seed=123)):
    """ Creates prior predictive samples plot of the stimulus response curve.
    """

    # sample from the posterior predictive
    with model:
        prior_predictive = pm.sample_prior_predictive(samples=500, random_seed=seed)

    # extract llike values
    prior_llike_CYTO = np.squeeze(prior_predictive.prior_predictive['llike_CYTO'].values)
    prior_llike_PM = np.squeeze(prior_predictive.prior_predictive['llike_PM'].values)
    prior_llike_CYTO_Rap1KD = np.squeeze(prior_predictive.prior_predictive['llike_CYTO_Rap1KD'].values)
    prior_llike_PM_Rap1KD = np.squeeze(prior_predictive.prior_predictive['llike_PM_Rap1KD'].values)

    fig_ax = []
    for prior_llike, name, data, data_std in zip([prior_llike_CYTO, prior_llike_PM, prior_llike_CYTO_Rap1KD, prior_llike_PM_Rap1KD], ['_CYTO', '_PM', '_CYTO_Rap1KD', '_PM_Rap1KD'], [data_CYTO, data_PM, data_CYTO_Rap1KD, data_PM_Rap1KD],      
                          [data_std_CYTO, data_std_PM, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD]):
        # generate the plot
        # reshape accordingly
        # nsamples,ntime=post.shape
        # prior_llike = np.reshape(post, (nsamples, ntime))
        
        fig, ax = plot_trajectory_responses_oneAxis(prior_llike, data, inputs, times,
                                            savedir+mapk_model_name+name+'_legend_prior_predictive.pdf', data_std=data_std)

        # save the figure
        fig.savefig(savedir + mapk_model_name + name + '_prior_predictive.pdf', 
                    bbox_inches='tight', transparent=True)

        fig_ax.append((fig, ax))
    
    return fig_ax

@jax.jit
def solve_traj_local(model_dfrx_ode, y0, params, t1, times, rtol, atol):
    """ simulates a model over the specified time interval and returns the 
    calculated values.
    Returns an array of shape (n_species, 1) """
    dt0=1e-3
    solver = diffrax.Kvaerno5()
    stepsize_controller=diffrax.PIDController(rtol, atol)
    t0 = 0.0
    saveat=diffrax.SaveAt(ts=times)

    sol = diffrax.diffeqsolve(
        model_dfrx_ode, 
        solver, 
        t0, 
        t1, 
        dt0, 
        tuple(y0), 
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        args=params,
        max_steps=600000,
        throw=False,)
    
    return jnp.array(sol.ys)


def build_pymc_model_local(prior_param_dict, data, y0, y0_Rap1KD,
                    output_states, max_time, model_dfrx_ode, model_func=None, 
                    simulator=ERK_stim_response, data_sigma=0.1):
    """ Builds a pymc model object for the MAPK models.

    Constructs priors for the model, and uses the ERK_stim_response function to 
    generate the stimulus response function and likelihood.
    
    If model is None, the function will use the default model. If a model is s
    pecified, it will use that model_func function to create a PyMC model.

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

        # predict
        prediction_CYTO = sol_op(*priors_CYTO)
        prediction_PM = sol_op(*priors_PM)
        prediction_CYTO_Rap1KD = sol_op_Rap1KD(*priors_CYTO)
        prediction_PM_Rap1KD = sol_op_Rap1KD(*priors_PM)

        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
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
                k1_C3G_Deactivation_CYTO = pm.LogNormal("k1_C3G_Deactivation_CYTO",sigma=0.7804389020524362, mu=0.9162907318741551)
                k1_C3G_Deactivation_PM = pm.LogNormal("k1_C3G_Deactivation_PM",sigma=0.7804389020524362, mu=0.9162907318741551)
    
        # predict dose response
        prediction_CYTO = sol_op(*priors[0:k1_C3G_Deact_idx], k1_C3G_Deactivation_CYTO, *priors[k1_C3G_Deact_idx+1:])
        prediction_PM = sol_op(*priors[0:k1_C3G_Deact_idx], k1_C3G_Deactivation_PM, *priors[k1_C3G_Deact_idx+1:])
        prediction_CYTO_Rap1KD = sol_op_Rap1KD(*priors[0:k1_C3G_Deact_idx], k1_C3G_Deactivation_CYTO, *priors[k1_C3G_Deact_idx+1:])
        prediction_PM_Rap1KD = sol_op_Rap1KD(*priors[0:k1_C3G_Deact_idx], k1_C3G_Deactivation_PM, *priors[k1_C3G_Deact_idx+1:])

        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
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
            
        k1_P90Rsk_Deactivation_CYTO = pm.LogNormal("k1_P90Rsk_Deactivation_CYTO",sigma=0.3902194510262181, mu=-5.298317366548036)
        k1_P90Rsk_Deactivation_PM = pm.LogNormal("k1_P90Rsk_Deactivation_PM",sigma=0.3902194510262181, mu=-5.298317366548036)
        k1_Sos_Deactivation_CYTO = pm.LogNormal("k1_Sos_Deactivation_CYTO",sigma=0.3902194510262181, mu=0.9162907318741551)
        k1_Sos_Deactivation_PM = pm.LogNormal("k1_Sos_Deactivation_PM",sigma=0.3902194510262181, mu=0.9162907318741551)
    
        # predict dose response
        prediction_CYTO = sol_op(k1_Sos_Deactivation_CYTO, *priors[k1_Sos_idx+1:k1_P90Rsk_idx], k1_P90Rsk_Deactivation_CYTO, *priors[k1_P90Rsk_idx+1:])
        prediction_PM = sol_op(k1_Sos_Deactivation_PM, *priors[k1_Sos_idx+1:k1_P90Rsk_idx], k1_P90Rsk_Deactivation_PM, *priors[k1_P90Rsk_idx+1:])
        prediction_CYTO_Rap1KD = sol_op_Rap1KD(k1_Sos_Deactivation_CYTO, *priors[k1_Sos_idx+1:k1_P90Rsk_idx], k1_P90Rsk_Deactivation_CYTO, *priors[k1_P90Rsk_idx+1:])
        prediction_PM_Rap1KD = sol_op_Rap1KD(k1_Sos_Deactivation_PM, *priors[k1_Sos_idx+1:k1_P90Rsk_idx], k1_P90Rsk_Deactivation_PM, *priors[k1_P90Rsk_idx+1:])

        
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
            
        k1_C3G_Deactivation_CYTO = pm.LogNormal("k1_C3G_Deactivation_CYTO",sigma=0.7804389020524362, mu=0.9162907318741551)
        k1_C3G_Deactivation_PM = pm.LogNormal("k1_C3G_Deactivation_PM",sigma=0.7804389020524362, mu=0.9162907318741551)
        k1_P90Rsk_Deactivation_CYTO = pm.LogNormal("k1_P90Rsk_Deactivation_CYTO",sigma=0.3902194510262181, mu=-5.298317366548036)
        k1_P90Rsk_Deactivation_PM = pm.LogNormal("k1_P90Rsk_Deactivation_PM",sigma=0.3902194510262181, mu=-5.298317366548036)
        k1_Sos_Deactivation_CYTO = pm.LogNormal("k1_Sos_Deactivation_CYTO",sigma=0.3902194510262181, mu=0.9162907318741551)
        k1_Sos_Deactivation_PM = pm.LogNormal("k1_Sos_Deactivation_PM",sigma=0.3902194510262181, mu=0.9162907318741551)
    
        # predict dose response
        prediction_CYTO = sol_op(k1_Sos_Deactivation_CYTO, \
                *priors[k1_Sos_idx+1:k1_C3G_Deact_idx], \
                k1_C3G_Deactivation_CYTO, \
                *priors[k1_C3G_Deact_idx+1:k1_P90Rsk_idx], \
                k1_P90Rsk_Deactivation_CYTO, *priors[k1_P90Rsk_idx+1:])
        
        prediction_PM = sol_op(k1_Sos_Deactivation_PM, \
                *priors[k1_Sos_idx+1:k1_C3G_Deact_idx], \
                k1_C3G_Deactivation_PM, \
                *priors[k1_C3G_Deact_idx+1:k1_P90Rsk_idx], \
                k1_P90Rsk_Deactivation_PM, *priors[k1_P90Rsk_idx+1:])
        
        prediction_CYTO_Rap1KD = sol_op_Rap1KD(k1_Sos_Deactivation_CYTO, \
                *priors[k1_Sos_idx+1:k1_C3G_Deact_idx], \
                k1_C3G_Deactivation_CYTO, \
                *priors[k1_C3G_Deact_idx+1:k1_P90Rsk_idx], \
                k1_P90Rsk_Deactivation_CYTO, *priors[k1_P90Rsk_idx+1:])
        
        prediction_PM_Rap1KD = sol_op_Rap1KD(k1_Sos_Deactivation_PM, \
                *priors[k1_Sos_idx+1:k1_C3G_Deact_idx], \
                k1_C3G_Deactivation_PM, \
                *priors[k1_C3G_Deact_idx+1:k1_P90Rsk_idx], \
                k1_P90Rsk_Deactivation_PM, *priors[k1_P90Rsk_idx+1:])
        
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
    kRap1Act = list(prior_param_dict.keys()).index('kRap1Act')
    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            if key != 'kRap1Act':
                prior = eval(value)
                priors.append(prior)
            else:
                priors.append(None)
                kRap1Act_CYTO = pm.LogNormal("kRap1Act_CYTO",sigma=0.3902194510262181, mu=0.0)
                kRap1Act_PM = pm.LogNormal("kRap1Act_PM",sigma=0.3902194510262181, mu=0.0)
    
        # predict dose response
        prediction_CYTO = sol_op(*priors[0:kRap1Act], kRap1Act_CYTO, *priors[kRap1Act+1:])
        prediction_PM = sol_op(*priors[0:kRap1Act], kRap1Act_PM, *priors[kRap1Act+1:])
        prediction_CYTO_Rap1KD = sol_op_Rap1KD(*priors[0:kRap1Act], kRap1Act_CYTO, *priors[kRap1Act+1:])
        prediction_PM_Rap1KD = sol_op_Rap1KD(*priors[0:kRap1Act], kRap1Act_PM, *priors[kRap1Act+1:])

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
                print(key)
                priors.append(None)
            
        ki39_CYTO = pm.LogNormal("ki39_CYTO",sigma=0.3902194510262181, mu=0.0)
        ki39_PM = pm.LogNormal("ki39_PM",sigma=0.3902194510262181, mu=0.0)
    
        # predict dose response
        prediction_CYTO = sol_op(*priors[0:ki39_idx], ki39_CYTO, *priors[ki39_idx+1:])
        prediction_PM = sol_op(*priors[0:ki39_idx], ki39_PM, *priors[ki39_idx+1:])
        prediction_CYTO_Rap1KD = sol_op_Rap1KD(*priors[0:ki39_idx], ki39_CYTO, *priors[ki39_idx+1:])
        prediction_PM_Rap1KD = sol_op_Rap1KD(*priors[0:ki39_idx], ki39_PM, *priors[ki39_idx+1:])
        
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
    kRap1Act = list(prior_param_dict.keys()).index('kRap1Act')
    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            if key not in ['ki39', 'kRap1Act']:
                prior = eval(value)
                priors.append(prior)
            else:
                print(key)
                priors.append(None)
            
        ki39_CYTO = pm.LogNormal("ki39_CYTO",sigma=0.3902194510262181, mu=0.0)
        ki39_PM = pm.LogNormal("ki39_PM",sigma=0.3902194510262181, mu=0.0)
        kRap1Act_CYTO = pm.LogNormal("kRap1Act_CYTO",sigma=0.3902194510262181, mu=0.0)
        kRap1Act_PM = pm.LogNormal("kRap1Act_PM",sigma=0.3902194510262181, mu=0.0)
    
        # predict dose response
        prediction_CYTO = sol_op(*priors[0:ki39_idx], ki39_CYTO, *priors[ki39_idx+1:kRap1Act], kRap1Act_CYTO, *priors[kRap1Act+1:])
        prediction_PM = sol_op(*priors[0:ki39_idx], ki39_PM, *priors[ki39_idx+1:kRap1Act], kRap1Act_PM, *priors[kRap1Act+1:])
        prediction_CYTO_Rap1KD = sol_op_Rap1KD(*priors[0:ki39_idx], ki39_CYTO, *priors[ki39_idx+1:kRap1Act], kRap1Act_CYTO, *priors[kRap1Act+1:])
        prediction_PM_Rap1KD = sol_op_Rap1KD(*priors[0:ki39_idx], ki39_PM, *priors[ki39_idx+1:kRap1Act], kRap1Act_PM, *priors[kRap1Act+1:])
        
        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
        llike_CYTO = pm.Normal("llike_CYTO", mu=prediction_CYTO, sigma=data_std_CYTO, observed=data_CYTO)
        llike_PM = pm.Normal("llike_PM", mu=prediction_PM, sigma=data_std_PM, observed=data_PM)
        llike_CYTO_Rap1KD = pm.Normal("llike_CYTO_Rap1KD", mu=prediction_CYTO_Rap1KD, sigma=data_std_CYTO_Rap1KD, observed=data_CYTO_Rap1KD)
        llike_PM_Rap1KD = pm.Normal("llike_PM_Rap1KD", mu=prediction_PM_Rap1KD, sigma=data_std_PM_Rap1KD, observed=data_PM_Rap1KD)

    return model

def Ryu_2015_Rap1_diff(prior_param_dict, sol_op, sol_op_Rap1KD, data_CYTO, data_PM, data_std_CYTO, data_std_PM,
             data_CYTO_Rap1KD, data_PM_Rap1KD, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD):
    # index of the parameter to change
    k_Rap1Act_idx = list(prior_param_dict.keys()).index('k_Rap1Act')
    D_Rap1Act_idx = list(prior_param_dict.keys()).index('D_Rap1Act')
    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            if key != 'k_Rap1Act':
                prior = eval(value)
                priors.append(prior)
            else:
                priors.append(None)
        
        k_Rap1Act_CYTO = pm.LogNormal("k_Rap1Act_CYTO",sigma=0.3902194510262181, mu=-0.6931471805599453)
        k_Rap1Act_PM = pm.LogNormal("k_Rap1Act_PM",sigma=0.3902194510262181, mu=-0.6931471805599453)
        D_Rap1Act_CYTO = pm.LogNormal("D_Rap1Act_CYTO",sigma=0.3902194510262181, mu=0.0)
        D_Rap1Act_PM = pm.LogNormal("D_Rap1Act_PM",sigma=0.3902194510262181, mu=0.0)
    
        # predict dose response
        prediction_CYTO = sol_op(*priors[0:k_Rap1Act_idx], k_Rap1Act_CYTO, D_Rap1Act_CYTO, *priors[D_Rap1Act_idx+1:])
        prediction_PM = sol_op(*priors[0:k_Rap1Act_idx], k_Rap1Act_PM, D_Rap1Act_PM, *priors[D_Rap1Act_idx+1:])
        prediction_CYTO_Rap1KD = sol_op_Rap1KD(*priors[0:k_Rap1Act_idx], k_Rap1Act_CYTO, D_Rap1Act_CYTO, *priors[D_Rap1Act_idx+1:])
        prediction_PM_Rap1KD = sol_op_Rap1KD(*priors[0:k_Rap1Act_idx], k_Rap1Act_PM, D_Rap1Act_PM, *priors[D_Rap1Act_idx+1:])

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
                print(key)
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
        prediction_CYTO = sol_op(*priors[0:D2_idx], D2_CYTO, *priors[D2_idx+1:dusp_ind_idx], \
                dusp_ind_CYTO, K_dusp_CYTO, T_dusp_CYTO, *priors[T_dusp_idx+1:])
        prediction_PM = sol_op(*priors[0:D2_idx], D2_PM, *priors[D2_idx+1:dusp_ind_idx], \
                dusp_ind_PM, K_dusp_PM, T_dusp_PM, *priors[T_dusp_idx+1:])
        prediction_CYTO_Rap1KD = sol_op_Rap1KD(*priors[0:D2_idx], D2_CYTO, *priors[D2_idx+1:dusp_ind_idx], \
                dusp_ind_CYTO, K_dusp_CYTO, T_dusp_CYTO, *priors[T_dusp_idx+1:])
        prediction_PM_Rap1KD = sol_op_Rap1KD(*priors[0:D2_idx], D2_PM, *priors[D2_idx+1:dusp_ind_idx], \
                dusp_ind_PM, K_dusp_PM, T_dusp_PM, *priors[T_dusp_idx+1:])

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
    k_Rap1Act_idx = list(prior_param_dict.keys()).index('k_Rap1Act')
    D_Rap1Act_idx = list(prior_param_dict.keys()).index('D_Rap1Act')

    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            if key not in ['D2', 'T_dusp', 'K_dusp', 'dusp_ind', 'k_Rap1Act', 'D_Rap1Act']:
                prior = eval(value)
                priors.append(prior)
            else:
                print(key)
                priors.append(None)

        D2_CYTO = pm.LogNormal("D2_CYTO",sigma=0.3902194510262181, mu=-2.3025850929940455)
        D2_PM = pm.LogNormal("D2_PM",sigma=0.3902194510262181, mu=-2.3025850929940455)
        T_dusp_CYTO = pm.LogNormal("T_dusp_CYTO",sigma=0.3902194510262181, mu=4.499809670330265)
        T_dusp_PM = pm.LogNormal("T_dusp_PM",sigma=0.3902194510262181, mu=4.499809670330265)
        K_dusp_CYTO = pm.LogNormal("K_dusp_CYTO",sigma=0.3902194510262181, mu=-2.3025850929940455)
        K_dusp_PM = pm.LogNormal("K_dusp_PM",sigma=0.3902194510262181, mu=-2.3025850929940455)
        dusp_ind_CYTO = pm.LogNormal("dusp_ind_CYTO",sigma=0.3902194510262181, mu=1.791759469228055)
        dusp_ind_PM = pm.LogNormal("dusp_ind_PM",sigma=0.3902194510262181, mu=1.791759469228055)
        k_Rap1Act_CYTO = pm.LogNormal("k_Rap1Act_CYTO",sigma=0.3902194510262181, mu=-0.6931471805599453)
        k_Rap1Act_PM = pm.LogNormal("k_Rap1Act_PM",sigma=0.3902194510262181, mu=-0.6931471805599453)
        D_Rap1Act_CYTO = pm.LogNormal("D_Rap1Act_CYTO",sigma=0.3902194510262181, mu=0.0)
        D_Rap1Act_PM = pm.LogNormal("D_Rap1Act_PM",sigma=0.3902194510262181, mu=0.0)
    
        # predict dose response
        prediction_CYTO = sol_op(*priors[0:D2_idx], D2_CYTO, *priors[D2_idx+1:dusp_ind_idx], \
                dusp_ind_CYTO, K_dusp_CYTO, T_dusp_CYTO, k_Rap1Act_CYTO, D_Rap1Act_CYTO, *priors[D_Rap1Act_idx+1:])
        prediction_PM = sol_op(*priors[0:D2_idx], D2_PM, *priors[D2_idx+1:dusp_ind_idx], \
                dusp_ind_PM, K_dusp_PM, T_dusp_PM, k_Rap1Act_PM, D_Rap1Act_PM, *priors[D_Rap1Act_idx+1:])
        prediction_CYTO_Rap1KD = sol_op_Rap1KD(*priors[0:D2_idx], D2_CYTO, *priors[D2_idx+1:dusp_ind_idx], \
                dusp_ind_CYTO, K_dusp_CYTO, T_dusp_CYTO, k_Rap1Act_CYTO, D_Rap1Act_CYTO, *priors[D_Rap1Act_idx+1:])
        prediction_PM_Rap1KD = sol_op_Rap1KD(*priors[0:D2_idx], D2_PM, *priors[D2_idx+1:dusp_ind_idx], \
                dusp_ind_PM, K_dusp_PM, T_dusp_PM, k_Rap1Act_PM, D_Rap1Act_PM, *priors[D_Rap1Act_idx+1:])

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
    parser.add_argument("-ERK_all_states", type=str, default=None, help="Names of ERK species to use for inference. Defaults to None.")
    parser.add_argument("-total_ERK_idx", type=int, default=None)
    parser.add_argument("-time_conversion_factor", type=float, default=1.0, help="Conversion factor to convert from seconds by division. Default is 1. Mins would be 60")
    parser.add_argument("-prior_family", type=str, default="[['Gamma()',['alpha', 'beta']]]", help="Prior family to use. Defaults to uniform.")
    parser.add_argument("-ncores", type=int, default=1, help="Number of cores to use for multiprocessing. Defaults to None which will use all available cores.")
    parser.add_argument("-nchains", type=int, default=4, help="Number of chains to run. Defaults to 4.")
    parser.add_argument("--skip_sample", action='store_false',default=True)
    parser.add_argument("--skip_prior_sample", action='store_false',default=True)
    parser.add_argument("-rtol", type=float,default=1e-6)
    parser.add_argument("-atol", type=float,default=1e-6)
    parser.add_argument("-upper_prior_mult", type=float,default=1e2)
    parser.add_argument("-lower_prior_mult", type=float,default=1e-2)

    args=parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    """ main function to execute command line script functionality.
    """
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
    inputs_PM, data_PM, data_std_PM, times_PM \
        = load_data_json(data_file+'PM_PMmax.json', data_std=True, time=True)

    data_file = '../../../results/MAPK/Keyes_et_al_2020-fig3-data1-v2-'
    inputs_CYTO_RAP1i, data_CYTO_RAP1i, data_std_CYTO_RAP1i, \
        times_CYTO_RAP1i = load_data_json(data_file+'CYTO_RAP1inhib_CYTOmax.json', \
        data_std=True, time=True)
    inputs_PM_RAP1i, data_PM_RAP1i, data_std_PM_RAP1i, times_PM_RAP1i \
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
    ERK_all_indices = [state_names.index(s) for s in args.ERK_all_states.split(',')]

    # construct the strings to make priors and constants
    prior_param_dict = set_prior_params(args.model, list(p_dict.keys()), plist, free_param_idxs, upper_mult=args.upper_prior_mult, lower_mult=args.lower_prior_mult, prior_family=args.prior_family, savedir=args.savedir, saveplot=False)

    # make simulator lambda function that solves at correct times with the time conversion factor taken into account
    def ERK_stim_traj_total_state(p, model, max_time, y0, output_states):
        traj = solve_traj_local(model, y0, p, max_time, times/args.time_conversion_factor, args.rtol, args.atol)

        # return normalized trajectory
        total_ERK = np.sum(np.array(y0)[ERK_all_indices])
        return [np.sum(traj[output_states,:], axis=0) / total_ERK], traj
    
    def ERK_stim_traj_total_param(p, model, max_time, y0, output_states):
        traj = solve_traj_local(model, y0, p, max_time, times/args.time_conversion_factor, args.rtol, args.atol)

        # return normalized trajectory
        total_ERK = p[args.total_ERK_idx]
        return [np.sum(traj[output_states,:], axis=0) / total_ERK], traj
    
    if args.total_ERK_idx != None:
        ERK_stim_traj = ERK_stim_traj_total_param
    else:
        ERK_stim_traj = ERK_stim_traj_total_state

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
    try:
        model_func = lambda prior_param_dict, sol_op, sol_op_Rap1KD, data, \
            data_std: eval(args.pymc_model)(prior_param_dict, sol_op, \
            sol_op_Rap1KD, [data_CYTO], [data_PM], [data_std_CYTO], \
                [data_std_PM],  [data_CYTO_RAP1i], [data_PM_RAP1i], \
                [data_std_CYTO_RAP1i], [data_std_PM_RAP1i])
    except OSError as e:
        print('Warning Pymc model {} not found'.format(args.pymc_model))
        raise
        
    pymc_model = build_pymc_model_local(prior_param_dict, None, y0_EGF_ins[0], 
                    y0_EGF_ins_Rap1_KD[0], ERK_indices, 
                    np.max(times/args.time_conversion_factor), diffrax.ODETerm(model), 
                    simulator=ERK_stim_traj, data_sigma=None, model_func=model_func,)

    if args.skip_prior_sample:
        create_prior_predictive_local(pymc_model, args.model, data_CYTO, \
            data_PM, data_std_CYTO, data_std_PM, data_CYTO_RAP1i, data_PM_RAP1i, \
            data_std_CYTO_RAP1i, data_std_PM_RAP1i, times/data_time_to_mins, \
            inputs_CYTO, args.savedir)

    # SMC sampling
    if args.skip_sample:
        posterior_idata = smc_pymc(pymc_model, args.model, args.savedir, 
                    nsamples=args.nsamples, ncores=args.ncores, threshold=0.5, chains=args.nchains,)
    else:
        posterior_idata, _ = load_smc_samples_to_idata(args.savedir + args.model + '_smc_samples.json')
    
    # trace plots and diagnostics
    plot_sampling_trace_diagnoses(posterior_idata, args.savedir, args.model)

    # posterior predictive samples
    create_posterior_predictive_local(pymc_model, posterior_idata, args.model, data_CYTO, data_PM, data_std_CYTO, data_std_PM, times/data_time_to_mins, inputs_CYTO, args.savedir)
    
    print('Completed {}'.format(args.model))

if __name__ == '__main__':
    main()