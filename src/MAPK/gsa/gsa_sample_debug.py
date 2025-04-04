from os import environ
environ['OMP_NUM_THREADS'] = '1'

import jax
import jax.numpy as jnp
import diffrax
import sys
import argparse

from SALib.sample import morris as morris_sample

sys.path.append("../models/")
from huang_ferrell_1996 import *
from bhalla_iyengar_1999 import *
from kholodenko_2000 import *
from levchenko_2000 import *
from brightman_fell_2000 import *
from schoeberl_2002 import *
from hatakeyama_2003 import *
from hornberg_2005 import *
from birtwistle_2007 import *
from orton_2009 import *
from vonKriegsheim_2009 import *
from shin_2014 import *
from ryu_2015 import *
from kochanczyk_2017 import *
from dessauges_2022 import *

sys.path.append("../")
from utils import *

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)

# print out device count
n_devices = jax.local_device_count() 
print(jax.devices())
print('Using {} jax devices'.format(n_devices))

##############################
# def arg parsers to take inputs from the command line
##############################
def parse_args():
    """ function to parse command line arguments
    """
    parser=argparse.ArgumentParser(description="Generate Morris samples for the specified model.")
    parser.add_argument("model", type=str, help="model to process.")
    parser.add_argument("--analyze_params", type=str, help="Comma separated list of additional parameters to include in the analysis.")
    parser.add_argument("--max_time", type=int, help="Max time to simulate the model.")
    parser.add_argument("--savedir", type=str, default='/oasis/tscc/scratch/nlinden/', 
    help="Path to save results.")
    parser.add_argument("--n_samples", type=int, default=256, help="Number of samples to generate. Defaults to 256. Must be a factor of 2.")
    parser.add_argument("--multiplier", type=float, default=0.25, help="Multiplier to use for the Morris sampling. Must be between 0 and 1.")
    args=parser.parse_args()
    return args


##############################
# def vmap function to solve for the steady-state solution of the ODE system
##############################
@jax.jit
def solve_ss(model_dfrx_ode, y0, params, t1):
    """ simulates a model over the specified time interval and returns the 
    calculated steady-state values.
    Returns an array of shape (n_species, 1) """
    dt0=1e-3
    event_rtol=1e-6
    event_atol=1e-6
    solver = diffrax.Kvaerno5()
    event=diffrax.SteadyStateEvent(event_rtol, event_atol)
    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6)
    t0 = 0.0
    dt0 = 1e-3

    sol = diffrax.diffeqsolve(
        model_dfrx_ode, 
        solver, 
        t0, t1, dt0, 
        y0, 
        stepsize_controller=stepsize_controller,
        discrete_terminating_event=event,
        args=tuple(list(params)),
        max_steps=None,
        throw=False,)
    
    return jnp.array(sol.ys)

@jax.jit
def solve_traj(model_dfrx_ode, y0, params, t1, times):
    """ simulates a model over the specified time interval and returns the 
    calculated steady-state values.
    Returns an array of shape (n_species, 1) """
    dt0=1e-3
    solver = diffrax.Kvaerno5()
    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6)
    t0 = 0.0
    dt0 = 1e-3
    saveat = saveat=diffrax.SaveAt(ts=times)

    sol = diffrax.diffeqsolve(
        model_dfrx_ode, 
        solver, 
        t0, t1, dt0, 
        y0, 
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        args=tuple(list(params)),
        max_steps=60000,
        throw=False,)
    
    return jnp.array(sol.ys)

# vmap it over the parameters
# assume that the first dimension of params is the number of samples
# and the second dimension is the number of parameters
vsolve_ss = jax.vmap(solve_ss, in_axes=(None, None, 0, None))
psolve_ss = jax.pmap(vsolve_ss, in_axes=(None, None, 0, None))

vsolve_traj = jax.vmap(solve_traj, in_axes=(None, None, 0, None, None))
psolve_traj = jax.pmap(vsolve_traj, in_axes=(None, None, 0, None, None))


def main():
    """ main function to execute command line script functionality.
    """
    args = parse_args()
    print('Processing model {}.'.format(args.model))

    # try calling the model
    try:
        model = eval(args.model + '()')
    except:
        print('Warning Model {} not found. Skipping this.'.format(args.model))

    # get parameter names and initial conditions
    pdict, plist = model.get_nominal_params()
    y0_dict, y0 = model.get_initial_conditions()

    # get the params to analyze
    analyze_params = args.analyze_params.split(',')
    param_idxs = jnp.array([list(pdict.keys()).index(p) for p in analyze_params])

    # get list of nominal vals for ID params
    analyze_nominal_params = jnp.array([pdict[p] for p in analyze_params])

    # define the bounds
    bounds = [[p*(1-args.multiplier), p*(1+args.multiplier)] for p in analyze_nominal_params]

    # set up the problem dictionary for SALib
    problem = {
        'num_vars': len(analyze_nominal_params),
        'names': analyze_params,
        'bounds': bounds
    }
    print(problem)


    # generate the samples
    n_levels = 4
    samples = morris_sample.sample(problem, args.n_samples, n_levels, seed=1234)
    jnp.save(args.savedir + '{}_morris_sample.npy'.format(args.model), samples)

    full_samples = jnp.repeat(jnp.array([plist]), samples.shape[0], axis=0)

    print(full_samples[1:3,:])

    for i in range(samples.shape[0]):
        full_samples = full_samples.at[i, param_idxs].set(samples[i])
   
    # RUN the samples
    dfrx_ode = diffrax.ODETerm(model)

    # reshape the sample to be (n_devices, n_samples/n_devices, n_params) so that
    # pmap doesn't complain
    reshaped_sample = full_samples.reshape((n_devices, int(full_samples.shape[0]/n_devices), full_samples.shape[-1]))


    dt0=1e-3
    event_rtol=1e-6
    event_atol=1e-6
    solver = diffrax.Kvaerno5()
    event=diffrax.SteadyStateEvent(event_rtol, event_atol)
    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6)
    t0 = 0.0
    dt0 = 1e-3

    sol = diffrax.diffeqsolve(
        dfrx_ode, 
        solver, 
        t0, args.max_time, dt0, 
        y0, 
        stepsize_controller=stepsize_controller,
        discrete_terminating_event=event,
        args=tuple(list(reshaped_sample[0,0,:])),
        max_steps=None,
        throw=False,)

    # sol = psolve_ss(dfrx_ode, y0, reshaped_sample, args.max_time)
    

    # reshape back to (n_samples, n_species)
    n_dev, n_samp_per_dev, n_states, n_dim = sol.shape
    sol = sol.reshape((n_dev*n_samp_per_dev, n_states, n_dim))

    # save the steady-state values
    jnp.save(args.savedir + '{}_morris_ss.npy'.format(args.model), sol)

    print('Completed {}'.format(args.model))

if __name__ == '__main__':
    main()

