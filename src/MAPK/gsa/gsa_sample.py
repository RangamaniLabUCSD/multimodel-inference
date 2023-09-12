import numpy as np
import multiprocessing as mp
import math

from os import environ
environ['OMP_NUM_THREADS'] = '1'

# use all available cores (do before loading jax)
# required for parallel cpu runs
n_cores = mp.cpu_count()
cpu_mult = 2
# print('Using {} cores'.format(n_cores))
n_devices = int(2**np.ceil(math.log(cpu_mult*n_cores, 2))) # sets n_devices to the next largest power of 2
# print('Set {} XLA devices'.format(n_devices))

xla_flag = '--xla_force_host_platform_device_count={}'.format(n_devices)
environ['XLA_FLAGS']=xla_flag


import jax
import jax.numpy as jnp
import diffrax
import sys

sys.path.append("../models/")
from huang_ferrell_1996 import *
from schoeberl_2002 import *
from birtwistle_2007 import *
from levchenko_2000 import *
from brightman_fell_2000 import *
from hatakeyama_2003 import *
from hornberg_2005 import *
from shin_2014 import *

# use 64bit and CPUmode in jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# print out device count
n_devices = jax.local_device_count() 
print('Using {} jax devices'.format(n_devices))

from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

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
def solve_traj(model_dfrx_ode, y0, params, t1):
    """ simulates a model over the specified time interval and returns the 
    calculated steady-state values.
    Returns an array of shape (n_species, 1) """
    dt0=1e-3
    solver = diffrax.Kvaerno5()
    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6)
    t0 = 0.0
    dt0 = 1e-3

    sol = diffrax.diffeqsolve(
        model_dfrx_ode, 
        solver, 
        t0, t1, dt0, 
        y0, 
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

vsolve_traj = jax.vmap(solve_traj, in_axes=(None, None, 0, None))
psolve_traj = jax.pmap(vsolve_traj, in_axes=(None, None, 0, None))


##############################
# loop over models and run the analysis
##############################
# model_list = ['shin_2014', 'huang_ferrell_1996', 'schoeberl_2002', 
#               'brightman_fell_2000', 'birtwistle_2007', 'hatakeyama_2003', 'hornberg_2005']
# sustained = [True, True, False, True, True, False, False]
# sim_times = [200, 1000, 60, 60, 1800, 1800, 6000]

model_list = ['birtwistle_2007', 'hatakeyama_2003', 'hornberg_2005']
sustained = [True, False, False]
sim_times = [1800, 1800, 6000]


for model_name, time, sus in zip(model_list, sim_times, sustained):
    print('Running: ', model_name)
    try:
        model = eval(model_name + '()')
    except:
        print('Warning Model {} not found. Skipping this.'.format(model_name))

    # get parameter names and initial conditions
    pdict, plist = model.get_nominal_params()
    y0 = model.get_initial_conditions()

    # define the bounds
    multiplier = 0.25
    bounds = [[p*(1-multiplier), p*(1+multiplier)] for p in plist]

    # set up the problem dictionary for SALib
    problem = {
        'num_vars': len(plist),
        'names': list(pdict.keys()),
        'bounds': bounds
    }

    # generate the samples
    n_samples = 256
    n_levels = 4
    sample = morris_sample.sample(problem, n_samples, n_levels, seed=1234)
    np.save('{}_morris_sample.npy'.format(model_name), sample)

    # run the model using vmap
    print('Running {} samples'.format(sample.shape[0]))
    dfrx_ode = diffrax.ODETerm(model)

    # reshape the sample to be (n_devices, n_samples/n_devices, n_params) so that
    # pmap doesn't complain
    reshaped_sample = sample.reshape((n_devices, int(sample.shape[0]/n_devices), sample.shape[-1]))

    if sus:
        sol = psolve_ss(dfrx_ode, y0, reshaped_sample, time)
    else:
        sol = psolve_traj(dfrx_ode, y0, reshaped_sample, time)

    # reshape back to (n_samples, n_species)
    n_dev, n_samp_per_dev, n_states, n_dim = sol.shape
    sol = sol.reshape((n_dev*n_samp_per_dev, n_states, n_dim))

    # save the steady-state values
    np.save('{}_morris_ss.npy'.format(model_name), sol)

    print('Completed {}'.format(model_name))