from os import environ
environ['OMP_NUM_THREADS'] = '1'

import diffrax
import jax
import jax.numpy as jnp
import equinox as eqx
import sys
import time

sys.path.append("../models/")
from hornberg_2005 import *

jax.config.update("jax_enable_x64", True)

#n_devices = jax.local_device_count() 
#print('Using {} jax devices'.format(n_devices))
with open('/oasis/tscc/scratch/nlinden/gpu_test_info.txt', 'a') as f:
    f.write('Devices are {}'.format(jax.devices()))

# function to solve the model and get a trajectory
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
vsolve_traj = jax.vmap(solve_traj, in_axes=(None, None, 0, None, None))
psolve_traj = jax.pmap(vsolve_traj, in_axes=(None, None, 0, None, None))

# import model
HB_2005 = hornberg_2005(transient=False)
pdict, plist = HB_2005.get_nominal_params()
y0_dict, y0 = HB_2005.get_initial_conditions()
HB_2005_ode = diffrax.ODETerm(HB_2005)

# compile + solve model
times = jnp.linspace(0, 6000, 501)
start = time.time()
sol =  solve_traj(HB_2005_ode, y0, plist, 6000, times)
end = time.time()

with open('/oasis/tscc/scratch/nlinden/gpu_test_info.txt', 'a') as f:
    f.write('Compile + solves took {} seconds'.format(end-start))

# repeat solve 10 times
nsolves = 100
plist = jnp.repeat(jnp.array([plist]), nsolves, axis=0)
times = jnp.linspace(0, 6000, 501)
start = time.time()
sol =  vsolve_traj(HB_2005_ode, y0, plist, 6000, times)
end = time.time()

with open('/oasis/tscc/scratch/nlinden/gpu_test_info.txt', 'a') as f:
    f.write('VMAP solves took {} seconds'.format(end-start))