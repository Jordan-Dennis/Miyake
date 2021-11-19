import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams

import ticktack
import jax.numpy as jnp
from jax import jit
from ticktack import fitting

rcParams['figure.figsize'] = (10.0, 5.0)    # Choosing the figure size

@jit
def sine(t):
    """
    Fits the underlying solar cycle model, as the basic production function.
    """
    prod =  1.87 + 0.7 * 1.87 * jnp.sin(2 * jnp.pi / 11 * t + jnp.pi/2)
    prod = prod * (t>=sf.start) + (1.87 + 0.18 * 1.87 * jnp.sin(2 * jnp.pi / 11 * sf.start + jnp.pi/2)) * (1-(t>=sf.start))
    return prod

cbm = ticktack.load_presaved_model('Guttler14', production_rate_units = 'atoms/cm^2/s') # Loads Carbon Box Model
# %% This cell fills the unfilled information in the class instance
sf = fitting.SingleFitter(cbm)  # Generates model
sf.prepare_function(f=sine) # Generates Production function 
sf.time_data = jnp.arange(200, 230) # Model time series
sf.d14c_data_error = jnp.ones((sf.time_data.size,))
sf.start = np.nanmin(sf.time_data)  # Start time 
sf.end = np.nanmax(sf.time_data)    # End time
sf.resolution = 1000                # Number of iterations over time series 
sf.burn_in_time = jnp.linspace(sf.start-1000, sf.start, sf.resolution)  # Array of times for burn in 
sf.time_grid_fine = jnp.arange(sf.start, sf.end, 0.05)  # Array of times
sf.time_oversample = 1000   
sf.offset = 0
sf.gp = True
sf.annual = jnp.arange(sf.start, sf.end + 1)
sf.mask = jnp.in1d(sf.annual, sf.time_data)[:-1]

# %% Plots simulated Carbon 14 concentration changes with random Gaussian noise
np.random.seed(0)
d14c = sf.dc14()    # All right so ODE int is somewhere in here. So this function is going to 
# have a very long path of wrappers and dumb shit like that. I will pull all of that out of the 
# Julia implementation but otherwise I will try and preserve everything that we see here. 
# I honestly have no fucking idea what is going on though since the MCMC takes for fucking 
# ever.
noisy_d14c = np.array(d14c) + np.random.randn(d14c.size) # add unit gaussian noise
noisy_d14c = np.append(noisy_d14c, noisy_d14c[-1]) # for compatibility with ticktack code

plt.figure(1)
plt.plot(sf.time_data[:-1], d14c, "-g", label="true d14c", markersize=10)
plt.errorbar(sf.time_data[:-1], noisy_d14c[:-1], yerr=sf.d14c_data_error[:-1], 
             fmt="k", linestyle='none', fillstyle="none", capsize=2, label="d14c")
plt.ylabel("$\Delta^{14}$C (‰)")
plt.xlabel("Calendar Year")
plt.legend()

# %% Determines the production function from the simulated data using a gaussian process
sf.d14c_data = jnp.array(noisy_d14c) # Assigning the production model to the class instance
sf.prepare_function(model="control_points") # A gaussian process interpolator
sf.control_points_time = jnp.arange(sf.start, sf.end, 2) # control points for the gaussian process occuring every other year 

print("%d years in the period, %d control-points used" % \
    (sf.annual.size, sf.control_points_time.size))

soln = sf.fit_ControlPoints(low_bound=0.)

plt.figure(2) # A plot of the production function sampled via gaussian process from the simulated carbon 14 changes
plt.plot(sf.control_points_time, soln.x, ".b", markersize=7, label="recovered production rate")
plt.plot(sf.time_grid_fine, sine(sf.time_grid_fine), 'k', label='true production rate')
plt.title("Recovery test")
plt.ylabel("Production rate ($cm^2s^{-1}$)")
plt.xlabel("Calendar Year")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True)

# %% A superposition of the previous two plots 
# fig = plt.figure(figsize=(12.0,8.0))
# plt.plot(sf.time_data[:-1], d14c, "-r", label="true d14c", markersize=10)
# plt.errorbar(sf.time_data[:-1], noisy_d14c[:-1], yerr=sf.d14c_data_error[:-1], 
#              fmt="k", linestyle='none', fillstyle="none", capsize=2, label="d14c")
# plt.plot(sf.time_data[:-1], sf.dc14(soln.x), ".b", label="recovered d14c", markersize=10)
# plt.title("Recovery test")
# plt.ylabel("$\Delta^{14}$C (‰)")
# plt.xlabel("Calendar Year")
# plt.legend(fancybox=True)

# Running the emcee mcmc model to determine the posterior distribtuion of the points.
sampler = sf.sampling(soln.x, sf.log_joint_gp, burnin=500, production=1000)
sf.plot_recovery(sampler, time_data=sf.time_grid_fine, true_production=sine(sf.time_grid_fine))
plt.show()