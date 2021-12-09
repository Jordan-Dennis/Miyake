import numpy as np
from time import process_time

# I am going to start implementing this as a function. This will need to be included within a class later on anyway. 

def bs3(x0: np.array, nsteps: np.int16, f, reltol: np.float64, time: np.float64):
    """
    It is assumed that dy/dt = f(t, x_t). For now anyway 
    """
    x = np.zeros((len(x0), nsteps)) # Storage for the numerical solution 
    x[:, 0] = x0                       # Storing initial positions
    h = reltol                      # Setting the initial tolerance 
    
    t = np.zeros(nsteps)    # Saving the times
    t[0] = time             # Adding the first time

    for i in range(1, nsteps):
        k1 = f(time, x[:, i - 1])    # First calc 
        k2 = f(time + 1 / 2 * h, x[:, i - 1] + 1 / 2 * h * k1)
        k3 = f(time + 3 / 4 * h, x[:, i - 1] + 3 / 4 * h * k2)

        x[:, i] = x[:, i - 1] + 2 / 9 * h * k1 + 1 / 3 * h * k2 + 4 / 9 * h * k3 # Third order approximation 

        k4 = f(time + h, x[:, i])

        z = x[:, i - 1] + 7 / 24 * h * k1 + 1 / 4 * h * k2 + 1 / 3 * h * k3 + 1 / 8 * h * k4 # Second order guess 

        h = 0.9 * h * np.linalg.norm(x[:, i] - z) / reltol    # New timestep 
        time = time + h                             # Updating the time 

        t[i] = time # saving the time 
    
    return t, x

# Testing architecture 
import ticktack

cbm = ticktack.load_presaved_model("Guttler14", production_rate_units="atoms/cm^2/s")
cbm.compile()
x0 = cbm.equilibrate(production_rate=1.88)
TO = np.array(cbm._matrix)

def production(t: np.float64, p: np.array):
    """
    The production function of the system
    """
    return p[0] * (1 + p[1] * np.sin(2 * np.pi / p[2] * t + p[3])) 
        # p[4] * np.exp(- (p[5] * (t - 775) ^ 16))

def dydx(t: np.float64, xt: np.array):
    """
    The derivative of the system
    """
    parameters = np.zeros(6)            # Array for the parameters of the production function
    parameters[0] = 7.044873503263437   # The intercept of the sinusoidal production term
    parameters[1] = 0.18                # The amplitude of the sinusoidal production term 
    parameters[2] = 11.0                # Setting period of the sinusoid 
    parameters[3] = 1.25                # The phase shift of the sinusoid
    parameters[4] = 120.05769867244142  # The height of the super gaussian 
    parameters[5] = 12.0                # Width of the super-gaussian

    projection = np.zeros(11)   # Array for the projection of the production function
    projection[0] = 0.7         # Projection in the stratosphere
    projection[1] = 0.3         # Projection in the troposphere

    return TO @ xt + production(t, parameters) * projection

### Running that fucking shit 
timer = process_time()
t, x = bs3(x0, 1000, dydx, 1e-6, 760.0)
print(process_time() - timer)

import matplotlib.pyplot as plt 
plt.plot(t, x)
plt.show()
