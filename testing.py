import ticktack
import ode
import numpy as np
from jax import numpy as jnp

parameters = np.zeros(6)
parameters[0] = 7.044873503263437   # Mean of the sinusoidal production 
parameters[1] = 0.18                # Amplitude of the sinusoidal production 
parameters[2] = 11.0                # Period of the sinusoidal production 
parameters[3] = 1.25                # Phase of the sinusoidal production 
parameters[4] = 120.05769867244142  # Height of the super-gaussian 
parameters[5] = 12.0                # Width of the super-gaussian 

projection = np.zeros(11)
projection[0] = 0.7
projection[1] = 0.3

def production(t):    
    return parameters[0] * (1 + parameters[1] * \
        jnp.sin(2 * jnp.pi / parameters[2] * t + parameters[3])) + \
        parameters[4] * jnp.exp(- (parameters[5] * (t - 775)) ^ 16)

cbm = ticktack.load_presaved_model("Guttler14", production_rate_units="atoms/cm^2/s")
cbm.compile()   # Constructing the transfer operator 

def dydx(y, t):
    return cbm._matrix * y + production(t) * projection
    
ode._bosh_odeint(dydx, 1e-6, )