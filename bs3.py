import numpy as np

# I am going to start implementing this as a function. This will need to be included within a class later on anyway. 

def bs3(x0: np.array, t: np.array, f: function, reltol: np.float64):
    """
    It is assumed that dy/dt = f(t, x_t). For now anyway 
    """
    x = np.zeros((len(x0), len(t))) # Storage for the numerical solution 
    h = reltol                      # Setting the initial tolerance 

    for i, time in enumerate(t):
        k1 = f(time, x[i])
        k2 = f(time + 1 / 2 * h, x[i] + 1 / 2 * h * k1)
        k3 = f(time + 3 / 4 * h, x[i] + 3 / 4 * h * k2)

        x[i + 1] = x[i] + 2 / 9 * h * k1 + 1 / 3 * h * k2 + 4 / 9 * h * k3 # Third order approximation 

        k4 = f(time + h, x[i + 1])

        z = x[i] + 7 / 24 * h * k1 + 1 / 4 * h * k2 + 1 / 3 * h * k3 + 1 / 8 * h * k4 # Second order guess 

        h = 0.9 * h * abs(x[i + 1] - z) / reltol    # New timestep 
    
    return x