using DifferentialEquations
using Random
using Plots
using PyCall

miyake12_csv_https = "https://raw.githubusercontent.com/"*
    "SharmaLlama/ticktack/main/notebooks/datasets/miyake12.csv"  # Address for the raw .csv file

# Seeting the coda ticktack environmnet as home directory
ENV["PYTHON"] = "/home/jordan/anaconda3/envs/ticktack/bin/python"
default(fmt = :png);    # Don't fucking know 
Random.seed!(12);       # Sets a seed for replicable results
ticktack = pyimport("ticktack");    # Importing ticktack from the conda environmnet
onp = pyimport("numpy");            # Importing numpy from the conda environmnet

cbm = ticktack.load_presaved_model("Guttler14", # Loads the Guttler14 model
    production_rate_units = "atoms/cm^2/s");    # Generates a CarbonBoxModel object
cf = ticktack.fitting.CarbonFitter(cbm);        # Calls load_model to Generate a CarbonBoxModel

# Alright, I'm so fucking confused. load_presaved_model calls load_model which seems to be
# the entire point of calling the CarbonFitter. Ohhh CarbonFitter is much more complex and 
# includes the load_model, CarbonBoxModel

cf.load_data(miyake12_csv_https);   # Loads the raw .csv from GitHub
cf.prepare_function(production="miyake", fit_solar=false);  # Chooses miyake_event_fixed_solar as production function
cbm_mat = onp.array(cbm._matrix);   # Creates a numpy.array from the CarbonBoxModel
coeff = onp.array(cbm._production_coefficients);    # numpy.array from CarbonBoxModel

"""
Parameters:
    p: list -> The parameters of the production function [start_date, duration, phi, area]
    t: ...

Returns:
    production: numpy.array -> Don't ask me whats in it.
"""
function get_production_py(p, t)
    # cf.production was assigned to a function, miyake_event_fixed_solar
    # production_rate is the output from calling miyake_event_fixed_solar
    production_rate = cf.production(t, p[1], p[2], p[3], p[4]); 
    production_rate = cbm._convert_production_rate(production_rate);
    production = onp.array(cbm._production_coefficients * production_rate);

    return production
end

"""
Parameters:
    t: ...
    start_time: Point of initialisation of Miyake event model
    duration: Length of the miyake event model
    area: The area under the Miyake curve

Returns:
    Array -> A super gaussian model for the Miyake event
"""
function super_gaussian(t, start_time, duration, area)
    middle = start_time + duration / 2.;
    height = area / duration;
    return height .* exp.(- ((t .- middle) / (1. ./ 1.93516 * duration)) .^ 16.);
end

"""

"""
function miyake_event_fixed_solar(t, p)
    height = super_gaussian(t, p[1], p[2], p[4]);
    prod = cf.steady_state_production .+ 0.18 .* cf.steady_state_production .* sin.(
        2 .* pi ./ 11 .* t .+ p[3]) .+ height;
    return prod
end

function get_production(t, p)
    prod = miyake_event_fixed_solar(t, p) * 14.003242 / 6.022 * 5.11 * 31536 / 1.e5;
    production = coeff .* prod;
    return production
end

function ticktack_deriv!(du, u, p, t)
    
    test = cbm_mat * u;
    production = get_production(t, p);
#     prod = miyake_event_fixed_solar(t, p) * 14.003242 / 6.022 * 5.11 * 31536 / 1.e5
#     production = coeff .* prod
    
    # This is a ridiculous hack. I hate it and whatever bug this is caused me 2 hrs of misery
    du[1] = test[1] + production[1];
    du[2] = test[2] + production[2];
    du[3] = test[3] + production[3];
    du[4] = test[4] + production[4];
    du[5] = test[5] + production[5];
    du[6] = test[6] + production[6];
    du[7] = test[7] + production[7];
    du[8] = test[8] + production[8];
    du[9] = test[9] + production[9];
    du[10] = test[10] + production[10];
    du[11] = test[11] + production[11];
    return du
    # Diagnostics?
#     print("Time: ", t, " Production ", get_production(p, t)[1]," du[1]: ",  du[1], "\n")
end

u0 = onp.array(cbm.equilibrate(production_rate=cbm.equilibrate(target_C_14=707))); # Initial guess
p = [775., 1/12, pi/2., 81/12]; # Parameters of solar

tspan = (-200., 800.) # Range for ODE to solve over

prob = ODEProblem(ticktack_deriv!,u0,tspan, p);
sol = solve(prob, Tsit5());

burn_in = [sol(t) for t in (760.)]; # Calculate solution for du[1] on times

# times = [760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772,
#        773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785,
#        786, 787]; # Where to save points at

# carbon = [-21.63 , -22.28 , -22.64 , -23.83 , -22.2  , -22.99 , -20.73 ,
#        -21.59 , -25.32 , -25.6  , -25.7  , -24.   , -23.73 , -21.91 ,
#        -23.44 ,  -9.335,  -6.46 ,  -9.7  , -11.17 , -10.31 , -11.1  ,
#        -10.72 , -10.67 ,  -8.63 ,  -9.68 ,  -9.31 , -12.33 , -14.44 ];
