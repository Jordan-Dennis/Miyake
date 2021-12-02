using Plots; gr();              # Moving Plot( ) into the namespace
using HDF5;                     # for .hd5 file manipulation
using DifferentialEquations;    # Provides a variety of differential solvers
using Optim: optimize, LBFGS;   # Importing the optimisation library and solver #? :foward
using LinearAlgebra: Diagonal;  # Efficient Diagonal matrixes

"""
Production:
 - The production function of C14 as a Vector{Float64} based on the projection
 of the production function into the system. The production function is stated 
    here with a number of Parameters that can be retrieved from the Guttler
    2014 paper

Parameters:
 - year::Float64 → The current year 

Returns:
 - Float64 → The production during the current year
"""
#! I want to get rid of all this extra parameter definition
#? If I do I might have to move this onto a 1-liner in which case I will have to document with #=
function production(year)
    local R = 1.88;                                             # Steady State Production in ^{14}Ccm^{2}s^{-1}
    local T = 2 * π / 11;                                       # Period of solar cycle years 
    local Φ = 1.25;                                             # Phase of the solar cycle 
    local unit_factor = 3.747273140033743;                      # Corrects the units.
    return unit_factor * (R + 0.18 * R * sin(T * year + Φ));    # correcting the units
end

# function main() # ::Vector{Float64}
Guttler2014 = h5open("Guttler14.hd5");              # Opening the HDF5 file
F = Guttler2014["fluxes"][1:end, 1:end];            # Retrieving the flux matrix 
P = Guttler2014["production coefficients"][1:end];  # Retrieving the projection of the production 
N = Guttler2014["reservoir content"][1:end, 1:end]; # The C14 reserviour contents 
close(Guttler2014);                                 # Closing the file 

#? I think this is very inefficient and I should just use a for loop in the TO section
λ = Diagonal([log(2) / 5730 for i in 1:11]);            # Constructing the decay matrix
F = transpose(F) ./ N;                                  # The proportion flux
TO = transpose(F) - Diagonal(vec(sum(F, dims=2))) - λ;  # Construncting the transfer operator

#* I might need to copy their structure here for now I'm just gonna thug life it
equilibrium_production = 3.747273140033743 * 1.88;  # Correct equilibrium production
equilibrium = TO \ (-equilibrium_production * P);   # Brehm equilibriation for Guttler 2014
#* Same until here

∇(y, p, t) = vec(TO * y + production(t) * P);   # Calculates the derivative
solution = solve(ODEProblem(∇, equilibrium, (-360.0, 760.0)));  # Solving the ode over a burn-in period
solution = solve(ODEProblem(∇, solution[end], (760.0, 790.0))); # Solving the ODE over the period of interest 

# Burn in period up to 760
# fitting.py 560 
# Bin takes the values and averages them over the entire year.
# time sampling to the wrong bin 
# I need to work out how to specify the time mesh
# Impulse response function as approximate solution 
# residual with respect to the median of all of them (median more stable so outliers)

# Tree uptake is not accounted for. 

# troposphere = Vector{Float64}(undef, 28);              # Empty vector to hold the troposphere data
# for i in 1:28; troposphere[i] = solution[i][2]; end    # Catching the troposphere value

# plot(troposphere)   # Plotting the troposphere C14 concentrations
# end 