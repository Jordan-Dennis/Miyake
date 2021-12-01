# using Plots; gr();              # Moving Plot( ) into the namespace
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
function production(year::Float64)::Vector{Float64}
    local R = 1.88;                                     # Steady State Production in ^{14}Ccm^{2}s^{-1}
    local T = 11;                                       # Period of solar cycle years 
    local Φ = 1.25;                                     # Phase of the solar cycle 
    local R += 0.18 * R * sin(2 * π / T * year + Φ);    # Evaluating the production fucntion at the time
    return P * 3.747273140033743;                       # correcting the units
end

"""
Calculate Euilibrium:
 - Calculates the equilibrium position of the system given a target C14 concentration 
 in the troposphere. This is done by optimising the residual squared function from the 
 known steady state of the system. The optimise solver is BFGS and is passed a starting 
 position of 6.0, which is a standard value for the reserviours. RSS takes a 
 ::Vector{Float64} as an input as well as an output.

Parameters:
 - TO::Matrix{Float64} → The transfer operator of the system
 - P::Vector{Float64} → The projection of the production function into the system
 
Returns:
 - Float64 → The optimised tropospheric concentration
"""
function calculate_equilibrium(TO::Matrix{Float64}, P::Vector{Float64})
    s = TO \ (-1.88 * P);                                       # Equlibriation as done by Brehm 
    RSS(TC14::Vector{Float64})::Float64 = (s[2] - TC14[1]) .^ 2;# Residual sum of squares for troposphere
    return optimize(RSS, [6.0], LBFGS());                       # 6.0 is a standard starting position for the reserviours
end

# function main()::Vector{Float64}
Guttler2014 = h5open("Guttler2014.hd5");              # Opening the HDF5 file
F = Guttler2014["fluxes"][1:end, 1:end];              # Retrieving the flux matrix 
P = Guttler2014["production coefficients"][1:end];    # Retrieving the projection of the production 
N = Guttler2014["reservoir content"][1:end, 1:end];   # The C14 reserviour contents 
λ = Guttler2014["decay coefficients"][1:end, 1:end];  # The decay constants as the diagonal elements
close(Guttler2014);                                         # Closing the file 

F = F ./ transpose(N);                        # The proportion flux
TO = F - Diagonal(vec(sum(F, dims=2))) - λ;   # Construncting the transfer operator
equilibrium = calculate_equilibrium(TO, P);   # optimisation from the Guttler2014
equilibrium = minimum(equilibrium);           # Getting the equilibrium position
equilibrium = TO \ (-equilibrium * P);        # Equibration for the total system based on the guttler production equilibriation 

∇(y, p, t) = vec(TO * y + production(t) * P);   # Calculates the derivative and returns

solve(ODEProblem(∇, equilibrium, (760.0, 790.0)), Rosenbrock23())   # Solving the ode
# end 