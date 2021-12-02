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
#? If I do I might have to move this onto a 1-liner in which case I will have to document with #=
function production(year::Float64)::Vector{Float64}
    local R = 1.88;                                             # Steady State Production in ^{14}Ccm^{2}s^{-1}
    local T = 2 * π / 11;                                       # Period of solar cycle years 
    local Φ = 1.25;                                             # Phase of the solar cycle 
    local unit_factor = 3.747273140033743;                      # Corrects the units.

     unit_factor * (R + 0.18 * R * sin(T * year + Φ));    # correcting the units

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

# function main() # ::Vector{Float64}
Guttler2014 = h5open("Guttler14.hd5");              # Opening the HDF5 file
F = Guttler2014["fluxes"][1:end, 1:end];            # Retrieving the flux matrix 
P = Guttler2014["production coefficients"][1:end];  # Retrieving the projection of the production 
N = Guttler2014["reservoir content"][1:end, 1:end]; # The C14 reserviour contents 
close(Guttler2014);                                 # Closing the file 
#! fml everything has come through as a transpose because julia is smartter than python

#? I think this is very inefficient and I should just use a for loop in the TO section
λ = Diagonal([log(2) / 5730 for i in 1:11]);    # Constructing the decay matrix

F = transpose(F) ./ N;                        # The proportion flux
#* The same up until this point 

TO = F - Diagonal(vec(sum(F, dims=2))) - λ;   # Construncting the transfer operator
#! new_c_14 is the problem

#! need to check that minimum is returning the value I want.
equilibrium = TO \ (- minimum(calculate_equilibrium(TO, P)) * P);   # optimisation from the Guttler2014

∇(y, p, t) = vec(TO * y + production(t) * P);   # Calculates the derivative and returns

solution = solve(ODEProblem(∇, equilibrium, (760.0, 790.0)), Rosenbrock23())   # Solving the ode

troposphere = Vector{Float64}(undef, 228);              # Empty vector to hold the troposphere data
for i in 1:228; troposphere[i] = solution[i][2]; end    # Catching the troposphere value

plot(troposphere)   # Plotting the troposphere C14 concentrations
# end 