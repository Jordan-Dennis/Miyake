using Plots; gr();              # Moving Plot( ) into the namespace
using SparseArrays;             # For sparse arrays used in the flux matrix
using HDF5;                     # for .hd5 file manipulation
using DifferentialEquations;    # Provides a variety of differential solvers
using Optim: optimize, LBFGS;   # Importing the optimisation library and solver #? :foward
using LinearAlgebra: Diagonal;  # For efficient diagonal computational

"""
Read Data:
 - Reads the contents of file_name, which is assumed to have the extension .hd5

Parameters:
 - file_name::String → The name of the file as it appears in the directory

Returns:
 - F::Matrix{Float64} → The matrix of flux values in Gt/yr 
 - P::Vector{Float64} → A projection vector for the production function
"""
function read_data(file_name::String = "Guttler2014.hd5")::AbstractArray
    local Guttler2014 = h5open(file_name);                      # Opening the HDF5 file
    local F = Guttler2014["fluxes"][1:end, 1:end];              # Retrieving the flux matrix 
    local P = Guttler2014["production coefficients"][1:end];    # Retrieving the projection of the production 
    local N = Guttler2014["reservoir content"][1:end, 1:end];   # The C14 reserviour contents 
    local λ = Guttler2014["decay coefficients"][1:end, 1:end];  # The decay constants as the diagonal elements
    close(Guttler2014);                                         # Closing the file 
    return sparse(F), P, N, Diagonal(λ);                        # Diagonal and sparse are for speed
end

"""
Production:
 - The production function of C14.

Parameters:
 - year::Float64 → The current year 

Returns:
 - Float64 → The production during the current year
"""
function production(year::Float64)::Float64         # The production function
    local P = 1.88;                                 # Steady State Production in ^{14}Ccm^{2}s^{-1}
    local T = 11;                                   # Period of solar cycle years 
    local Φ = 1.25;                                 # Phase of the solar cycle 
    local P += 0.18 * P * sin(2 * π / T * y + Φ);   # Evaluating the production fucntion at the time
    return P * 14.003242 / 6.022 * 5.11 * 31536. / 1.e5;    # Correction the units
end

#// Mock objective function
#? Should I create an overarching function to perform the entire optimisation
"""
Troposphere Residual:
 - A function that is passed to the Optim.optimize function

Parameters:
 - TC14::Vector{Float64} → 
"""
function troposphere_residual(TC14::Vector{Float64})
    rss = Vector{Float64}(undef, 1);    # A vector to house the sum of square 
    rss[1] = (steady_state[2] - TC14) ^ 2; # TC14 is the target C14 and steady_state[2] is the equilibrium tropospher position  
    return rss
    #! still returning a Float64
end

troposphere_concentration = Vector{Float64}(undef, 1);  # An initial position Vector{Float64}
troposphere_concentration[1] = 6.0; # The initial position. 6.0 is a standard choice

optimized_residuals = optimize(troposphere_residual, troposphere_concentration, LBFGS());    #* optimisation from the Guttler2014
#* I need beter variable names in this section 

#// This is effectively the derivative function within run.
"""
∇:
 - Calculates the derivative given the system y::Vector{Float64} and
 time t::Float64

Parameters:
 - y::Vector{Float64} → A vector specifying the position of the system
 - t::Float64 → The time in years.

Returns:
 - Vector{Float64} → The ∇ at the position y
"""
function ∇(y::Vector{Float64}, t::Float64)::Vector{Float64}
    return transfer_operator * y + production(t);   # Calculates the derivative and returns
end

#! From here I start at steady_state

function main()
    F, P, N, λ = read_data();

    C14F = F ./ transpose(N);   # Proportion flow of C14 (axis=1 is RowSum)
    NC14F = Vector{Float64}(undef, 11); # Empty vector with the goal of applying |> latter 
    C14Content = sum(C14F, dims=2); # The C14 content in each reserviour 
    for i in 1:11; NC14F[i] = C14Content[i]; end;    # Filling NC14F with the C14Content elements

    #! I need to make a flow chart and work out how I can use pipe to achieve this without the random declarations 

    NC14F = Diagonal(NC14F);  # C14 content of each reserviour
    transfer_operator = transpose(C14F) - NC14F - λ;    #* I want to use the |> here

    #// Mock equilibriate Brehm 
    steady_state = transfer_operator \ (- 1.88 * P); #! The 1.88 is the steady state quoted in the paper 
end
ODESolution = DifferentialEquations. (∇, steady_state);

# I need to set up the precompiliation down here 