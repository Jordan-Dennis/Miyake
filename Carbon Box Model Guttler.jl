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

λ = Diagonal([log(2) / 5730 for i in 1:11]);            # Constructing the decay matrix
F = transpose(F) ./ N;                                  # The proportion flux
TO = transpose(F) - Diagonal(vec(sum(F, dims=2))) - λ;  # Construncting the transfer operator

equilibrium_production = 3.747273140033743 * 1.88;  # Correct equilibrium production
equilibrium = TO \ (-equilibrium_production * P);   # Brehm equilibriation for Guttler 2014

∇(y, p, t) = vec(TO * y + production(t) * P);                                   # Calculates the derivative
solution = solve(ODEProblem(∇, equilibrium, (-360.0, 760.0)), reltol = 1e-6);   # Solving the ode over a burn-in period
solution = @time solve(ODEProblem(∇, solution[end], (760.0, 790.0)), reltol = 1e-6);  # Solving the ODE over the period of interest 

time = Array(solution.t);               # Storing the time sampling 
solution = Array(solution)[2, 1:end];   # Storing the solution #? Is this wasteful of the other information?

#* I want to improve this binning method 
function bin(time_series::Vector{Float64}, solution_vector::Vector{Float64})::Vector{Float64}
    binned_solution = Vector{Float64}(undef, 0);   # Setting a vector to hold the bins 
    bin_start = 760.0; # The start of the bin 
    bin_sum = 0.0; # The sum of the values in the bin 
    bin_num = 0.0; # The number of entries in the bin
    for t in 1:size(solution_vector)[1]    # Looping through the entires 
        if bin_start < time_series[t] < (bin_start + 1.0)    # Checking if the value is in the bin 
            bin_sum += solution[t]; # Incrementing the sum in the bin
            bin_num += 1.0; # Incrementing the count of the entires
        else
            bin_start += 1.0; # Incrementing the bin start by one year 
            append!(binned_solution, bin_sum / bin_num);    # Adding the anual mean to the bon matrix 
            bin_sum = solution[t]; # Starting the new count 
            bin_num = 1.0; # Startingn the new count
        end
    end
    return binned_solution
end

binned_solution = bin(time, solution);  # Binning the results into years 

# residual with respect to the median of all of them (median more stable so outliers)
# Tree uptake is not accounted for. 

plot(time, solution[1:end]);   # Plotting the troposphere C14 concentrations
# plot!(760:790, binned_solution); # Adding the means to the plot
# end 