using Plots; gr();              # Moving Plot( ) into the namespace
using HDF5;                     # for .hd5 file manipulation
using DifferentialEquations;    # Provides a variety of differential solvers
using LinearAlgebra: Diagonal;  # Efficient Diagonal matrixes
using Statistics;               # For the data analysis 
using Threads

"""
Takes time series data and calculates the average of each year.
"""
function bin(time_series::Vector{Float64}, solution_vector::Vector{Float64})::Vector{Float64}
    local binned_solution = Vector{Float64}(undef, 0);  # Setting a vector to hold the bins 
    local whole_times = @. floor(time_series);          # Creating a vector of discrete time.
    for whole_time in unique(whole_times)                                           # Looping over the unique elements discrete times 
        local indexes = findall(whole_times .== whole_time);                        # Getting the indexes of the entries 
        append!(binned_solution, sum(solution_vector[indexes]) / length(indexes));  # Appending to binned_solution
    end
    return binned_solution
end

"""
Calculates the production of C14 based on the projection based on the model 
presented in the _Guttler 2014_ paper.
"""
function production(year)                                       
    local gh::Float64 = 20 * 1.60193418235;  # height of the super-gaussian  
    local uf::Float64 = 3.747273140033743;   # unit correcting factor
    return uf * (1.88 + 0.18 * 1.88 * sin(2 * π / 11 * year + 1.25) +   # Sinusoidal production 
        gh * exp(- (12 * (year - 775)) ^ 16));                             # Super gaussian
end

"""
Opens a file 'ODE comparison.txt' and writes the binned data to the file 
in a csv format.
"""
function write_txt(data::Vector{Float64})::Nothing
    open("ODE comparison.txt", "a+") do # Opening a file to store the results 
        write(760.0:790.0)              # Writing the time series data to the file
        for year in binned_solution;        # Looping over the binned values 
            write()
        end
    end
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
binned_solution = bin(time, solution);  # Binning the results into years 


# end 