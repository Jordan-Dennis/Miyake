using Plots; gr();              # Moving Plot( ) into the namespace
using HDF5;                     # for .hd5 file manipulation
using DifferentialEquations;    # Provides a variety of differential solvers
using LinearAlgebra: Diagonal;  # Efficient Diagonal matrixes
using Statistics;               # For the data analysis 

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
function write_hd5(data::Vector{Float64}, solver::String)::Nothing
    ode_data = h5open("ODE comparison.txt", "cw")   # Opening a file to store the results 
    ode_data[solver] = data;                        # Writing to a new field
    close(ode_data);                                # Closing the file
end

"""
Reads the flux (amounts), production (projection) and reserviour contents from 
a .hd5 file with file_name. It returns the transfer operator and production 
projection 
"""
function read_hd5(file_name::String)::Tuple{Matrix{Float64}, Vector{Float64}}
    local hd5 = h5open(file_name);                      # Opening the HDF5 file
    local F = hd5["fluxes"][1:end, 1:end];              # Retrieving the flux matrix 
    local P = hd5["production coefficients"][1:end];    # Retrieving the projection of the production 
    local N = hd5["reservoir content"][1:end, 1:end];   # The C14 reserviour contents 
    close(hd5);                                         # Closing the file 

    local λ = Diagonal([log(2) / 5730 for i in 1:11]);          # Constructing the decay matrix
    F = transpose(F) ./ N;                                      # The proportion flux
    local TO = transpose(F) - Diagonal(vec(sum(F, dims=2))) - λ;# Construncting the transfer operator
    return TO, P                                           
end

# equilibrium_production = 3.747273140033743 * 1.88;  # Correct equilibrium production
# equilibrium = TO \ (-equilibrium_production * P);   # Brehm equilibriation for Guttler 2014

# ∇(y, p, t) = vec(TO * y + production(t) * P);                                       # Calculates the derivative
# solution = solve(ODEProblem(∇, equilibrium, (-360.0, 760.0)), reltol = 1e-6);       # Solving the ode over a burn-in period
# solution = @time solve(ODEProblem(∇, solution[end], (760.0, 790.0)), reltol = 1e-6);# Solving the ODE over the period of interest 

# time = Array(solution.t);               # Storing the time sampling 
# solution = Array(solution)[2, 1:end];   # Storing the solution
# binned_solution = bin(time, solution);  # Binning the results into years 

function main()
    local (TO, P) = read_hd5("Guttler14.hd5");  # Reading the data into the scope 
end

main(); # Calling the program