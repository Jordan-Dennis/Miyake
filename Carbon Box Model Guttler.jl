using Plots; gr();              # Moving Plot( ) into the namespace
using HDF5;                     # for .hd5 file manipulation
using DifferentialEquations;    # Provides a variety of differential solvers
using LinearAlgebra: Diagonal;  # Efficient Diagonal matrixes

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
    ode_data = h5open("ODE comparison.hd5", "cw")   # Opening a file to store the results 
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

"""
Passed a solver function runs the solver and returns the speed and binned data
"""
function run_solver(solver, ∇::Function, U0::Vector{Float64})
    problem = ODEProblem(∇, U0, (760.0, 790.0));            # Creating the ODEProblem instance
    solution = solve(problem, reltol = 1e-6, solver); # Solving the ODE over the period of interest 

    time = Array(solution.t);               # Storing the time sampling 
    solution = Array(solution)[2, 1:end];   # Storing the solution for troposphere 
    return bin(time, solution);             # Binning the results into years 
end

"""
Takes a list of solvers as an input and runs a multithreaded comparison that 
stores the time information and the binned output of the ODE solver.
"""
function profile_solvers(solvers::Tuple)
    local (TO, P) = read_hd5("Guttler14.hd5");  # Reading the data into the scope 
    
    #! not a big fan of this equilbrium calculation
    eq_prod = 3.747273140033743 * 1.88;  # Correct equilibrium production
    equilibrium = TO \ (-eq_prod * P);   # Brehm equilibriation for Guttler 2014

    ∇(y, p, t) = vec(TO * y + production(t) * P);           # Calculates the derivative
    burn_in = ODEProblem(∇, equilibrium, (-360.0, 760.0));  # Burn in problem  
    burn_in = solve(burn_in, reltol = 1e-6);                # Running the brun in

    test = time();
    for solver in solvers   # Looping over the solvers 
        timer = time();                                     # Starting a timer
        solution = run_solver(solver(), ∇, burn_in[end]);   # Running the solver 
        timer =- timer();                                   # Getting the run time
        write_hd5(solution, string(solver));                # Storing the results
    end 
    test =- time()
    println(test);
end

solvers = (Rosenbrock23, ROS34PW1a, KenCarp58); # A list of solvers
profile_solvers(solvers);                       # Calling the program

function compare_accuracy(file_name::String, solvers::Tuple)::Vector{Float64}
    hd5 = h5open(file_name);    # Opening the stored information 
    for solver in solvers;      # Looping through the solvers
        hd5[solver]
    end

    # If this were in R I would be working with a data frame of 3 cols"
    # solver::String, time::Float64, C14::Float64
    # I would be calculating the median in each bin from the 
    # solver sample and then calculating the deviation
    # Possibly need 
    # using csv
    # using DataFrames
    # using Statistics
end

#* to read back the results from the .hd5 file use a = ode["solver"][1:end]