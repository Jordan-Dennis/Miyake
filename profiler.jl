using HDF5;                     # for .hd5 file manipulation
using DifferentialEquations;    # Provides a variety of differential solvers
using LinearAlgebra: Diagonal;  # Efficient Diagonal matrixes
using Statistics;               # Let's get this fucking bread 
using DataFrames;               # For succinct data manipulation
using CSV;                      # For writing the data to prevent constant re-running
using ForwardDiff;              # For profiling the gradients
using DynamicPipe;              # For better code

"""
Takes time series data and calculates the average of each year.
"""
function bin(time_series::Vector{Float64}, solution_vector::Vector{Float64})::Vector{Float64}
    local binned_solution = Vector{Float64}(undef, 0);  # Setting a vector to hold the bins 
    local whole_times = @. floor(time_series);          # Creating a vector of discrete time.
    for whole_time in unique(whole_times)                                           # Looping over the unique elements discrete times 
        local indexes = findall(whole_times .== whole_time);                        # Getting the indexes of the entries 
        append!(binned_solution, mean(solution_vector[indexes]));  # Appending to binned_solution
    end
    return binned_solution
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
function run_solver(solver, ∇::Function, U0::Vector{Float64}, p::Vector{Any})::Vector{Float64}
    local problem = ODEProblem(∇, U0, (760.0, 790.0), p);      # Creating the ODEProblem instance
    local solution = solve(problem, reltol = 1e-6, solver); # Solving the ODE over the period of interest 
    local time = Array(solution.t);                         # Storing the time sampling 
    
    solution = Array(solution)[2, 1:end];   # Storing the solution for troposphere 
    solution = bin(time, solution);         # Getting the annual means
    return solution;                        # Binning the results into years 
end

"""
A function that calculates the gradient of the model at a specific time, `t::Float64`
with parameters, `p::Array` and position `x::Vector{Float64}`. p holds the 1: period 
of the production function, 2: The transfer operator of the model, and 3: the 
projection of the production function into the system.
"""
function derivative(x, p, t)
    """
    Calculates the production of C14 based on the projection based on the model 
    presented in the _Guttler 2014_ paper.
    """
    function production(t, T)                                       
        local gh::Float64 = 20 * 1.60193418235;  # height of the super-gaussian  
        local uf::Float64 = 3.747273140033743;   # unit correcting factor
        return uf * (1.88 + 0.18 * 1.88 * sin(2 * π / T * t + 1.25) +   # Sinusoidal production 
            gh * exp(- (12 * (t - 775)) ^ 16));                             # Super gaussian
    end
    return vec(p[2] * x + production(t, p[1]) * p[3]);    # Derivative with extra argument 
end 

"""
Takes a list of solvers as an input and runs a multithreaded comparison that 
stores the time information and the binned output of the ODE solver.
"""
function profile_solvers(solvers::Vector, ∇::Function, u0::Vector{Float64},
        p::Vector)::DataFrame
    local C14 = Matrix{Float64}(undef, length(solvers), 31);    # Creating the storage Matrix 
    local t_mean = Vector{Float64}(undef, length(solvers));     # For the mean of the times
    local t_var = Vector{Float64}(undef, length(solvers));      # For the time varience 
    local results = DataFrame(solver = @.string(solvers));      # DataFrame of summary Statistics

    for (index, solver) in enumerate(solvers)           # Looping over the solvers 
        local time_sample = Vector{Float64}(undef, 10); # A vector to hold the different run times of each trial 
        for i in 1:10
            local timer = time();                       # Starting a timer
            solution = run_solver(solver(), ∇, u0, p);  # Running the solver
            time_sample[i] = time() - timer;            # ending the timer 

            if i == 10                      # Storing final run
                C14[index, 1:end] = solution;   # filling C14
            end
        end
        #! If I break inside the loop this will produce an error 
        t_mean[index] = mean(time_sample[2:end]);   # Storing run time ignoring compile run.
        t_var[index] = var(time_sample[2:end]);     # Storing time error
    end 

    #? The first two columns are returning NaN
    C14 = (C14' .- median(C14, dims=1)')';          # Calculating deviations from median 
    results.accuracy = vec(mean(C14, dims=2));      # Calculating the mean of the deviation from the median 
    results.accuracy_var = vec(var(C14, dims=2));   # Calculating the RMSE error << is better 
    results.time_mean = t_mean;                     # Storing the mean run time
    results.time_var = t_var;                       # Storing the varience of the time sample 
    return results
end

"""
Calculates the gradient using a χ² loss function. 
"""
function profile_gradients(solver, ∇::Function, u0::Vector{Float64},
        parameters::Vector)
    local solution = run_solver(solver, ∇, u0, parameters); #! need consistent naming conventions for parameters
    local ΔC14 = solution[2:end] - solution[1:end - 1];     # Calculating the modelled DC14

    local miyake = DataFrame(CSV.File("Miyake12.csv"));                     # Reading the Miyake data
    local χ² = sum(((miyake.d14c .- ΔC14[1:28]) ./ miyake.sig_d14c) .^ 2);  # calculating a χ² statistic
    #! The Chi squared is huge
    return -0.5 * χ², ΔC14, solution

    #? For generating the test plot I have the following things 
    plot(layer(x=miyake.year, y=miyake.d14c, Geom.point),
        layer(x=miyake.year, y=ΔC14, Geom.line));

end

function main()
    local parameters = Vector(undef, 3);                        # Storing the model parameters #? |>  
    (parameters[2], parameters[3]) = read_hd5("Guttler14.hd5"); # Reading the data into the scope 
    parameters[1] = 11.0;                                       # Setting period for burn in
    #! Flag
    local uf = 3.747273140033743;                               # Correct equilibrium production
    local u0 = parameters[2] \ (- uf *  1.88 * parameters[3]);  # Brehm equilibriation for Guttler 2014

    local position = @>> ODEProblem(derivative, u0, #! Yay Pipes Kind of 
        (-360.0, 760.0), parameters) |>             # Burn in problem  
        solve(_, reltol=1e-6).u[end];               # Running the model and returning final position

    local solvers = [TRBDF2, BS3, Tsit5, Rosenbrock23,
        ROS34PW1a, QNDF1, ABDF2, ExplicitRK, DP5,
        TanYam7, Vern6];
    profiles = profile_solvers(solvers, derivative, position, parameters); # Running the first batch of solvers 
    
    if isfile("solver_profiles.csv");                           # Checking for the .csv file 
        CSV.write("solver_profiles.csv", profiles, append=true);# Adding new solvers to the CSV
    else 
        CSV.write("solver_profiles.csv", profiles, append=false);# Creating the file if it does not exist 
    end
end

# main();