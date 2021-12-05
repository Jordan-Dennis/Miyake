using HDF5;                     # for .hd5 file manipulation
using DifferentialEquations;    # Provides a variety of differential solvers
using LinearAlgebra: Diagonal;  # Efficient Diagonal matrixes
using Statistics;               # Let's get this fucking bread 
using DataFrames;               # For succinct data manipulation
using CSV;                      # For writing the data to prevent constant re-running
using ForwardDiff;              # For profiling the gradients
using DynamicPipe;              # For better code #! FLAG

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
function run_solver(solver, ∇::Function, U0::Vector{Float64},
        p::Vector{Float64})::Vector{Float64}

    local problem = ODEProblem(∇, U0, (760.0, 790.0), p);      # Creating the ODEProblem instance
    local solution = solve(problem, reltol = 1e-6, solver); # Solving the ODE over the period of interest 
    local time = Array(solution.t);                         # Storing the time sampling 
    
    solution = Array(solution)[2, 1:end];   # Storing the solution for troposphere 
    solution = bin(time, solution);         # Getting the annual means
    return solution;                        # Binning the results into years 
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
        parameters::Vector{Float64}, steady_state::Vector{Float64})  
    
    local solution = run_solver(solver, ∇, u0, parameters);         #! need consistent naming conventions for parameters
    local ΔC14 = (solution .- steady_state[2]) ./ steady_state[2] .* 1000; # Calculating the modelled DC14

    local miyake = DataFrame(CSV.File("Miyake12.csv"));                     # Reading the Miyake data
    local χ² = sum(((miyake.d14c .- ΔC14[1:28]) ./ miyake.sig_d14c) .^ 2);  # calculating a χ² statistic
    return -0.5 * χ²

    #? For generating the test plot I have the following things 
    # plot(layer(x=miyake.year, y=miyake.d14c, Geom.point),
    #     layer(x=miyake.year, y=ΔC14[1:28], Geom.line));
    #! I need to resume from the testing here

end

# function main()
    TO, P = read_hd5("Guttler14.hd5");      # Reading the data into the scope 

    params = Vector{Float64}(undef, 6);   # Storing the model params 
    params[1] = 7.044873503263437;              # The mean position of the sinusoid 
    params[2] = 0.18;                           # The modulation of the sinusoid w. r. t the mean
    params[3] = 11.0;                           # Setting period of the sinusoid 
    params[4] = 1.25;                           # The phase shift of the sinusoid
    params[5] = 120.05769867244142;             # The height of the super gaussian 
    params[6] = 12.0;                           # Width of the super-gaussian 

    # The parameters to the model must be incorrect
    production(t, params) = params[1] * (1 + params[2] * 
        sin(2 * π / params[3] * t + params[4])) +               # Sinusoidal production 
        params[5] * exp(- (params[6] * (t - 775)) ^ 16);        # Super Gaussian event
    derivative(x, params, t) = vec(TO * x - production(t, params) * P);  # The derivative of the system 

    u0 = TO \ (- params[1] * P);  # Brehm equilibriation for Guttler 2014

    burnproblem = ODEProblem(derivative, u0, (-360.0, 760.0), params); # Burn in problem  
    burnsolution = solve(burnproblem, reltol=1e-6).u[end];             # Running the model and returning final position

    # local solvers = [TRBDF2, BS3, Tsit5, Rosenbrock23,
    #     ROS34PW1a, QNDF1, ABDF2, ExplicitRK, DP5,
    #     TanYam7, Vern6];
    # profiles = profile_solvers(solvers, derivative, position, params); # Running the first batch of solvers 
    
    # if isfile("solver_profiles.csv");                           # Checking for the .csv file 
    #     CSV.write("solver_profiles.csv", profiles, append=true);# Adding new solvers to the CSV
    # else 
    #     CSV.write("solver_profiles.csv", profiles, append=false);# Creating the file if it does not exist 
    # end

    #= Foward and Reverse Autodiff =#
    profile_gradients(BS3(), derivative, position, params, u0); #? Confusing position reversal
# end

# main();