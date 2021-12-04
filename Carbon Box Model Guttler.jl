using Gadfly;                   # Moving Plot( ) into the namespace
using HDF5;                     # for .hd5 file manipulation
using DifferentialEquations;    # Provides a variety of differential solvers
using LinearAlgebra: Diagonal;  # Efficient Diagonal matrixes
using Statistics;               # Let's get this fucking bread 
using DataFrames;               # For succinct data manipulation

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
function profile_solvers(solvers::Vector)::Matrix{Union{Float64, String}}
    local (TO, P) = read_hd5("Guttler14.hd5");  # Reading the data into the scope 
    
    local eq_prod = 3.747273140033743 * 1.88;  # Correct equilibrium production
    local equilibrium = TO \ (-eq_prod * P);   # Brehm equilibriation for Guttler 2014

    ∇(y, p, t) = vec(TO * y + production(t) * P);               # Calculates the derivative
    local burn_in = ODEProblem(∇, equilibrium, (-360.0, 760.0));# Burn in problem  
    local burn_in = solve(burn_in, reltol = 1e-6);              # Running the brun in

    local C14 = Matrix{Float64}(undef, length(solvers), 34);      # Creating the storage Matrix 
    local time_mean = Vector{Float64}(undef, length(solvers));    # For the mean of the times
    local time_var = Vector{Float64}(undef, length(solvers));     # For the time varience 
    local results = DataFrame(solver = @.string(solvers));        # DataFrame of summary Statistics

    for (index, solver) in enumerate(solvers)           # Looping over the solvers 
        local time_sample = Vector{Float64}(undef, 10); # A vector to hold the different run times of each trial 
        for i in 1:10
            local timer = time();                               # Starting a timer
            solution = run_solver(solver(), ∇, burn_in[end]);   # Running the solver
            time_sample[i] = -timer + time();                   # ending the timer 
        end
        time_mean[index] = mean(time_sample);  # Storing run time 
        time_var[index] = var(time_sample);     # Storing time error
    end 

    C14 = (C14' .- median(C14, dims=1)')';      # Calculating deviations from median 
    results.accuracy = mean(C14, dims=2);       # Calculating the mean of the deviation from the median 
    results.accuracy_var = var(C14, dims=2);    # Calculating the RMSE error << is better 
    results.time_mean = time_mean;              # Storing the mean run time
    results.time_var = time_var;                # Storing the varience of the time sample 
    return results
end

# function main()
    solvers = [Rosenbrock23, ROS34PW1a, QNDF1, ABDF2, ExplicitRK,
        DP5, TanYam7, Vern6, SSPRK43, VCAB5];   # A list of solvers
    r = profile_solvers(solvers);         # Calling the program

    datavisual = Gadfly.plot(
        y=r.accuracy, x=r.time_mean, label=r.solvers,  
        Guide.ylabel("Accuracy"), Guide.xlabel("Time"),          
        ymin=r.accuracy - r.accuracy_var, ymax=r.accuracy + r.accuracy_var,
        xmin=r.time_mean - r.time_var, xmax=r.time_mean + r.time_var, 
        Geom.yerrorbar, Geom.xerrorbar, Geom.point, Geom.label)   
# end