using HDF5;                     # for .hd5 file manipulation
using DifferentialEquations;    # Provides a variety of differential solvers
using LinearAlgebra: Diagonal;  # Efficient Diagonal matrixes
using Statistics;               # Let's get this fucking bread 
using DataFrames;               # For succinct data manipulation
using CSV;                      # For writing the data to prevent constant re-running
using ForwardDiff;              # For profiling the gradients

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
    local problem = ODEProblem(∇, U0, (760.0, 790.0));      # Creating the ODEProblem instance
    local solution = solve(problem, reltol = 1e-6, solver); # Solving the ODE over the period of interest 
    local time = Array(solution.t);                         # Storing the time sampling 
    
    solution = Array(solution)[2, 1:end];   # Storing the solution for troposphere 
    solution = bin(time, solution);         # Getting the annual means
    return solution;                        # Binning the results into years 
end

"""
"""


"""
Runs the model for 1000 years prior to the sampled data and returns the final
position of the system
"""


"""
Takes a list of solvers as an input and runs a multithreaded comparison that 
stores the time information and the binned output of the ODE solver.
"""
function profile_solvers(solvers::Vector)::DataFrame
    local (TO, P) = read_hd5("Guttler14.hd5");  # Reading the data into the scope 
    
    local uf = 3.747273140033743;                   # Correct equilibrium production
    local equilibrium = TO \ (- uf *  1.88 * P);    # Brehm equilibriation for Guttler 2014

    ∇(y, p, t) = vec(TO * y + production(t) * P);               # Calculates the derivative
    local burn_in = ODEProblem(∇, equilibrium, (-360.0, 760.0));# Burn in problem  
    local burn_in = solve(burn_in, reltol = 1e-6);              # Running the brun in

    local C14 = Matrix{Float64}(undef, length(solvers), 31);    # Creating the storage Matrix 
    local t_mean = Vector{Float64}(undef, length(solvers));     # For the mean of the times
    local t_var = Vector{Float64}(undef, length(solvers));      # For the time varience 
    local results = DataFrame(solver = @.string(solvers));      # DataFrame of summary Statistics

    for (index, solver) in enumerate(solvers)           # Looping over the solvers 
        local time_sample = Vector{Float64}(undef, 10); # A vector to hold the different run times of each trial 
        for i in 1:10
            local timer = time();                               # Starting a timer
            solution = run_solver(solver(), ∇, burn_in[end]);   # Running the solver
            time_sample[i] = time() - timer;                    # ending the timer 
            
            #! not a big fan of this
            if length(solution) !== 31                                  # Checking that the solver had the appropriate resolution 
                local dims = size(C14);                                 # soft dimensions to save multiple function calls 
                local nC14 = Matrix{Float64}(undef, dims[1], dims[2]);  # Creating the new array to house the values
                nC14[1:index - 1, 1:31] = C14[1:index - 1, 1:31];       # Filling with the old values 
                C14 = nC14;                                             # Assigning over the old matrix 
                break
            elseif i == 10                      # Storing final run
                C14[index, 1:end] = solution;   # filling C14
            end
        end
        t_mean[index] = mean(time_sample[2:end]);   # Storing run time ignoring compile run.
        t_var[index] = var(time_sample[2:end]);     # Storing time error
    end 

    C14 = (C14' .- median(C14, dims=1)')';          # Calculating deviations from median 
    results.accuracy = vec(mean(C14, dims=2));      # Calculating the mean of the deviation from the median 
    results.accuracy_var = vec(var(C14, dims=2));   # Calculating the RMSE error << is better 
    results.time_mean = t_mean;                     # Storing the mean run time
    results.time_var = t_var;                       # Storing the varience of the time sample 
    return results
end

function main()
    local batch_1 = [Rosenbrock23, ROS34PW1a, QNDF1, ABDF2, 
        ExplicitRK, DP5, TanYam7, Vern6, SSPRK43, VCAB5];   # First batch of solvers 
    local batch_2 = [KenCarp4, TRBDF2, Trapezoid, BS3, Tsit5,
        RadauIIA5, SRIW1, Rodas5, AutoVern7, Kvaerno5]      # Second batch of solvers    
    
    test = time();
    @async r = profile_solvers(batch_1); # Running the first batch of solvers 
    b = profile_solvers(batch_2); # Running the second batch in parallel
    println(time() - test)

    local profiles = vcat(r, b)
    
    if isfile("solver_profiles.csv");                           # Checking for the .csv file 
        CSV.write("solver_profiles.csv", profiles, append=true);# Adding new solvers to the CSV
    else 
        CSV.write("solver_profiles.csv", profiles, append=false);# Creating the file if it does not exist 
    end
end

main();