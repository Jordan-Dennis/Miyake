using Plots; gr();              # Moving Plot( ) into the namespace
using SparseArrays;             # For sparse arrays used in the flux matrix
using HDF5;                     # for .hd5 file manipulation
using DifferentialEquations;    # Provides a variety of differential solvers
using Optim: optimize, LBFGS;   # Importing the optimisation library and solver #? :foward
using LinearAlgebra: Diagonal;  # For efficient diagonal computational
#! I do not like the amount of imports that are going on 
# using DistributedArrays;  #? Probs not since I am using SparseArrays

const DURATION = 1000;  # Number of iterations
const DT = 1 / 12;      # Time step in years

# So I need to look into @threads and @task at the same time 
Guttler2014 = h5open("Guttler2014.hd5");                # Opening the HDF5 file
F = Guttler2014["fluxes"][1:end, 1:end];                # Retrieving the flux matrix 
P = Guttler2014["production coefficients"][1:end];      # Retrieving the projection of the production 
N = Guttler2014["reservoir content"][1:end, 1:end];     # The C14 reserviour contents 
λ = Guttler2014["decay coefficients"][1:end, 1:end];    # The decay constants as the diagonal elements
close(Guttler2014);                                     # Closing the file 
#* @async is not a speed boost here.
#* @spawn is much worse.
#? Huge opertunity to use the |> pipe operator (similar to %>% from R)

# F = sparse(F);  # Removing zero elements for computational efficiency
λ = Diagonal(λ);  #* @async and @spawn (worse) not a speed boost 
#! Diagonal is faster than sparse 

function production(y)      # The production function
    local P = 1.88;         # Steady State Production in ^{14}Ccm^{2}s^{-1}
    local T = 11;           # Period of solar cycle years 
    local Φ = 1.25;         # Phase of the solar cycle 
    local P += 0.18 * P * sin(2 * π / T * y + Φ); # Evaluating the production fucntion at the time
    return P * 14.003242 / 6.022 * 5.11 * 31536. / 1.e5;    # Correction the units
    #* This is from the convert production rate function 
end

precompile(production, (Int, ));    # precompiling for speed

#// Mock compile 
C14F = F ./ transpose(N);   # Proportion flow of C14 (axis=1 is RowSum)
NC14F = Vector{Float64}(undef, 11); # Empty vector with the goal of applying |> latter 
C14Content = sum(C14F, dims=2); # The C14 content in each reserviour 
for i in 1:11; NC14F[i] = C14Content[i]; end;    # Filling NC14F with the C14Content elements

#! I need to make a flow chart and work out how I can use pipe to achieve this without the random declarations 

NC14F = Diagonal(NC14F);  # C14 content of each reserviour
transfer_operator = transpose(C14F) - NC14F - λ;    #* I want to use the |> here

#// Mock equilibriate Brehm 
steady_state = transfer_operator \ (- 1.88 * P); #! The 1.88 is the steady state quoted in the paper 
#? ticktack has production rate (as an argument to the function)

#// Mock objective function
troposphere_residual(TC14::Float64) = (steady_state[2] - TC14) ^ 2; # TC14 is the target C14 and steady_state[2] is the equilibrium tropospher position  
optimized_residuals = optimize(troposphere_residual, [6.0], LBFGS());    #* optimisation from the Guttler2014
#! I need to ask why Urakash is using 6. as the starting place.
#? Check the speed of implicit function definitions. 
#? Can I move the function declaration into the optimisation function 
#* I need beter variable names in this section 

for t in 2:DURATION # Iterating module
    local y = 760 + t / 12;             # Present year
    # The ODE Solving happens here


    # N[1:end, t] = N[1:end, t - 1] +     # Previous position 
    #     F * N[1:end, t - 1] .* DT +     # Unstable 
    #     λ .* N[1:end, t - 1] .* DT +    # Stable
    #     production(y) .* P .* DT;       # Overpowers the decay 
end

# Unit PPT
ΔN = N[1:end, 2:end] - N[1:end, 1:end - 1]; # Calculating the changes between time steps 
RN = ΔN ./ N[1:end, 1:end - 1];             # Calculating the proportion change in each reserviour

carbon = plot(RN[1, 1:end]); # Stratosphere 
# for i in 2:11
#     plot!(RN[i, 1:end]); # Remaining Boxes
# end
display(carbon)

# To Do's
#   I need to look into the @async macro 
#   I need to look into @task macro 
#   @spawnat let's me chose the worker task to asign 
#   The @async is closely realted to @task (equivalent to creating then running)