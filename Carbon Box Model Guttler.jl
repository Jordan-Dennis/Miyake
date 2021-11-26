using Plots; gr();  # Moving Plot( ) into the namespace
using SparseArrays; # For sparse arrays used in the flux matrix
using HDF5;         # for .hd5 file manipulation

# threads is now set to 4. Look for opportunities to use this

const DURATION = 1000;   # Number of iterations

λ = zeros(11);  # The decay vector (dpm / g C)
λ[1] = 15.1;    # Stratosphere value 
λ[2] = 14.0;    # Troposphere values
λ[3] = 13.1;    # Surface Ocean
λ[4] = 13.1;    # Surface Biota
λ[5] = 11.8;    # Deep Water
λ[6] = 14.0;    # Short-Lived Biota
λ[7] = 13.9;    # Long-Lived Biota
λ[8] = 13.9;    # Litter
λ[9] = 13.2;    # Soil
λ[10] = 13.1;   # Peat
λ[11] = 0.0;    # Sediments

P = zeros(11);  # Projection of the production function into the box
P[1] = 0.70;    # Stratosphere production projection 
P[2] = 0.30;    # Troposphere production projection 

N = zeros(11, DURATION); # Holds the current mass of ^{14}C
N[1] = 135.0;   # Stratosphere
N[2] = 707.0;   # Troposphere 
N[3] = 1187.0;  # Surface Water 
N[4] = 4.0;     # Surface biota
N[5] = 44995.0; # Deep water
N[6] = 155.0;   # Short-Lived Biota
N[7] = 632.0;   # Long-Lived Biota
N[8] = 422.0;   # Litter 
N[9] = 1802.0;  # Soil 
N[10] = 664.0;  # Peat 
N[11] = 7320.0; # Sedimentary sink?

# The direction of the arrow determines the sign of the flux element 
# The new plan is to simply add the values out and in as percentages of the total 
# I also need to check what Urakash is doing.
F = zeros(11, 11);          # Flux matrix
F[2, 1] = 45.0 / 89.0;      # Stratosphere to Troposphere
F[1, 2] = 45.0 / 501.0;     # Troposphere to Stratosphere
F[3, 2] = 60.5 / 501.0;     # Troposphere to Surface Water
F[6, 2] = 115.0 / 501.0;    # Troposphere to Short-Lived Biota
F[2, 3] = 61.0 / 900.0;     # The flux from Surface Water to Troposphere
F[4, 3] = 40.0 / 900.0;     # Surface Water to Surface Biota
F[5, 3] = 38.2 / 900.0;     # Surface Water to Deep Water
F[11, 3] = 0.3;             # Surface Water to Sedimentary Sink
F[3, 4] = 36.0 / 3.0;       # Surface Biota to Surface Water # FLAG
F[5, 4] = 4.0 / 3.0;        # Surface Biota to Deep Water
F[3, 5] = 42.0 / 37800.0;   # Deep Water to Surface Water
F[11, 5] = 0.2;             # Deep Water to Sedimentary Sink
F[2, 6] = 60.0 / 110.0;     # Short-Lived Biota to Troposphere
F[7, 6] = 15.0 / 110.0;     # Short-Lived Biota to Long-Lived Biota
F[8, 6] = 40.0 / 110.0;     # Short-Lived Biota to Litter
F[8, 7] = 15.0 / 450.0;     # Long-Lived Biota to Litter
F[2, 8] = 50.0 / 300.0;     # Litter to Troposphere
F[3, 8] = 1.00 / 300.0;     # Litter to Surface Water
F[9, 8] = 3.00 / 300.0;     # Litter to Soil
F[10, 8] = 1.0 / 300.0;     # Litter to Peat
F[2, 9] = 3.0 / 1350.0;     # Soil to Troposphere
F[2, 10] = 0.8 / 500.0;     # Peat to Troposphere
F[11, 10] = 0.2 / 500.0;    # Peat to Sedimentary Sink
F[2, 11] = 0.7;             # Sedimentary Sink to Troposphere
F = - F + transpose(F);     # Accounting for outgoing and ingoing #FLAG
F = sparse(F);              # Making F Sparse for computation
F = F / 12;                 # Adjusting the flux so that it is monthly 

# C12 is in equilibrium 

function production(y)      # The production function
    local P = 1.88;         # Steady State Production in ^{14}Ccm^{2}s^{-1}
    local T = 11;           # Period of solar cycle years 
    local Φ = 1.25;         # Phase of the solar cycle 
    local P += 0.18 * P * sin(2 * π / T * y + Φ); # Evaluating the production fucntion at the time
    return P * 14.003242 / 6.022 * 5.11 * 31536. / 1.e5;    # Correction the units
end

for t in 2:DURATION # Iterating module
    local y = 760 + t / 12;             # Present year
    local DT = 1 / 12;                  # Time step in years
    # The ODE Solving happens here
    # Do I need to catch the negative terms
    N[1:end, t] = N[1:end, t - 1] + # Previous position 
        F * N[1:end, t - 1] .* DT + # Unstable 
        λ .* N[1:end, t - 1] .* DT + # Stable
        production(y) .* P .* DT; # Overpowers the decay 

    # So I am thinking that I can calculate the change and then use this quantity 
end

# Plot the changes from one to another time step. 
# Units are in parts per thousand 
# Divide the change by the amount n the reserviour

ΔN = N[1:end, 2:end] - N[1:end, 1:end - 1];
RN = ΔN ./ N[1:end, 1:end - 1];

# Solve the equilibrium position
#   → set target values and use matrix inverse to find the other coefficients 
#   → Check out the equilibriate function 

carbon = plot(RN[1, 1:end]); # Stratosphere 
# for i in 2:11
#     plot!(RN[i, 1:end]); # Remaining Boxes
# end
display(carbon)

#= Notes:
So the steady state production function is:
P_{mod} = P_{s, t} + 0.18P_{s, t}\sin\left(\frac{2Π}{T}y + Φ)
    T = 11yrs: The length of the solar cycle
    y = indep: The current year 
    Φ = 1.25: The phase of the solar cycle
    P_{s, t} = 1.88e14Ccm^{-2}s^{-1}: steady state production

Construction of matrix attempt one.
The order of my dimensions is:
    1 = Stratosphere
    2 = Troposphere 
    3 = Surface Water 
    4 = Surface Biota
    5 = Deep Water 
    6 = Short-lived Biota 
    7 = Long-lived biota
    8 = Litter 
    9 = Soil 
    10 = Peat
    11 = Sedimentary Sink
F_{ij} represents the flux from i to j. 
i < j is positive and i > j is negative.
where there is no connection F_{ij} = 0.
=#

# HD5 Experiment
using HD5;
Guttler2014 = h5open("Guttler14.hd5");
F = Guttler2014["fluxes"][1:end, 1:end];
P = Guttler2014["production coefficients"][1:end];
N = Guttler2014["reservoir content"][1:end, 1:end];

# To Do's:
# I need to make a csv file so that everything is simplified.
# So HD5 files are supported by julia 
