import Plots; gr(); # Moving Plot( ) into the namespace
using  SparseArrays; # For sparse arrays used in the flux matrix

const DURATION = 1000;   # Number of iterations

P = zeros(11);  # Projection of the production function into the box
P[1] = 0.70;    # Stratosphere production projection 
P[2] = 0.30;    # Troposphere production projection 

N = zeros(11, DURATION); # Holds the current mass of ^{14}C
N[1] = 89.0;      # Stratosphere
N[2] = 501.0;     # Troposphere 
N[3] = 900.0;     # Surface Water 
N[4] = 3.0;       # Surface biota
N[5] = 37800.0;   # Deep water
N[6] = 110.0;     # Short-Lived Biota
N[7] = 450.0;     # Long-Lived Biota
N[8] = 300.0;     # Litter 
N[9] = 1350.0;    # Soil 
N[10] = 500.0;    # Peat 
N[11] = 0.0;      # Sedimentary sink?

# The direction of the arrow determines the sign of the flux element 
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
F = sparse(F);

function production(y)      # The production function
    local P = 1.88;         # Steady State Production in ^{14}Ccm^{2}s^{-1}
    local T = 11;           # Period of solar cycle years 
    local Φ = 1.25;         # Phase of the solar cycle 
    local P += 0.18 * P * sin(2 * π / T * y + Φ); # Evaluating the production fucntion at the time
    return P * 14.003242 / 6.022 * 5.11 * 31536. / 1.e5;    # Correction the units # FLAG
end

for t in 2:DURATION # Iterating module
    local y = 760 + t / 12;             # Present year
    local DT = 1 / 12;                  # Time step in years
    local λ = 1 / 5700 * log(1 / 2);    # Decay Constant
    # The ODE Solving happens here
    N[1:end, t] = 
        N[1:end, t - 1] + 
        # F * N[1:end, t - 1] .* DT + # Unstable 
        λ .* N[1:end, t - 1] .* DT + # Stable
        production(y) .* P .* DT; # Overpowers the decay 
end

plot(N[1, 1:1000]);     # Stratosphere 
plot!(N[2, 1:1000]);    # Troposphere
plot!(N[3, 1:1000]);    # Surface Water 
plot!(N[4, 1:1000]);    # Surface Biota 
plot!(N[5, 1:1000]);    # Deep Water
plot!(N[6, 1:1000]);    # Short-Lived Biota
plot!(N[7, 1:1000]);    # long-Lived Biota
plot!(N[8, 1:1000]);    # Litter
plot!(N[9, 1:1000]);    # Soil
plot!(N[10, 1:1000]);   # Peat
plot!(N[11, 1:1000]);   # Sedimentary sink

# Fixing the units draft
# m = 14.0032420 a.u. 1.66054e-27 kg a.u.^{-1} * 1e-12      # Mass of ^{14}C (Gt)
# sa = 4 * π * ((6371(Earth's radius) + 51(Stratosphere height)) * 1e5) ^ 2 *    # Surface arear in cm ^ 2
# production = m * sa

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