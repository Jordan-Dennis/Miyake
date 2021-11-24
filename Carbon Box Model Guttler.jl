#= 
So the steady state production function is:
P_{mod} = P_{s, t} + 0.18P_{s, t}\sin\left(\frac{2Π}{T}y + Φ)
    T = 11yrs: The length of the solar cycle
    y = indep: The current year 
    Φ = 1.25: The phase of the solar cycle
    P_{s, t} = 1.88e14Ccm^{-2}s^{-1}: steady state production

The ODE that I need to solve is given by:
\frac{dN_{i}}}{dt} + \sum_{j = 1}^{11} F_{i → j} - \sum_{j = 1}^{11}F_{j → i} + \lambda N_{i} - P_{s, t} = 0

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
i = j is one and where there is no connection F_{ij} = 0.
=#

P = zeros(11);  # Projection of the production function into the box
P[1] = 0.70;    # Stratosphere production projection 
P[2] = 0.30;    # Troposphere production projection 

N = Vector{Float64}(undef, 11); # Holds the current mass of ^{14}C
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
F = zeros(11, 11); # The flux matrix of the gods 
for i in 1:11; F[i, i] = 1.0; end;  # The flux from one reserviour to itself

# Stratosphere Outgoing [Column 1, (1 → i)]
F[2, 1] = 45.0 / 89.0; # Stratosphere to Troposphere

# Troposphere Outgoing [Column 2 (2 → i)]
F[1, 2] = 45.0 / 501.0;     # The flux from Troposphere to Stratosphere
F[3, 2] = 60.5 / 501.0;     # The flu from Troposphere to Surface Water
F[6, 2] = 115.0 / 501.0;    # Flux from Troposphere to Short-Lived Biota

# Surface Water Outgoing [Column 3, (3 → i)]
F[2, 3] = 61.0 / 900.0; # The flux from Surface Water to Troposphere
F[4, 3] = 40.0 / 900.0; # Flux from Surface Water to Surface Biota
F[5, 3] = 38.2 / 900.0; # Flux from Surface Water to Deep Water
F[11, 3] = 0.3;         # Flux from Surface Water to Sedimentary Sink

# Surface Biota Outgoing [Column 4, (4 → i)]
F[3, 4] = 36.0 / 3.0;   # Flux from Surface Biota to Surface Water # FLAG
F[5, 4] = 4.0 / 3.0;    # Flux from  Surface Biota to Deep Water

# Deep Water Outgoing [Column 5, (5 → i)]
F[3, 5] = 42.0 / 37800.0;   # Flux from Deep Water to Surface Water
F[11, 5] = 0.2;             # Flux from Deep Water to Sedimentary Sink

# Short Lived Biota [Column 6, (6 → i)]
F[2, 6] = 60.0 / 110.0; # Flux from Short-Lived Biota to Troposphere
F[7, 6] = 15.0 / 110.0; # Flux from Short-Lived Biota to Long-Lived Biota
F[8, 6] = 40.0 / 110.0; # Flux from Short-Lived Biota to Litter

# Long-Lived Biota [Column 7, (7 → i)]
F[8, 7] = 15.0 / 450.0; # Flux from Long-Lived Biota to Litter

# Litter [Column 8, (8 → i)]
F[2, 8] = 50.0 / 300.0; # Flux from Litter to Troposphere
F[3, 8] = 1.00 / 300.0; # Flux from Litter to Surface Water
F[9, 8] = 3.00 / 300.0; # Flux from Litter to Soil
F[10, 8] = 1.0 / 300.0; # Flux from Litter to Peat

# Soil [Column 9, (9 → i)]
F[2, 9] = 3.0 / 1350.0; # Flux from Soil to Troposphere

# Peat [Column 10, (10 → i)]
F[2, 10] = 0.8 / 500.0; # Flux from Peat to Troposphere
F[11, 10] = 0.2 / 500.0; # Flux from Peat to Sedimentary Sink

# Sedimentary Sink [Column 11, (11 → i)]
F[2, 11] = 0.7; # Flux from Sedimentary Sink to Troposphere

function P(y)
    local P = 1.88e14;  # Steady State Production in ^{14}Ccm^{2}s^{-1}
    local T = 11; # Period of solar cycle years 

    return P + 0.18 * P * sin(2 * Π / T * y + Φ)
end