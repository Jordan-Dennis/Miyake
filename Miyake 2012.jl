#= Miyake 2012 
    - 1: Stratosphere
    - 2: Troposphere
    - 3: Biosphere
    - 4: Marine Surface  
=#
N = zeros(4);   # The reserviour values 
N[1] = 0.15;    # Proportion in Stratosphere
N[2] = 0.85;    # Proportion in Troposphere
N[3] = 2.52;    # Atmospheric Proportion in Biosphere
N[4] = 2.0;     # Atmospheric Proportion in Marine Surface

F = zeros(4, 4);    # The flux matrix 
F[2, 1] = 1 / 3;    # Stratosphere → Troposphere
F[1, 2] = 1 / 3;    # Troposphere → Stratosphere
F[3, 2] = 1 / 23;   # Troposphere → Biosphere
F[4, 2] = 1 / 11;   # Troposphere → Marine Surface
F[2, 3] = 1 / 23;   # Biosphere → Troposphere
F[2, 4] = 1 / 11;   # Marine Surface → Troposphere

const DURATION = 100; # Iterative steps 

