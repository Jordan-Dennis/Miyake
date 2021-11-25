#= Miyake 2012 
    - 1: Stratosphere
    - 2: Troposphere
    - 3: Biosphere
    - 4: Marine Surface  
=#
N = zeros(4);       # The reserviour values 
N[1] = 0.15;    # Proportion in Stratosphere
N[2] = 0.85;    # Proportion in Troposphere
N[3] = 0.

F = zeros(4, 4);    # The flux matrix 
F[2, 1] = 1 / 3;    # Stratosphere → Troposphere
F[1, 3] = 1 / 23;   # Troposphere → Biosphere