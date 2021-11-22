using PyCall;
using DifferentialEquations;
using Profile;
using Plots; gr();
using Random;

# ticktack = pyimport("ticktack");

# @time and @allocated are for memory and time.
# Functions can also be defined anonymously
# All right I really want some hands on experience.

# I might just create a quick particle simulation and a quick orbit simulation so I know
# what is going on.

global const POPULATION = 10;  # The number of particles 
global const DIMENSIONS = 2;    # Number of dimensions 
global const DURATION = 1000;  # Number of discrete time units 
global const DT = 1e-3;         # Interval between discrete time units 

pos = rand(Float64, (DIMENSIONS, POPULATION, DURATION));    # Array to store positions
vel = 5 * rand(Float64, (DIMENSIONS, POPULATION, DURATION));    # Array to store velocities 
acc = zeros(Float64, (DIMENSIONS, POPULATION, DURATION));   # Array to store accelerations 

for i in 1:DURATION - 1
    vel[1:end, 1:end, i + 1] = vel[1:end, 1:end, i] + 0.5 * DT * acc[1:end, 1:end, i];  # Velocity update 
    pos[1:end, 1:end, i + 1] = pos[1:end, 1:end, i] + vel[1:end, 1:end, i + 1] * DT; # New positions

    # for j in 1:POPULATION   # Looping through each particle 
    #     for k in 1:j - 1    # Looping through every particle 
    #         local r = (sum((pos[1:end, j, i] - pos[1:end, k, i]) .^ 2)) .^ 0.5; # Finding the radius
    #         acc[1:end, k, i + 1] += 1e-6 * (- (12 / r) ^ 14 + (6 / r) ^ 8) * (pos[1:end, j, i] - pos[1:end, k, i]); # Lenard Jones Potential 
    #         acc[1:end, j, i + 1] -= 1e-6 * (- (12 / r) ^ 14 + (6 / r) ^ 8) * (pos[1:end, j, i] - pos[1:end, k, i]); # Lenard Jones Potential
    #         # So this is all wrong I need to take tha partial derivatives of the potential
    #         # I also need to work out how to animations. 
    #         # I need to tune the lenard jones to work
    #     end
    # end

    vel[1:end, 1:end, i + 1] = vel[1:end, 1:end, i + 1] + 0.5 * DT * acc[1:end, 1:end, i + 1]; # Secind velocity update 
    for j in 1:POPULATION # Looping through the particles to prevent from leaving the box
        for k in 1:DIMENSIONS # Looping through the dimensions
            if pos[k, j, i + 1] >= 1 || pos[k, j, i + 1] <= 0   # Checking if out of bounds
                vel[k, j, i + 1] = -vel[k, j, i + 1];   # Reversing the direction of travel
            end
        end
    end
end

# @gif 
animation = Animation(); # Creating an Animation object
p = scatter(pos[1, 1, 1:end], pos[2, 1, 1:end]);
for i in 2:POPULATION
    scatter!(pos[1, i, 1:end], pos[2, i, 1:end]);
    frame(animation);
end
# plot!(x2, y2);
display(p);


# Findings thus far:
# += is faster than a = a + 
# + is a lot faster than .+
# *= is a lot slower than a = a *
# * is a lot faster than .*
# ^ does not work for matrixes 
# .^= and a = a .^ are similar

# Questions to answer:
# How to store the time values for analysi
    # Possibly Profile.retrieve
    # Alternately it could be changing the profile(io argument )