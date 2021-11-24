# using PyCall;
# using Profile;
using Plots; gr();

using Random;   # For the random generation of position 

const POPULATION = 10;  # The number of particles 
const DIMENSIONS = 2;    # Number of dimensions 
const DURATION = 10000;  # Number of discrete time units 
const DT = 1e-3;         # Interval between discrete time units 
const ϵ = 119.8e-12; # σ Parameter for Argon 
const σ = 3.405e-10; # ϵ Parameter for argon

pos = 200 * rand(Float64, (DIMENSIONS, POPULATION, DURATION));    # Array to store positions
vel = 100 * rand(Float64, (DIMENSIONS, POPULATION, DURATION));    # Array to store velocities 
acc = zeros(Float64, (DIMENSIONS, POPULATION, DURATION));   # Array to store accelerations 

for i in 1:DURATION - 1
    vel[1:end, 1:end, i + 1] = vel[1:end, 1:end, i] + 0.5 * DT * acc[1:end, 1:end, i];  # Velocity update 
    pos[1:end, 1:end, i + 1] = pos[1:end, 1:end, i] + vel[1:end, 1:end, i + 1] * DT; # New positions

    for j in 1:POPULATION   # Looping through each particle 
        for k in 1:j - 1    # Looping through every particle 
            local r = (sum((pos[1:end, j, i] - pos[1:end, k, i]) .^ 2)) .^ 0.5; # Finding the radius
            acc[1:end, k, i + 1] += 48 * ϵ * (σ ^ 12 / r ^ 14 - σ ^ 6 / r ^ 8) *
                (pos[1:end, j, i] - pos[1:end, k, i]); # Lenard Jones Potential 
            acc[1:end, j, i + 1] -= 48 * ϵ * (σ ^ 12 / r ^ 14 - σ ^ 6 / r ^ 8) *
                (pos[1:end, j, i] - pos[1:end, k, i]); # Lenard Jones Potential via newton's third law
            # So this is all wrong I need to take tha partial derivatives of the potential
            # I also need to work out how to animations. 
            # I need to tune the lenard jones to work
        end
    end

    vel[1:end, 1:end, i + 1] = vel[1:end, 1:end, i + 1] + 0.5 * DT * acc[1:end, 1:end, i + 1]; # Secind velocity update 
    for j in 1:POPULATION # Looping through the particles to prevent from leaving the box
        for k in 1:DIMENSIONS # Looping through the dimensions
            if pos[k, j, i + 1] >= 200 || pos[k, j, i + 1] <= 0   # Checking if out of bounds
                vel[k, j, i + 1] = -vel[k, j, i + 1];   # Reversing the direction of travel
            end
        end
    end
end

@gif for i in 1:10:DURATION
    scatter(pos[1, 1:end, i], pos[2, 1:end, i], xlim = (0.0, 200.0), ylim = (0.0, 200.0));   # Animating the particles
end