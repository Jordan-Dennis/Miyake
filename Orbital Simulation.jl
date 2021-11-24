using Plots; gr(); Plots.GRBackend();
using Profile;

global const DIMENSIONS = 2;  # The spatial dimensions present in the simulation
global const DURATION = 10000;    # The number of discrete iterations to be performed.
global const DT = 1e-3;   # The time in seconds ellapsed by a discrete time step.

pos = Matrix{Float64}(undef, (DIMENSIONS, DURATION)); # A matrix holding the position data 
vel = Matrix{Float64}(undef, (DIMENSIONS, DURATION));    # A vector for the instantaneous velocities 
acc = Matrix{Float64}(undef, (DIMENSIONS, DURATION));    # A vector for the instantaneous accelerations

for i in 1:DURATION - 1
    vel[1:end, i + 1] = vel[1:end, i] + 0.5 * DT * acc[1:end, i];  # Updating the velocity 
    pos[1:end, i + 1] = pos[1:end, i] + DT * vel[1:end, i + 1];   # Updating the positions
    acc[1:end, i + 1] = 1 / sum(pos[1:end, i + 1] .^ 2) ^ 1.5 * pos[1:end, i + 1];  # Updatig the accelerations
    vel[1:end, i + 1] += 0.5 * DT * acc[1:end, i + 1];  # Updating the velocity
end

@gif for i in 1:DURATION
    scatter!(pos[1:end, i]);
end

