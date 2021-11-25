using Plots; gr();  # Importing plots with the GR backend  
using Profile;      # Profiling library   

const DIMENSIONS = 2;   # The spatial dimensions present in the simulation
const DURATION = 50000;  # The number of discrete iterations to be performed.
const DT = 1e-3;        # The time in seconds ellapsed by a discrete time step.

pos = zeros(DURATION, DIMENSIONS); # A matrix holding the position data 
vel = zeros(DURATION, DIMENSIONS); # A vector for the instantaneous velocities 
acc = zeros(DURATION, DIMENSIONS); # A vector for the instantaneous accelerations

pos[1, 1:end] = [1.0, 0.0];   # Initial x and y position
vel[1, 1:end] = [0.0, 1.0];   # Initial x and y velocity
acc[1, 1:end] = - 1 / sum(pos[1, 1:end] .^ 2) ^ 1.5 * pos[1, 1:end];  # accelerations

for i in 1:DURATION - 1
    vel[i + 1, 1:end] = vel[i, 1:end] + 0.5 * DT * acc[i, 1:end];   # Updating the velocity 
    pos[i + 1, 1:end] = pos[i, 1:end] + DT * vel[i + 1, 1:end];     # Updating the positions
    acc[i + 1, 1:end] = - 1 / sum(pos[i + 1, 1:end] .^ 2) ^ 1.5 * pos[i + 1, 1:end];  # Updatig the accelerations
    vel[i + 1, 1:end] += 0.5 * DT * acc[i + 1, 1:end];  # Updating the velocity
    if i == 6283 # Calculated using 2π / DT (v = t = 1.0)
        vel[i + 1, 2] += sqrt(4 / 3) - 1;    # Hohmann speed 1
    elseif i == 29827 # Calculated numerically on the first run
        vel[i + 1, 2] -= 1 / sqrt(2) * (1 - sqrt(2 / 3)); # Hohmann impulse 2
    end
end

@gif for i in 1:100:DURATION    # Creating a gif of the simulation 
    scatter([pos[i, 1]], [pos[i, 2]], xlim = (-2.0, 2.0), ylim = (-2.0, 2.0));  # Plotting the orbit 
end

