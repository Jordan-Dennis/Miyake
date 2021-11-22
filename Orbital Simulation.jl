using Plots; gr(); Plots.GRBackend();
using Profile;

global DIMENSIONS = 2;  # The spatial dimensions present in the simulation
global DURATION = 10000;    # The number of discrete iterations to be performed.
global DT = 1e-3;   # The time in seconds ellapsed by a discrete time step.

pos = Matrix{Float64}(undef, (DIMENSIONS, DURATION)); # A matrix holding the position data 
vel = Matrix{Float64}(undef, (DIMENSIONS, DURATION));    # A vector for the instantaneous velocities 
acc = Matrix{Float64}(undef, (DIMENSIONS, DURATION));    # A vector for the instantaneous accelerations

for i in 1:DURATION - 1
    vel[1:end, i + 1] = vel[1:end, i] + 0.5 * DT * acc[1:end, i];  # Updating the velocity 
    pos[1:end, i + 1] = pos[1:end, i] + DT * vel[1:end, i + 1];   # Updating the positions
    acc[1:end, i + 1] = 1 / sum(pos[1:end, i + 1] .^ 2) ^ 1.5 * pos[1:end, i + 1];  # Updatig the accelerations
    vel[1:end, i + 1] += 0.5 * DT * acc[1:end, i + 1];  # Updating the velocity
end

positions = scatter();  # Creating plot object. Is this nessecary or can scatter! be used in the for loop?
animation = Animation();    # Creating animation object 
for i in 1:DURATION
    scatter!(pos[1:end, i]);
    frame(animation);
    # Need to stop from plotting all ten thousand in the animation
end

# Is range() or 1:DURATION faster?
# Something interesting to try how local changed the time of assignment
# It looks like assigning a matrix is much faster than assigning a vector using []
# How does the precision of the assignment i.e. Float64, Float32 ec.t affect the assignment time
# Ones and presumably zeros seem to be similar to the matrix assignment
# How does the size of the array affect theassignment time.
# Vector and matrix creation time seem similar.
# How does this compare as the size increases.
# Array vs MAtrix vs Vector 
# Does the order of multiplication affet the time.