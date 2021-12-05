using CSV;          # For reading the information 
using DataFrames;   # For convinient storage 
using Gadfly;       # for generating the plots

r = DataFrame(CSV.File("solver_profiles.csv"));  # Reading data into the workspace
r = r[2:end, 1:end];

datavisual = Gadfly.plot(# Plot object for manipultation 
    y=r.accuracy, x=r.time_mean, label=r.solver,    # Raw data 
    Guide.ylabel("Accuracy"), Guide.xlabel("Time"), # X, Y Labels  
    ymin=r.accuracy - r.accuracy_var,   # Y lower error bar
    ymax=r.accuracy + r.accuracy_var,   # Y upper error bar
    xmin=r.time_mean - r.time_var,  # X lower error bar
    xmax=r.time_mean + r.time_var,  # X upper error bar
    Geom.yerrorbar, Geom.xerrorbar, Geom.point, Geom.label)  # Geometries  

savefig(datavisual, "solver_profiles.pdf"); # Saving the figure