# To Do's:
 - I need to look into |> operator to simplify the global namespace 
instead of using local namespaces.
 - I want to have a plot that places things into quadrants based on 
 the speed and the accuracy 
 - I first need to look into how to measure the accuracy of the result
 using an assymptotic behaviour of the RK45 solver as the time step is
 reduced 
 - I need to work on the comparison of accuracy using the median 
 - First I need to get the bin function working
 - I want to web scrape the list of ODEs from DifferentialEquations.jl
 - I want to export the data to a .csv for ease of manipulation.
 - I need to look into using CSV and Requests and Statisitics
 - Lower priority is fixing Github 
 - So I need a function to read the .hd5 file and return the transfer 
 operator and P but i think they are the only thinds that need to be 
 returned
 - I also need a function to write the data to a file 

# Docstring plan
 - I want to reduce the size of my doc strings while maintaining the 
 amount of information that is included within them
 - Doc strings are written in markdown
 - remove the parameter bit and returns and include this all within a 
 paragraph. Type hinting already tells the type

# Plan
 - function for the basic reading of the file
 - function for solving the ODE based on a solver request 
 - function to multithread solvers in batches 
 - I still want a main() function to implement all of this jazz 

# Goals
 - I think I should be able to do all this in _100_ lines of code



