# To Do's:
 - I need to look into |> operator to simplify the global namespace 
instead of using local namespaces.
 - I want to have a plot that places things into quadrants based on 
 the speed and the accuracy 
 - I first need to look into how to measure the accuracy of the result
 using an assymptotic behaviour of the RK45 solver as the time step is
 reduced 
 - I need to work on the comparison of accuracy using the median 
 - I want to web scrape the list of ODEs from DifferentialEquations.jl
 - Lower priority is fixing Github 

# Docstring plan
 - I want to reduce the size of my doc strings while maintaining the 
 amount of information that is included within them
 - Doc strings are written in markdown

# Plan
 - function to multithread solvers in batches (yet to come)
 - Chose a starting sample of 5 solvers and compare 
 - write a function to compare the accuracy 
 - I need to work out how I want to store the time information
 - consider using the DataFrames.jl package
 - I need to work out a way of selecting a minimum resolution of 31
 - Actually, I am just going to ignore the solvers that don't meet 
 this minum requirement 

# Goals
 - I think I should be able to do all this in _100_ lines of code



