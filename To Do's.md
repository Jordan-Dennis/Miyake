# To Do's:
 - I need to look into |> operator to simplify the global namespace 
instead of using local namespaces.
 - I want to have a plot that places things into quadrants based on 
 the speed and the accuracy 
 - I first need to look into how to measure the accuracy of the result
 using an assymptotic behaviour of the RK45 solver as the time step is
 reduced 
 - I need to work on the comparison of accuracy using the median 
 - Lower priority is fixing Github 
 - So I want to use GadFly as it is similar to ggplot

# Plan
 - function to multithread solvers in batches (yet to come)
 - write a function to compare the accuracy 
 - Let's get the basic plot going
 - Also need to call multiple times to oget a measure of the speeed
 - Thinking of refactoring so that all calls to user defined functions
 happen in main(). I just like this style a little more. (more thought)
 - That fucking equilibrium
  

# Goals
 - I think I should be able to do all this in _100_ lines of code



