# To Do's:
 - I need to look into |> operator to simplify the global namespace 
instead of using local namespaces.
 - I want to have a plot that places things into quadrants based on 
 the speed and the accuracy 
 - I need to work on the comparison of accuracy using the median 
   - medians are calculated for each time step. Now I want to use the
   MAE error although I also have the right information to use the 
   RMSE error. I think I will go with the varience for ease of use
 - Lower priority is fixing Github 
 - So I want to use GadFly as it is similar to ggplot
 - Fix the local declarations 

# Plan
 - function to multithread solvers in batches (yet to come)
 - Let's get the basic plot going
 - Also need to call multiple times to oget a measure of the speeed
 - Thinking of refactoring so that all calls to user defined functions
 happen in main(). I just like this style a little more. (more thought)
  

# Goals
 - I think I should be able to do all this in _100_ lines of code



