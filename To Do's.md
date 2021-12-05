# To Do's:
 - I need to look into |> operator for simplicity -- SERIOUSLY 

# Plan
 - Fix the local declarations 
 - Generalisation to any of the prebuilt models is important

# Dumping thoughts
 - I also have to profile the gradients and the solvers separately
  which means running exactly twice as much as I would like 
 - Things are getting very long and I am not a big fan
 - static typing only once everything is working 

# Goals
 - I want to have the profiling script in under _200_ lines 
      - This is including the generalisations and the derivatives 
 - I will have the analysis script in under _50_ lines

Pipes is something that I will implement at the end.

I need to look into the loss function for the profile gradients function.