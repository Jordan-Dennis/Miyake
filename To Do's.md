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

So the next problem is that the $\Delta C14$ is measured as a percentage and as a result I need to add this to the calculations. It also means that I have been plotting the wrong thing this entire time. 

The relevant bit of ticktack code seems to be `d_14_c = (data[:, index] - box_steady_state) / box_steady_state * 1000`, where data comes from the function `run_bin`. This resutrns the actual C14 concentrations.Now I just need to work out how to get the `box_steady_state`. It is literally retrieved from the brehm equilibriation. That is easy dope but -> Possibly problematic.