# To Do's:
 - I need to look into |> operator for simplicity -- SERIOUSLY 
 - Error catching is a priority 

# Plan
 - Fix the local declarations 
 - Generalisation to any of the prebuilt models is important

# Dumping thoughts
 - I want to refactor the code and have it more general
 - I also have to profile the gradients and the solvers separately
  which means running exactly twice as much as I would like 
 - Things are getting very long and I am not a big fan
 - static typing only once everything is working aesvf

# Goals
 - I want to have the profiling script in under _200_ lines 
      - This is including the generalisations and the derivatives 
 - I will have the analysis script in under _50_ lines

So everything is currently a mess. I have not got parameters passing the way that I want I am going to simply remove the burn_in function and move it into main. Passing the steady state as a parameter to the run_solvers. This is much better. I also want to do something about that horrible production functipon. Maybe I can be fancy with some function programming techniques.

I will not be generalising any more since I will make the assumption that the behaviour of the guttler 2014 model is sufficient for the remaining models. The last thing to do in my script would be to organise the profiling of the gradients which I have nearly got worked out. 

