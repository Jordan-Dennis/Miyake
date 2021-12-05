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

I need to look into the loss function for the profile gradients function. Lo_likelihood is te ticktack function that I need to emulate.

Ok so this is going to be nearly impossible. No matter how I try their code is just shocking. I mean mine is starting to develop some bugs but seriously what the fuck is that bullshit. They play parse the parsel like there is no tomorrow. So what does the function do? Basically it solves the ODE and then bins the data into yearly groups. Ok so I can cut all of that growth season bull shit out. The profiling forces separateness to the previous solutions. So once I have the binned values I can then calculate a chi-squarred statistic using them 