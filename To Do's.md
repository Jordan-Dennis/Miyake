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

In the main scope I plan to run `profile_gradients` using the afore constructed equilibrium position. This will prevent that repetition. I still need to run the burn in period, which can also be passed as `u0` whereas I will use `steady_state` to represent the already burnt in position.

I need to resume from testing the `profile_gradients` function. I think that the solver will have no impact on the gradient. I am more interested in seeing if I should foward autodiff for python. 

If this is the case should I do it inn a separate script? There will be common elements but not the ones that I m thinking. I read to fix the naming conventions and this should be a priority.

So something is going wrong. Either in the solution to the ODE or in the underlying model