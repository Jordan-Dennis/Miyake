# To Do's:
 - I need to look into |> operator to simplify the global namespace 
instead of using local namespaces.
 - I want to have a plot that places things into quadrants based on 
 the speed and the accuracy 
 - I first need to look into how to measure the accuracy of the result
 using an assymptotic behaviour of the RK45 solver as the time step is
 reduced 
 - How the hell does the Brehm equilibriation work?
 - All right. So I use Brehm equilibriation to work out the steady state 
 production and then I use it again to work out the matrix. WTF
 - In the production function I am using 1.88. How does this make sense 
 given the mess of an equilibriation that just occured.
 - I might add the super-gaussian before I check that the outputs match
    - So Dan has a function that processes the sinusoidal part.
    - Dan also has a function for the super-gaussian
    - He then adds the two within another function 
    - Finally he converts the units in a 4th function 
    - Dan calculates the super gaussian and then converts the units
 - So I need to choose step sizes for both the burn in period and the 30 year run time 
    - I am going to use a fixed step size for ease of implementation and 
    accuracy of the comparison.
    - set_proposed_dt may be an option 
    - I also need to look into the integrator() and step! functions which may 
    allow me to move the solver in a or loop 



