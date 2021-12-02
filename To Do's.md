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
 - Am I getting propotions as my output?
 - I am going to create a new .hd5 file 

# Problem's
 - So the decay matrixes are not the same.
 - Neither is the transfer operator.
 - The non-diagonal elements are fine. The diagonals are the problem.
 - So they are not using the different C14 decay values in each box 
 but rather are using the constant rate based on half life. 
 - The problem seems to be limited to the $\lambda$ 
 - I can fix this by using the fact that $$\frac{dN}{dt} = A = -\lambda N$$
 - I just need to get A or N in the right units
 - I should just be able to divide by the $60 \times$ avagardro's number 
 divided by the molecular mass of C14.

# **Urgent**
 - _New .hd5_ without decay field.