using DifferentialEquations;    # for the differential equations 
using ForwardDiff;              # For the derivatives

#* Alright let's clean this code up and write a production function 
#* that will give me all the information that I need 
derivative(t, p)
hi(p) = ODEProblem(∇, burn_in[end], (760.0, 790.0), p);     # An ODEProblem that is a function of the parameters 
f(p) = solve(hi(p));                                        # Wrapping complete. #! for fucks sake this is like parse the parsel
bye = ForwardDiff.gradient(f, 11.0);                        # Implementation 1