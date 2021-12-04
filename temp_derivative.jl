using DifferentialEquations;    # for the differential equations 
using ForwardDiff;              # For the derivatives

production(p, t) = uf * (1.88 + 0.18 * 1.88 * sin(2 * π / p * t) +  # Sin wave
20 * 1.60193418235 * exp(-(12 * (t - 775)) ^ 16));          # Super gaussian
∇(x, p, t) = vec(TO * x + production(p, t) * P);            # Derivative with extra argument 
hi(p) = ODEProblem(∇, burn_in[end], (760.0, 790.0), p);     # An ODEProblem that is a function of the parameters 
f(p) = solve(hi(p));                                        # Wrapping complete. #! for fucks sake this is like parse the parsel
bye = ForwardDiff.gradient(f, 11.0);                        # Implementation 1