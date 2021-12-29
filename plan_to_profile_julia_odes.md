# Profiling Goals
So I want to revisit my julia implementation and run it for all of the models and all of the data sets. Additionally, I want to do a better job on the code aiming to have it in less than 100 lines. This will involve planning each step carefully. The `main` scope of the program should have. 

 - `get_derivative`: This function will load `model` from the `.hd5` file. I might `return` the `model` as a `Module` or some other structure like a dictionary. `Dict{T, T}(key => value)` constructs a dictionary by element. This is due some experimentation. 
     - Parameters:
         - `model: str` The name of the model that is to be used. 
     - Returns:
         - `Module` So the model needs to contain what? I could have a closure actually. This ould return just a function, but what would that function do. Let's see the things that the model needs are:
             - `production: function` A production function (with parameters)
             - `derivative: function` A derivative function.
         - The `derivative` function includes the production function so this is best returned as a closure. Indeed the entire model is summarised by the derivative since it is an ode. 
         - So I want to return the derivative as a function of parameters. This works really well actually.
         - OK. I have decided that the binning will automatically occur within this function based on the hemisphere loaded from the `.hd5` file. I need to check this against `ticktack`

 - `get_carbon`: The idea is that this function will and run the model. This function will call `get_derivative` and will then generate an `ODEProblem` from this. I have the problem that the parameters of the production function need to be passed as a unitary argument to the `ODEProblem` when I want to profile the autodiff
     - Parameters:
         - `solver: str` The name of the solver that to be used by the model.
         - `model: str` The name of the model that is to be used. 
         - I might not pass `model` as a `str` since I will need to generate the model elsewhere and only want to load it into the namespace once.
     - Returns:
         - `function` This is a function of `parameters` that generates the `d14c` of the given model. 

 - `get_chi_squared`: This will produce the loss function and is mostly implemented in `profiler.ipynb`. The binning will happen raw within this fuction and will take into account the growth season which can be retrieved from the hemisphere of the model. I will need to work out how this interfaces with the `get_derivative` function. 
     - Parameters:
         - `data_file_name: str` The file name for the data set in question.
         - `carbon_model: Function` The output of a call to `get_carbon`
     - Returns:
         - `Function` A function of `parameters` that generates the chi squared.

 - `profile_function`: This will more or less be taken verbatim from the previous implementation. Although I want to do a more rigorous analysis of the results. Additionally, my new set-up should enable me much more control with composite algorithms. I will have to be carful with how I set this up though and it may require an if statement.
     - Parameters:
         - `function: Function`: The function that is to be profiled.
     - Returns:
         - `DataFrame` This will contain raw data to be analysed

So at the moment I have a lot of closures as I go through this code. My current problem is working out how I want to pass the model. I think I get the model using `model = get_derivative(model: String)` which I then pass arround through `carbon = get_carbon(solver: String, model: Function)`. So now I can call `carbon(parameters)`. I might rely on definitions in the outer scope to avoid further closures. 

## Psuedo-Code
So I need to weigh up the pros and the cons of using `Dict` and `String` as the storage mechanism fof the model and the data files. If I use a `Dict` then I do not need to retrieve the simplified name for plotting purposes later on. I'm not sure about the performance differences of each.
```
models = Dict{String: String}(name, hd5_file_address)
data_sets = Dict{String: String}(name, csv_file_address)
solvers = Array{String}(solve_name)

for model in models:
    derivative::Function = get_derivative(model::String)
    for data_set in data_sets:
        for solver in solvers:
            carbon::Function = get_carbon(solver::String, derivative::Function)
            chi_squared::Function = get_chi_squared(data_file_name::String, carbon::Function)

            carbon_profile::DataFrame = profile_function(carbon::Function, parameters::Vector{Float64})
            gradient_profile::DataFrame = profile_function(ForwardDiff.gradient::Function, (parameters::Vector{Float64}, chi_squared::Function))
            hessian_profile::DataFrame = profile_function(ForwardDiff.hessain::Function, (parameters::Vector{Float64}, chi_squared::Function))
```

The goal is to have what as an output. I want a `DataFrame` but what will it look like?

```
DataFrame:
model::String | dataset::String | solver::String | time::Float64 | time_error::Float64 | accuracy::Float64 | accuracy_error::Float64
```

Alternately I could just have a fucking huge `DataFrame` with all of the time runs. I really want a better system for the accuracy though.

# MCMC 
So I want to go looking for python implementations of `mcmc` outside of `emcee` and profile them on the different models. This should allow me to reduce the run time of my code considerably.

# ODE Solvers Python
So I want to try the `bosh` solver for each of the models and data sets. This should be easy.