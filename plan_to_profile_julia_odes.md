# Profiling Goals
So I want to revisit my julia implementation and run it for all of the models and all of the data sets. Additionally, I want to do a better job on the code aiming to have it in less than 100 lines. This will involve planning each step carefully. The `main` scope of the program should have. 

 - `get_model`: This function will load `model` from the `.hd5` file. I might `return` the `model` as a `Module` or some other structure like a dictionary. `Dict{T, T}(key => value)` constructs a dictionary by element. This is due some experimentation. 
     - Parameters:
         - `model: str` The name of the model that is to be used. 
     - Returns:
         - `Module` So the model needs to contain what? I could have a closure actually. This ould return just a function, but what would that function do. Let's see the things that the model needs are:
             - `production: function` A production function (with parameters)
             - `derivative: function` A derivative function.
         - The `derivative` function includes the production function so this is best returned as a closure. Indeed the entire model is summarised by the derivative since it is an ode. 
         - So I want to return the derivative as a function of parameters. This works really well actually.

 - `get_dc14`: The idea is that this function will and run the model. This function will call `get_model` and will then generate an `ODEProblem` from this. I have the problem that the parameters of the production function need to be passed as a unitary argument to the `ODEProblem` when I want to profile the autodiff
     - Parameters:
         - `solver: str` The name of the solver that to be used by the model.
         - `model: str` The name of the model that is to be used. 
         - I might not pass `model` as a `str` since I will need to generate the model elsewhere and only want to load it into the namespace once.
     - Returns:
         - `function` This is a function of `parameters` that generates the `d14c` of the given model. 

 - `get_chi_squared`: This will produce the loss function and is mostly implemented in `profiler.ipynb`.
     - Parameters:
         - `data_file_name: str` The file name for the data set in question.
         - `carbon_model: Function` The output of a call to `get_carbon`
     - Returns:
         - `Function` A function of `parameters` that generates the chi squared.

 - `profile_function`: This will more or less be taken verbatim from the previous implementation. Although I want to do a more rigorous analysis of the results. Additionally, my new set-up should enable me much more control with composite algorithms. I will have to be carful with how I set this up though and it may require an if statement.
     - Parameters:
         - `function: Function`: The function that is to be profiled.

So at the moment I have a lot of closures as I go through this code. My current problem is working out how I want to pass the model. I think I get the model using `model = get_derivative(model: String)` which I then pass arround through `carbon = get_carbon(solver: String, model: Function)`. So now I can call `carbon(parameters)`. I might rely on definitions in the outer scope to avoid further closures. 

I need to think about where and how I bin the modelled data. I want to make things more similar to the python implementation in this case.

### Naming notes:
I'm going to rename `get_model` as `get_derivative`. I will also rename `get_dc14` as `get_carbon`.

Is it better to have two functions `get_model` and `get_dc14`. I may need `model` in more than one place so I might have the extra `get_model` function 

# MCMC 
So I want to go looking for python implementations of `mcmc` outside of `emcee` and profile them on the different models. This should allow me to reduce the run time of my code considerably.

# ODE Solvers Python
So I want to try the `bosh` solver for each of the models and data sets. This should be easy.