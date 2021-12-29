# TO DOs:
 - I need to plan the `mcmc` profiling
 - I need to plan the `bosh` profiling 
 - I need to organise my directory
 - I need to devlop a naming convention 
 - I need to fix the injection recovery python script.
 - I need to add a `readme.md` that contains
     - directory key: This will explain where everything is and how to access it. 
         - Nicely formatted map of the directory 
 - I need to make my repo public following the improvements

## Goals: 


## Aims:
```
Investigate Miyake events:
    | Investigation
    |   | Injection recovery across all data sets and all models
    |   
    | Implementation  
    |   | Profile a sample of the ODE solvers in DifferentialEquations.jl
    |   | Profile a sample of the mcmc algorithms that are available in python
    |   | Profile the bosh solver against the JAX odeint for each model              
```

## Organisation:
```
logical_division (Taken from aims:)
    | plan.md
    | logical_division.ipynb
    | figures.pdf
    |
    | datasets (dir)
    |   | year
    |   |   | hemisphere
    |   |   |   | data_set.csv
    |
    | models (dir)
    |   | model.hd5
```


