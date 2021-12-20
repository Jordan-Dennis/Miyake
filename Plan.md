So I want to take all of the data and use bayesian statistics to fit all of the parameters of a production function. I might try multiple different production function models. I also want to have a more advanced carbon box model.

The idea would be to minimize the number of free parameters that were in the production function. I could try with $3$, $4$ or $5$ and see what fit the data most well. 

Using this production function I could then simulate a number of Miyake events with different parameters and shapes. The shape of the event, i.e. super-gaussian, rectangle, lorentzian, ect. could have a major impact on the viability of the model. This could probably be done using the `ticktack` fitters. However, I wanted to try and combine the data by normalising it first. I need to consider the distribution of the $C14$ data before hand and this can be done using some density plots.

## Implementation
I have the choice of $4$ programming languages. _R_, _Python_, _Julia_ and _LuaJIT_. I would learn the most using _LuaJIT_ or _Julia_. I like _Gadfly_ from _Julia_ but would also like some more experience using _Matplotlib_ and _Pandas_ from _Python_. If I used _Python_ I would have access to the _Ticktack_ tools which would greatly reduce the amount of work that I had to do. 

I will take time and chose the right tool for the job. This will need to be objective goddammit.

### Python:
 - #### Pros
     - Ben will be able to help.
     - _Ticktack_ is available with existing implementations.
     - Widely used in industry.
     - Will have the opportunity to improve my skill with the libraries of _Scipy_.
 - #### Cons
     - I'm not a huge fan of _Python_ for numerical code.
     - Code will be slow.
     - I could learn more using other languages

### R:
 - #### Pros
     - Excellent programming experience for data science.
         - Piplines `%>%`
         - `ggplot2` for fast density plots
 - #### Cons
     - Slow
     - Ben will not be able to help 
     - Increasingly not used in industry

### Julia:
 - #### Pros
     - Statically typed
     - Fast 
     - I would a similar amount to python
     - Increasingly used in industry 
 - #### Cons
     - Ben could not help

### LuaJIT
 - #### Pros
     - Very fast 
     - I would learn the most using LuaJIT
 - #### Cons
     - Barely used in industry 
     - Ben would not help 
     - Very limited ecosystem

I think that I can rule out _R_ since the loop implementation is poor. _R_ is strongest for investigating data in a vectorised way and not when it comes to simulations and numerical code. __I need to fact check that _R_ has a relatively poor array implementation__. I could choose to use _R_ for the exploratory analysis and then switch to _Julia_ for the numerical code. I kind of like this idea and it is supported by _Jupyter_ so that makes it very easy. At the moment I am leaning towards _Julia_ or a combination/comparison. Using _Julia_ for a very complex carbon box model would allow me the freedom to develop my own code instead of using _Ticktack_. 

An interesting idea would be to compare many differently sized CBMs and see if I could identify an ideal size.

I might start off by generating a denisity plot of the $C14$ data from the different studies and then go about looking at normalisation. I can normalise via the mean, median and mode. I should do so research into which of these methods is best in each case.

Turns out that there is a normalisation formula so I will begin with researching this.

Another thing would be to do a fourier analysis of the production function. The psuedo-code would be something like this.
 - Generate a sinusoid with 2 parameters, amplitude and phase. 
 - Minimise the $\chi^{2}$ statistic using gradient descent of each parameter.
 - Add another sinusoidal term until some cut-off $\chi^{2}$ is reached.
A problem wih this method is it is a local analytical solution that cannot be used to extrapolate and which does not have physical significance.

So the plan for today:
 - I need to sort out my directory.
 - I think I will go with the .file_ext
     - `.csv`
     - `.ipynb`
     - `.hd5`
     - `.py`
     - `.jl`
     - `.r`
 - As well as this I might also have minimal running environments/directories
     - These will have a name formatted `like julia_ode_profiles`.
 - In terms of my actual programming goals for today
     - I want to have a function `get_residual_distribution`, which will use `mcmc` to determine the noise around the simulated model. I will define another function within this namespace called `get_production_function`, which will use the single fitter type events to generate the ideal production function from which the residuals and their distribution can be calculated.
     - Next I will have a function `simulate_event` which will do just that. I'm not sure about the parameters but it will use the error distribution calculated by `get_residual_distribution` (not sure how this will work) and the production function determined by `get_production_function` to generate a simulated data set with an event.
     - I will have a function `get_event` which will use some form of statistical analysis to detect the event with some degree of certainty.
     - The basic usage will be:
         - loop over the datasets using `for data_file in list(data_files):`
             - loop over the models using `for model_file in list(model_files):`
                 - `load_data` into global namespace
                 - Use `load_presaved_model` for a model 
                 - Use `get_production_function` to determine the model
                 - Use `get_residual_distribution` to determine the sampling distribution
                 - loop over a range of widths using `for width in list(widths):`
                     - loop over a range of heights using `for height in list(heights):`
                         - Use `get_event` to simulate and event with the parameters `width` and `height`.
                         - Construct a contour plot using `plotnine` of the peak measure within the parameter space and save it as a `.pdf` 

A class could work quite well here. `set_model` would let me chose the `.hd5` file to get the model from. `set_data` would let me chose the data set and the rest could be done using the functions above.

There is already bloat and I want to simulate more or less all of the cases so I might as well use a double nested `for` loop.