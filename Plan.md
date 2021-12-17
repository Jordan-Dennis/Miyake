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

I might start off by generating a denisity plot of the $C14$ data from the different studies and then go about looking at normalisation. I can normalise via the mean, median and mode. I should do so research into which of these methods is best in each case.

Turns out that there is a normalisation formula so I will begin with researching this.