So I want to take all of the data and use bayesian statistics to fit all of the parameters of a production function. I might try multiple different production function models. I also want to have a more advanced carbon box model.

The idea would be to minimize the number of free parameters that were in the production function. I could try with $3$, $4$ or $5$ and see what fit the data most well. 

Using this production function I could then simulate a number of Miyake events with different parameters and shapes. The shape of the event, i.e. super-gaussian, rectangle, lorentzian, ect. could have a major impact on the viability of the model. This could probably be done using the `ticktack` fitters. However, I wanted to try and combine the data by normalising it first. I need to consider the distribution of the $C14$ data before hand and this can be done using some density plots.

