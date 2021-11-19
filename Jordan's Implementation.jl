using PyCall 

ticktack = pyimport("ticktack");
jaxnumpy = pyimport("jax.numpy");
numpy = pyimport("numpy")

# The first thing to do is to write the sine function in julia so that it can be used
function sine(t)
    prod =  1.87 + 0.7 * 1.87 * jaxnumpy.sin(2 * jaxnumpy.pi / 11 * t + jaxnumpy.pi/2);
    prod = prod * (t>=sf.start) +
        (1.87 + 0.18 * 1.87 * jaxnumpy.sin(2 * jaxnumpy.pi / 11 * sf.start + jaxnumpy.pi/2)) *
        (1-(t>=sf.start));
    return prod
end

cbm = ticktack.load_presaved_model("Guttler14", production_rate_units = "atoms/cm^2/s");
sf = ticktack.fitting.SingleFitter(cbm);
sf.prepare_function(f=sine);
sf.time_data = jaxnumpy.arange(200, 230) ;
sf.d14c_data_error = jaxnumpy.ones((sf.time_data.size,));
sf.start = jaxnumpy.nanmin(sf.time_data);
sf.end = jaxnumpy.nanmax(sf.time_data);
sf.resolution = 1000;
sf.burn_in_time = jaxnumpy.linspace(sf.start-1000, sf.start, sf.resolution);
sf.time_grid_fine = jaxnumpy.arange(sf.start, sf.end, 0.05);
sf.time_oversample = 1000;
sf.offset = 0;
sf.gp = true;
sf.annual = jaxnumpy.arange(sf.start, sf.end + 1);
sf.mask = last(jaxnumpy.in1d(sf.annual, sf.time_data));

numpy.random.seed(0);
d14c = sf.dc14(); # Recall that d_c_14 = d_c_14[self.mask] is comment in the egg

noisy_d14c = numpy.array(d14c) + numpy.random.randn(d14c.size); # add unit gaussian noise
noisy_d14c = numpy.append(noisy_d14c, last(noisy_d14c)); # for compatibility with ticktack code