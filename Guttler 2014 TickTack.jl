using PyCall;

ticktack = pyimport("ticktack");
numpy = pyimport("numpy");

cbm = ticktack.load_presaved_model("Guttler14", production_rate_units="atoms/cm^2/s");

cbm.compile();
# cbm.equilibrate(production_rate=1.88);

#* I need to make a python production function 

cbm.run(760:790, steady_state_production=1.88)
