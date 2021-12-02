using PyCall;

ticktack = pyimport("ticktack");

cbm = ticktack.load_presaved_model("Guttler14", production_rate_units="atoms/cm^2/s");

cbm.compile()
cbm.equilibrate(production_rate=1.88)
