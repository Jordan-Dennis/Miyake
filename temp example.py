import ticktack
import numpy as np

cbm = ticktack.load_presaved_model('Guttler14', production_rate_units='atoms/cm^2/s')
# cbm.define_growth_season(['april', 'may', 'june', 'july', 'august', 'september'])
cbm.compile()
import matplotlib.pyplot as plt

start = 760
resolution = 1000
burn_in_time = np.linspace(760 - 1000, 760, resolution)
steady_state_burn_in = cbm.equilibrate(target_C_14=707)
burn_in_solutions = cbm.equilibrate(production_rate=steady_state_burn_in)
d_14_time_series_fine = np.linspace(760, 788, 2700)
d_14_time_series_coarse = np.arange(760, 788)


def sg(t, start_time, duration, area):
    middle = start_time + duration / 2.
    height = area / duration
    return height * np.exp(- ((t - middle) / (1. / 1.88349 * duration)) ** 8.)


def miyake_event(t, start_time, duration, phase, area):
    height = sg(t, start_time, duration, area)
    prod = steady_state_burn_in + 0.18 * steady_state_burn_in * np.sin(2 * np.pi / 11 * t + phase) + height
    return prod


burn_in, _ = cbm.run(burn_in_time, production=miyake_event, args=(775, 1 / 12, np.pi / 2, 81 / 12),
                     y0=burn_in_solutions)
prod = miyake_event(d_14_time_series_fine, 775, 1 / 12, np.pi / 2, 81 / 12)

event, _ = cbm.run(d_14_time_series_fine, production=miyake_event, args=(775, 1 / 12, np.pi / 2, 81 / 12),
                   y0=burn_in[-1, :])
d_14_c = cbm.run_D_14_C_values(d_14_time_series_coarse, 1000, production=miyake_event,
                               args=(775, 1 / 12, np.pi / 2, 81 / 12),
                               y0=burn_in[-1, :], steady_state_solutions=burn_in_solutions)
