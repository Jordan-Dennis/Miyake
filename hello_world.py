import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import ticktack
from ticktack import fitting

from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = (16.0, 8.0)

# cbm = ticktack.load_presaved_model('Guttler14', production_rate_units = 'atoms/cm^2/s')
# cf = fitting.CarbonFitter(cbm)
# cf.load_data('miyake12.csv')
# cf.prepare_function(production='miyake', fit_solar=False)

# default_params = [775., 1./12, np.pi/2., 81./12]
# sampler = cf.sampling(default_params, burnin=500, production=1000)

# cf.corner_plot(sampler, labels=[r"Start Date (yr)", r"Duration (yr)", r"$\phi$ (yr)", r"Area"])

# cf.plot_samples(sampler)