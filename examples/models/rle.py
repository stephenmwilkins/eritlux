# Example script for fitting literature r_eff as function of luminosity (and redshift)

import numpy as np

from eritlux.models import rle

# Initialising the r_eff(L, z) model (this case using Kawamata+18 data)
reff_model = rle.linear(rle.Kawamata2018())

print(reff_model.lp)

#list['log10r_eff'][i] = np.log10(reff_model.r_eff_sampler_lognorm(z_, list['log10L1500'][i], r_eff_lims=[0., 20]))
