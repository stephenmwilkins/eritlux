

# Example script for fitting literature \beta as function of luminosity (and redshift)

import numpy as np
import scipy.stats

# --- this allows us to import the module from the directory above. Useful for testing before "deploying" the module
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import eritlux.models.betale as beta_fitter

import time

import matplotlib.pyplot as plt

# Initialise the model, in this case we are including Bouwens+14 data and using a "piecewise" function to fit it
beta_model = beta_fitter.Bouwens2014(beta_fitter.piecewise)


# Display a figure with fits for a range of z
z = np.linspace(4, 8, 100)
log10L_limits = (27, 30)
log10L = np.linspace(*log10L_limits, 100)
dataz = [4, 5, 6, 7] # list of redshifts at which datapoints are shown
# dataz = False
fig, ax, ax_scale = beta_fitter.beta_evo_plot(z, log10L, beta_model, beta_fitter.piecewise, dataz=dataz, print_fit_at_dataz=True, xlims=log10L_limits) #print_fit_at_dataz shows the fits at each z available for data
plt.show()

fig.savefig('figs/betale.pdf')
fig.clf()
