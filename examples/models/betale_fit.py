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

# The model automatically finds curve_fits to first \beta(L) for each redshift individually and then linear fits to
# those in order to find the \beta(z), parameters for each are printed below

print("")
print(r"Model best fit parameters for \beta(log_{10}[L]) at each z:")
print(beta_model.lp)
print(r"Model best fit parameters for \beta(z, log_{10}[L]):")
print(beta_model.zp)

# The code cal also fit the full evolution in both z and log10L simultaneously using bayesian inference:
# Here we select the fitter (currently two options, fitter_emcee and fitter_zeus) and feed it the initialised model
source = beta_fitter.fitter_emcee(beta_model)

# Adding prior space for initialising walker positions
source.priors['a11'] = scipy.stats.uniform(loc = -0.5, scale = 1.)
source.priors['a12'] = scipy.stats.uniform(loc = -0.3, scale = 0.6)
source.priors['a21'] = scipy.stats.uniform(loc = -1., scale = 2.)
source.priors['a22'] = scipy.stats.uniform(loc = -1., scale = 2.)
source.priors['b11'] = scipy.stats.uniform(loc = -1., scale = 2.)
source.priors['b12'] = scipy.stats.uniform(loc = -3., scale = 1.5)
source.priors['b21'] = scipy.stats.uniform(loc = -2., scale = 4.)
source.priors['b22'] = scipy.stats.uniform(loc = 27., scale = 3)

# Fitting the model and timing it
t1 = time.time()
samples = source.fit(nwalkers=50, nsamples = 1000, burn=500)
t2 = time.time()

print(f'took {t2-t1} seconds')

# Here we export the results and display median plus 68-percentile range limits
import _pickle as pickle

pickle.dump(samples, open('output/betale_fit_samples.p','wb'))

analysis = beta_fitter.analyse(samples, parameters = ['a11','a12','a21', 'a22', 'b11', 'b12', 'b21', 'b22'])
analysis.P() # print central 68 and median with Truth if provided

# Save the full corner plot for samples
analysis.corner(filename = 'output/betale_fit_corner.pdf') # produce corner plot


# Display a figure with fits for a range of z
z = np.linspace(4, 8, 100)
log10L = np.linspace(27, 30, 100)
dataz = [4, 5, 6, 7] # list of redshifts at which datapoints are shown
beta_fitter.beta_evo_plot(z, log10L, beta_model, beta_fitter.piecewise, dataz=dataz, print_fit_at_dataz=True) #print_fit_at_dataz shows the fits at each z available for data
#plt.show()