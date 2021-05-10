

import numpy as np


import FLARE
import FLARE.SED.models
import FLARE.filters


def beta(self, filters):

    rest_lam = np.arange(0., 5000., 1.)

    self.filters = filters

    for f in filters:
        self.o[f'intrinsic/flux/{f}'] = np.zeros(self.N)

    for i, (z, beta, log10L) in enumerate(zip(self.o['intrinsic/z'], self.o['intrinsic/beta'], self.o['intrinsic/log10L'])):

        F = FLARE.filters.add_filters(filters, new_lam = rest_lam * (1. + z))
        sed = FLARE.SED.models.beta(rest_lam, beta, 10**log10L, normalisation_wavelength = 1500.)

        sed.get_fnu(self.cosmo, z) # --- generate observed frame spectrum (necessary to get broad band photometry)
        sed.get_Fnu(F) # --- generate broadband photometry

        for f in filters:
            self.o[f'intrinsic/flux/{f}'][i] = sed.Fnu[f]
