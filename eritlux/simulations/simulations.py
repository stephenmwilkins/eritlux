

import os
import numpy as np
from scipy.stats import uniform
import h5py

import flare

from . import intrinsic
from . import photo
from . import pz
from .. import selection


class delta():
    def __init__(self, value):
        self.value = value
    def rvs(self, N = None):
        if N:
            return self.value*np.ones(N)
        else:
            return self.value



def default_prange(sed_model, profile_model):

    prange = {}

    prange['z'] = uniform(*[6, 4]) # uniform from z = 6 to 13

    if profile_model == 'simple':
        pass
    elif profile_model == 'cSersic':
        prange['log10r_eff_kpc'] = uniform(*[-0.5, 1.0])
        prange['n'] = delta(1.0)
        prange['ellip'] = delta(0.0)
        prange['theta'] = delta(0.0)
    else:
        print('WARNING: model not yet implemented')

    if sed_model == 'beta':
        prange['beta'] = uniform(*[-3., 4]) # uniform from \beta = -3 to 1
        prange['log10L'] = uniform(*[27, 3])
    else:
        print('WARNING: model not yet implemented')

    return prange





class Simulation():

    intrinsic_beta = intrinsic.beta
    photo_idealised = photo.idealised
    photo_idealisedimage = photo.idealisedimage
    photo_realimage = photo.realimage
    pz_idealised = pz.idealised
    pz_eazy = pz.eazy
    apply_selection = selection.apply_selection


    def __init__(self, morph_model = 'simple', sed_model = 'beta', run_id = 0, prange = False, cosmo = flare.default_cosmo(), verbose = False):

        self.verbose = verbose
        self.run_id = str(run_id) # needed here to label the EAZY run, could do something else? e.g. random
        self.morph_model = morph_model
        self.profile_model = morph_model
        self.sed_model = sed_model
        self.cosmo = cosmo

        if prange:
            self.prange = prange
        else:
            self.prange = default_prange(profile_model = self.profile_model, sed_model = self.sed_model)

    def i(self, i=0):

        # --- returns a dictionary with just one object
        return {k: v[i] for k,v in self.o.items()}


    def n(self, N):

        self.N = N

        # --- define input properties
        self.o = {} #
        for param, f in self.prange.items():
            self.o[f'intrinsic/{param}'] = f.rvs(N)

        # --- calculate observed size
        if self.profile_model in ['cSersic', 'Sersic']:
            self.o['intrinsic/r_eff_kpc'] = 10**self.o['intrinsic/log10r_eff_kpc']
            self.o['intrinsic/r_eff_arcsec'] = self.o['intrinsic/r_eff_kpc'] * self.cosmo.arcsec_per_kpc_proper(self.o['intrinsic/z']).value


    def export_to_HDF5(self, output_dir, output_filename, return_hf = False):

        # --- make directory structure for the output files
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        hf = h5py.File(f'{output_dir}/{output_filename}.h5', 'w')


        hf.attrs['N'] = self.N
        hf.attrs['profile_model'] = self.profile_model
        hf.attrs['morph_model'] = self.morph_model
        hf.attrs['sed_model'] = self.sed_model

        for k, v in self.o.items():
            hf.create_dataset(k, data = v)

        if return_hf:
            return hf
        else:
            hf.flush()
