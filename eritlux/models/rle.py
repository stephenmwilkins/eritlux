# Module containing the models for effective radius

import numpy as np

from scipy.optimize import curve_fit

from FLARE.photom import M_to_lum


def line(z, a, b):
    return a*z + b


class ReffEvolution:

    def __init__(self):

        print(self.model_style)

    def r_eff_sampler(self, z, log10L, L0=M_to_lum(-21), r_eff_lims=False):
        # Version that resamples output below or above the r_eff_lims

        L = 10**log10L

        p = self.parameters(z)
        r_eff_0 = p['r_eff']
        sigma = p['sigma']
        gamma = p['gamma']

        r_eff_bar = r_eff_0 * (L / L0) ** gamma

        r_e = np.random.normal(r_eff_bar, sigma)

        if r_eff_lims:
            if r_e <= r_eff_lims[0]:
                while r_e <= r_eff_lims[0]:
                    r_e = np.random.normal(r_eff_bar, sigma)

            if r_e >= r_eff_lims[1]:
                while r_e >= r_eff_lims[1]:
                    r_e = np.random.normal(r_eff_bar, sigma)


        return r_e


    def r_eff_sampler_lognorm(self, z, log10L, L0=M_to_lum(-21), r_eff_lims=False, test=False, log10=False):
        # Version that resamples output below or above the r_eff_lims

        L = 10**log10L

        p = self.parameters(z)
        r_eff_0 = p['r_eff']
        sigma = p['sigma']
        gamma = p['gamma']

        if test:
            if log10:
                r_eff_bar = np.log10(r_eff_0 * (L / L0) ** gamma)
            else:
                r_eff_bar = np.log(r_eff_0 * (L / L0) ** gamma)

        else:
            r_eff_bar = r_eff_0 * (L / L0) ** gamma

        r_e = np.random.lognormal(r_eff_bar, sigma)

        if r_eff_lims:
            if r_e <= r_eff_lims[0]:
                while r_e <= r_eff_lims[0]:
                    r_e = np.random.lognormal(r_eff_bar, sigma)

            if r_e >= r_eff_lims[1]:
                while r_e >= r_eff_lims[1]:
                    r_e = np.random.lognormal(r_eff_bar, sigma)

        return r_e


    def r_eff_bar(self, z, log10L, L0=M_to_lum(-21)):

        L = 10**log10L

        p = self.parameters(z)
        r_eff_0 = p['r_eff']

        gamma = p['gamma']

        r_eff_bar = r_eff_0 * (L / L0) ** gamma

        return r_eff_bar


class linear(ReffEvolution):
    # --- creates the linear redshift evolution for effective radius.

    def __init__(self, model, z_ref=6.):
        # lp is a dictionary of the parameters of the linear evolution model

        self.z_ref = z_ref

        self.model = model

        self.lp = model.lp

        self.model_style = 'Linear regression r_eff evolution method.'

        super().__init__()


    def parameters(self, z):
        # use linear evolution model
        # get parameters as a function of z
        # returns a dictionary of parameters
        p = {}
        #print(self.z_ref)
        for param in self.lp:
            p[param] = np.maximum(self.lp[param][0] * (z-self.z_ref) + self.lp[param][1], 0.)

        return p




class model:


    def __init__(self):
        self.lp = self.linear_evolution_coefficients()

        print(self.ref)


    def linear_evolution_coefficients(self, z_ref = 6.):

        p_opt_r0, p_cov_r0 = curve_fit(line, np.array(self.redshift)-z_ref, np.array(self.r0), sigma=np.array(self.r0_err[1]), absolute_sigma=True)

        p_opt_sigma, p_cov_sigma = curve_fit(line, np.array(self.redshift)-z_ref, np.array(self.sigma), sigma=np.array(self.sigma_err[1]), absolute_sigma=True)

        p_opt_gamma, p_cov_gamma = curve_fit(line, np.array(self.redshift)-z_ref, np.array(self.gamma), sigma=np.array(self.gamma_err[1]), absolute_sigma=True)


        self.lp = {'r_eff': p_opt_r0, 'sigma': p_opt_sigma, 'gamma': p_opt_gamma}

        return self.lp


class Kawamata2018(model):
    # --- r_eff evolution after Kawamata et al. (2018)

    def __init__(self):
        # Contains model redshift range (must be increasing) and corresponding r_eff evolution model parameters
        # Custom models should be created following the same form

        self.ref = 'Kawamata+2018'

        self.redshift = [6.5, 8, 9]                                    # Array of redshifts

        self.r0 = [0.95, 0.69, 0.53]                                    # Array of modal radii at M_UV = -21
        self.r0_err = [[0.14, 0.14, 0.13], [0.18, 0.24, 0.27]]          # Error on modal radii

        self.sigma = [0.86, 0.62, 0.68]                                 # Width of the log-normal distribution
        self.sigma_err = [[0.07, 0.12, 0.18], [0.09, 0.18, 0.27]]       # Error on the width of log-normal distribution

        self.gamma = [0.47, 0.49, 0.34]                                 # Slope of the size-luminosity relation
        self.gamma_err = [[0.06, 0.14, 0.14], [0.06, 0.13, 0.13]]       # Error on the slope of the size-luminosity relation

        super().__init__()

