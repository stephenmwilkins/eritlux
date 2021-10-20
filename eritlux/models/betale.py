# Module containing the models for UV beta slope

# ****************************THIS NEEDS TO BE TIGHTENED UP AFTER SOME TESTING*********************************
# ****************************************REWORKING IN PROGRESS************************************************

import zeus

import numpy as np

from scipy.optimize import curve_fit

from scipy.stats.distributions import chi2

from scipy.special import erf

from scipy.interpolate import interp1d

from flare.photom import M_to_lum

from types import SimpleNamespace

import emcee

import corner

import matplotlib.pyplot as plt
import matplotlib as mpl

# \beta(M) models

def linear(x, a, b):
    x_lim = np.log10(M_to_lum(-18.8))
    return a*(x-x_lim) + b


def piecewise_smooth(x, a1, a2, b1, b2):
    x_lim = b2 #np.log10(M_to_lum(-18.8))
    stretch = 3
    return (np.maximum(a1, 0)*(x-x_lim) + b1) + 0.5*(1 + np.tanh(stretch*(x-x_lim)))*(a2*(x-x_lim)-a1*(x-x_lim))


def piecewise_interpolation_test(x, a1, a2, b1, b2):
    x_lim = b2 #np.log10(M_to_lum(-18.8))
    s_l = x<=x_lim
    s_h = [not item for item in s_l]

    x_new = np.linspace(x[s_l][-1]-x_lim, x[s_h][0]-x_lim, 1000)

    x_interp = np.array([x[s_l][-2], x[s_l][-1], x[s_h][0], x[s_h][1]])-x_lim

    trans = interp1d(x_interp, np.array([a1*(x_interp[0])+b1, a1*(x_interp[1])+b1, a2*(x_interp[2])+b1, a2*(x_interp[3])+b1]), kind='quadratic')

    x_out = np.concatenate((np.array([x_i for x_i in x[s_l][:-1]] + [x_j for x_j in x[s_h][1:]]), x_new), axis=None)
    y_out = np.concatenate((np.array([(a1*(x_i-x_lim) + b1) for x_i in x[s_l][:-1]] + [(a2*(x_j-x_lim) + b1) for x_j in x[s_h][1:]]), trans(x_new)), axis=None)

    return x_out, y_out


def piecewise_smooth_2(x, a1, a2, b1, b2):
    x_lim = b2 #np.log10(M_to_lum(-18.8))
    stretch = 5
    return (np.maximum(a1, 0)*(x-x_lim))*(0.5*(1-erf(stretch*(x-x_lim)))) + (a2*(x-x_lim))*(0.5*erf(stretch*(x-x_lim))+1) + b1


def piecewise(x, a1, a2, b1, b2):
    x_lim = b2 #np.log10(M_to_lum(-18.8))
    s_l = x<=x_lim
    s_h = [not item for item in s_l]

    # np.array([(a1*(x_i-x_lim) + b1) for x_i in x[s_l]] + [(a2*(x_j-x_lim) + b1) for x_j in x[s_h]])
    return np.concatenate(((np.maximum(a1, 0)*(x[s_l] - x_lim) + b1),  (a2*(x[s_h] - x_lim)) + b1), axis=None)


def piecewise_test(x, a1, a2, b1, b2):
    x_lim = b2
    s_l = x<=x_lim
    s_h = [not item for item in s_l]

    # np.array([(a1*(x_i-x_lim) + b1) for x_i in x[s_l]] + [(a2*(x_j-x_lim) + b1) for x_j in x[s_h]])
    return np.concatenate(((np.maximum(a1, 0)*(x[s_l] - x_lim) + b1),  (a2*(x[s_h] - x_lim)) + b1), axis=None)

def z_line(z, a, b):
    z_lim = 6.
    return a*(z-z_lim)+b



def model_2d(z, log10L, a11, a12, a21, a22, a31, a32, a41, a42):

    a1 = a11 * z + a12
    a2 = a21 * z + a22
    b1 = a31 * z + a32
    b2 = a41 * z + a42

    return piecewise(log10L, a1, a2, b1, b2)


def model_2d2(z, log10L, p, model):

    arr = []


    for i, zi in enumerate(z):
        a1 = p['a1'][0] * zi + p['a1'][1]
        a2 = p['a2'][0] * zi + p['a2'][1]
        b1 = p['b1'][0] * zi + p['b1'][1]
        b2 = p['b2'][0] * zi + p['b2'][1]

        for j, log10Lj in enumerate(log10L):
            y = model(log10Lj, a1, a2, b1, b2)
            arr.append(y)

    return np.array(arr)


class fitter_emcee():


    def __init__(self, beta_model):

        self.input = beta_model.input
        self.input_z = beta_model.input_z

        self.parameters = ['a11','a12','a21', 'a22', 'b11', 'b12', 'b21', 'b22']
        self.priors = {}

    def lnlike(self, params):
        """log Likelihood function"""

        v = 0.
        for zi in self.input_z:
            a1 = params[0] * (zi - 6) + params[1]
            a2 = params[2] * (zi - 6) + params[3]
            b1 = params[4] * (zi - 6) + params[5]
            b2 = params[6] * (zi - 6) + params[7]

            v += np.sum(-0.5 * (self.input['z' + str(zi)].beta_biweight - piecewise(self.input['z' + str(zi)].log10L, a1, a2, b1, b2)) ** 2 / (self.input['z' + str(zi)].beta_biweight_err[1] ** 2) *
                        np.sqrt(np.sum(self.input['z' + str(zi)].num_sources)/np.sum(self.input['z4'].num_sources)))

        if not np.isfinite(v):
            return -np.inf

        return v

    def lnprob(self, params):
        """Log probability function"""

        p = {parameter: params[i] for i, parameter in enumerate(self.parameters)}

        lp = np.sum([self.priors[parameter].logpdf(p[parameter]) for parameter in self.parameters])

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.lnlike(params)


    def fit(self, nwalkers = 50, nsamples = 1000, burn = 200):

        self.ndim = len(self.parameters)
        self.nwalkers = nwalkers
        self.nsamples = nsamples

        p0 = [ [self.priors[parameter].rvs() for parameter in self.parameters] for i in range(nwalkers)]

        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.lnprob, args=())
        pos, prob, state = self.sampler.run_mcmc(p0, burn)
        self.sampler.reset()
        self.sampler.run_mcmc(pos, nsamples)

        chains = self.sampler.chain[:, :, ].reshape((-1, self.ndim))
        samples = {p: chains[:,ip] for ip, p in enumerate(self.parameters)}

        return samples


class fitter_zeus():


    def __init__(self, beta_model):

        self.input = beta_model.input
        self.input_z = beta_model.input_z

        self.parameters = ['a11','a12','a21', 'a22', 'b11', 'b12', 'b21', 'b22']
        self.priors = {}

    def lnlike(self, params):
        """log Likelihood function"""
        # Note: this uses a weighting that attempts to penalise low object counts

        v = 0.
        for zi in self.input_z:
            a1 = params[0] * (zi - 6) + params[1]
            a2 = params[2] * (zi - 6) + params[3]
            b1 = params[4] * (zi - 6) + params[5]
            b2 = params[6] * (zi - 6) + params[7]

            v += np.sum(-0.5 * (self.input['z' + str(zi)].beta_biweight - piecewise(self.input['z' + str(zi)].log10L, a1, a2, b1, b2)) ** 2 / (self.input['z' + str(zi)].beta_biweight_err[1] ** 2) *
                        np.sqrt(np.sum(self.input['z' + str(zi)].num_sources)/np.sum(self.input['z4'].num_sources)))

        if not np.isfinite(v):
            return -np.inf

        return v

    def lnprob(self, params):
        """Log probability function"""

        p = {parameter: params[i] for i, parameter in enumerate(self.parameters)}

        lp = np.sum([self.priors[parameter].logpdf(p[parameter]) for parameter in self.parameters])

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.lnlike(params)


    def fit(self, nwalkers = 50, nsamples = 1000, burn = 200):

        self.ndim = len(self.parameters)
        self.nwalkers = nwalkers
        self.nsamples = nsamples

        p0 = [ [self.priors[parameter].rvs() for parameter in self.parameters] for i in range(nwalkers)]

        self.sampler = zeus.EnsembleSampler(nwalkers, self.ndim, self.lnprob, args=())  # Initialise the sampler
        self.sampler.run_mcmc(p0, nsamples+burn)  # Run sampling

        chains = self.sampler.get_chain(flat=True, discard=burn, thin=10)
        samples = {p: chains[:,ip] for ip, p in enumerate(self.parameters)}

        return samples


class BetaOfZ:
    def __init__(self):

        print('Linear regression UVC beta evolution method.')

        self.zp, self.zc = self.calculate_z_evolution_coefficients()

    def calculate_z_evolution_coefficients(self):

        zp = {}
        zc = {}

        if self.model == linear:
            zp['a1'], zc['a1'] = curve_fit(z_line, self.input_z, [self.lp[label][0] for label in self.labels])
            zp['b1'], zc['b1'] = curve_fit(z_line, self.input_z, [self.lp[label][1] for label in self.labels])

        if self.model == piecewise_smooth:
            zp['a1'], zc['a1'] = curve_fit(z_line, self.input_z, [self.lp[label][0] for label in self.labels])
            zp['a2'], zc['a2'] = curve_fit(z_line, self.input_z, [self.lp[label][1] for label in self.labels])
            zp['b1'], zc['b1'] = curve_fit(z_line, self.input_z, [self.lp[label][2] for label in self.labels])
            zp['b2'], zc['b2'] = curve_fit(z_line, self.input_z, [self.lp[label][3] for label in self.labels])

        if self.model == piecewise_smooth_2:
            zp['a1'], zc['a1'] = curve_fit(z_line, self.input_z, [self.lp[label][0] for label in self.labels])
            zp['a2'], zc['a2'] = curve_fit(z_line, self.input_z, [self.lp[label][1] for label in self.labels])
            zp['b1'], zc['b1'] = curve_fit(z_line, self.input_z, [self.lp[label][2] for label in self.labels])
            zp['b2'], zc['b2'] = curve_fit(z_line, self.input_z, [self.lp[label][3] for label in self.labels])

        if self.model == piecewise_test:
            zp['a1'], zc['a1'] = curve_fit(z_line, self.input_z, [self.lp[label][0] for label in self.labels], sigma= [1/np.sum(self.input[label].num_sources) for label in self.labels])
            zp['a2'], zc['a2'] = curve_fit(z_line, self.input_z, [self.lp[label][1] for label in self.labels], sigma= [1/np.sum(self.input[label].num_sources) for label in self.labels])
            zp['b1'], zc['b1'] = curve_fit(z_line, self.input_z, [self.lp[label][2] for label in self.labels], sigma= [1/np.sum(self.input[label].num_sources) for label in self.labels])
            zp['b2'], zc['b2'] = curve_fit(z_line, self.input_z, [self.lp[label][3] for label in self.labels], sigma= [1/np.sum(self.input[label].num_sources) for label in self.labels])

        if self.model == piecewise:
            zp['a1'], zc['a1'] = curve_fit(z_line, self.input_z, [self.lp[label][0] for label in self.labels])
            zp['a2'], zc['a2'] = curve_fit(z_line, self.input_z, [self.lp[label][1] for label in self.labels])
            zp['b1'], zc['b1'] = curve_fit(z_line, self.input_z, [self.lp[label][2] for label in self.labels])
            zp['b2'], zc['b2'] = curve_fit(z_line, self.input_z, [self.lp[label][3] for label in self.labels])

        return zp, zc


    def beta_z_log10L(self, z, log10L):

        if self.model == linear:
            return self.model(log10L, z_line(z, *self.zp['a1']), z_line(z, *self.zp['b1']))

        if self.model == piecewise:
            return self.model(log10L, z_line(z, *self.zp['a1']), z_line(z, *self.zp['b1']), z_line(z, *self.zp['a2']))


    def beta_z_log10L_sampler(self, z, log10L, sigma=0.1):

        if self.model == linear:
            return np.random.normal(self.model(log10L, z_line(z, *self.zp['a1']), z_line(z, *self.zp['b1'])), sigma)

        if self.model == piecewise:
            return np.random.normal(self.model(log10L, z_line(z, *self.zp['a1']), z_line(z, *self.zp['b1']), z_line(z, *self.zp['a2'])), sigma)


class betafitter(BetaOfZ):

    def __init__(self):

        print(self.ref)

        self.labels = ['z'+str(item) for item in np.linspace(*self.z_range, self.z_range[1]-self.z_range[0]+1, dtype=int)]

        self.input = {}
        for label in self.labels: self.input[label] = self.data[label]

        self.input_z = np.array(self.redshift)[(np.array(self.redshift) >= self.z_range[0])&(np.array(self.redshift)<= self.z_range[1])]

        self.lp, self.le = self.beta_of_log10L_coeffs()
        self.chisq = self.chisquared()


        super().__init__()

    def beta_of_log10L_coeffs(self):

        lp = {}
        le = {}

        lims = self.fit_bounds
        for label in self.labels:
            p_opt, p_cov = curve_fit(self.model, self.input[label].log10L, self.input[label].beta_biweight, bounds=lims) #sigma=self.input[label].beta_biweight_err[1], absolute_sigma=True,

            lp[label] = p_opt
            le[label] = p_cov

        return lp, le

    def chisquared(self):

        chisq = {}

        for label in self.labels:
            chi_squared = np.sum(((self.model(np.array(self.input[label].log10L), *self.lp[label]) -
                                   np.array(self.input[label].beta_biweight)) / np.array(self.input[label].beta_biweight_err[1])) ** 2)

            ndf = len(self.input[label].log10L) - len(self.lp[label])
            chired = chi_squared / ndf
            prob = chi2.sf(chi_squared, ndf)

            chisq[label] = [chi_squared, chired, ndf, prob]

        return chisq


class Bouwens2014(betafitter):
    # --- \beta(L) evolution after Bouwens et al. (2014)

    def __init__(self, model, z_range=[4, 6], fit_bounds=[[0, 0, -10, 27],[np.inf, np.inf, 0, 30]]):
        # Contains model redshift range (must be increasing) and corresponding LF evolution model parameters
        # Custom models should be created following the same form

        self.model = model

        self.z_range = z_range
        self.fit_bounds = fit_bounds

        self.ref = 'Bouwens+2014'

        z4 = SimpleNamespace()
        z4.M_UV = np.array(np.flipud([-21.75, -21.25, -20.75, -20.25, -19.75, -19.25, -18.75, -18.25, -17.75, -17.25, -16.75, -16.25,
                   -15.75]))
        z4.log10L = np.log10(M_to_lum(z4.M_UV))
        z4.beta_biweight = np.flipud(np.array([-1.54, -1.61, -1.70, -1.80, -1.81, -1.90, -1.97, -1.99, -2.09, -2.09, -2.23, -2.15, -2.15]))
        z4.beta_biweight_err = np.array([np.flipud([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]),
                                np.flipud([0.07, 0.04, 0.03, 0.02, 0.03, 0.02, 0.06, 0.06, 0.08, 0.07, 0.10, 0.12, 0.12])])
        z4.beta_median = np.array(np.flipud([-1.49, -1.61, -1.70, -1.81, -1.81, -1.88, -1.96, -1.98, -2.00, -2.07, -2.25, -2.19, -2.16]))
        z4.beta_inv_var = np.array(np.flipud([-1.42, -1.52, -1.57, -1.71, -1.74, -1.85, -1.90, -1.97, -1.92, -2.04, -2.10, -1.94, -1.95]))
        z4.num_sources = np.array(np.flipud([54, 141, 285, 457, 552, 586, 57, 70, 94, 69, 86, 96, 53]))

        z5 = SimpleNamespace()
        z5.M_UV = np.array(np.flipud([-21.75, -21.25, -20.75, -20.25, -19.75, -19.25, -18.75, -18.25, -17.75, -17.25, -16.50]))
        z5.log10L = np.log10(M_to_lum(z5.M_UV))
        z5.beta_biweight = np.array(np.flipud([-1.36, -1.62, -1.74, -1.85, -1.82, -2.01, -2.12, -2.16, -2.09, -2.27, -2.16]))
        z5.beta_biweight_err = np.array([np.flipud([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]),
                                np.flipud([0.48, 0.11, 0.05, 0.05, 0.04, 0.07, 0.10, 0.09, 0.10, 0.14, 0.17])])
        z5.beta_median = np.array(np.flipud([-1.14, -1.55, -1.77, -1.88, -1.82, -2.00, -2.15, -2.19, -2.02, -2.26, -2.20]))
        z5.beta_inv_var = np.array(np.flipud([-0.72, -1.44, -1.61, -1.75, -1.82, -1.99, -2.08, -2.08, -2.02, -2.33, -2.34]))
        z5.num_sources = np.array(np.flipud([12, 35, 83, 134, 150, 72, 38, 58, 38, 31, 26]))

        z6 = SimpleNamespace()
        z6.M_UV = np.array(np.flipud([-21.75, -21.25, -20.75, -20.25, -19.75, -19.25, -18.75, -18.25, -17.75, -17.00]))
        z6.log10L = np.log10(M_to_lum(z6.M_UV))
        z6.beta_biweight = np.array(np.flipud([-1.55, -1.58, -1.74, -1.90, -1.90, -2.22, -2.26, -2.19, -2.40, -2.24]))
        z6.beta_biweight_err = np.array([np.flipud([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]),
                                np.flipud([0.17, 0.10, 0.1, 0.09, 0.13, 0.18, 0.14, 0.22, 0.30, 0.2])])
        z6.beta_median = np.array(np.flipud([-1.53, -1.55, -1.73, -1.88, -1.93, -2.17, -2.23, -2.12, -2.45, -2.25]))
        z6.beta_inv_var = np.array(np.flipud([-1.59, -1.60, -1.71, -1.81, -1.91, -2.24, -2.19, -2.07, -2.28, -2.19]))
        z6.num_sources = np.array(np.flipud([6, 10, 23, 53, 37, 12, 17, 11, 17, 25]))

        z7 = SimpleNamespace()
        z7.M_UV = np.array(np.flipud([-21.25, -19.95, -18.65, -17.35]))
        z7.log10L = np.log10(M_to_lum(z7.M_UV))
        z7.beta_biweight = np.array(np.flipud([-1.75, -1.89, -2.30, -2.42]))
        z7.beta_biweight_err = np.array([np.flipud([0.13, 0.13, 0.13, 0.13]), np.flipud([0.18, 0.13, 0.18, 0.28])])
        z7.beta_median = np.array(np.flipud([-1.74, -1.88, -2.66, -2.15]))
        z7.beta_inv_var = np.array(np.flipud([-1.67, -1.86, -2.39, -2.39]))
        z7.num_sources = np.array(np.flipud([26, 102, 43, 13]))

        z8 = SimpleNamespace()
        z8.M_UV = np.array(np.flipud([-19.95, -18.65]))
        z8.log10L = np.log10(M_to_lum(z8.M_UV))
        z8.beta_biweight = np.array(np.flipud([-2.30, -1.41]))
        z8.beta_biweight_err = np.array([np.flipud([0.27, 0.27]), np.flipud([0.01, 0.60])])
        z8.beta_median = np.array(np.flipud([-2.30, -1.51]))
        z8.beta_inv_var = np.array(np.flipud([-2.30, -1.88]))
        z8.num_sources = np.array(np.flipud([2, 4]))

        z9 = SimpleNamespace()
        z9.M_UV = np.array(np.flipud([-18.5]))
        z9.log10L = np.log10(M_to_lum(z9.M_UV))
        z9.beta_biweight = np.array(np.flipud([-2.06]))
        z9.beta_biweight_err = np.array([np.flipud([0.27]), np.flipud([0.51])])
        z9.beta_median = np.array(np.flipud([-1.82]))
        z9.beta_inv_var = np.array(np.flipud([-2.07]))
        z9.num_sources = np.array(np.flipud([2]))

        self.redshift = [4, 5, 6, 7, 8, 9]  # Array of redshifts
        self.data = {'z4': z4, 'z5': z5, 'z6': z6, 'z7': z7, 'z8': z8, 'z9': z9}

        super().__init__()




import matplotlib.pyplot as plt

def beta_evo_plot_singlez(z, log10L, beta_model, model):
    zp = beta_model.zp
    a1 = zp['a1'][0] * (z - 6) + zp['a1'][1]
    a2 = zp['a2'][0] * (z - 6) + zp['a2'][1]
    b1 = zp['b1'][0] * (z - 6) + zp['b1'][1]
    b2 = zp['b2'][0] * (z - 6) + zp['b2'][1]

    lptag = False
    lp = 0

    if 'z'+str(z) in beta_model.lp:
        lp = beta_model.lp['z' + str(z)]
        lptag = True

    plt.errorbar(beta_model.data['z' + str(z)].log10L, beta_model.data['z' + str(z)].beta_biweight,
                 beta_model.data['z' + str(z)].beta_biweight_err[1], fmt='x', label='data at z=' + str(z))
    plt.plot(log10L, model(log10L, a1, a2, b1, b2), label='z-evo model')

    if lptag:
        plt.plot(log10L, model(log10L, *lp), label='fit at z=' + str(z))

    plt.ylim(-2.6, -1.)
    plt.xlabel(r'$\rm log_{10}(L/erg \: s^{-1})$')
    plt.ylabel(r'$\rm \beta$')
    plt.legend(loc='best')


def beta_evo_plot(z, log10L, beta_model, model, dataz=False, cmap=False, cmap_range=False, data_cmap=False, data_marker='x', print_fit_at_dataz=False, legend=True, xlims=False, ylims=[-2.6, -1.], colorbar=True, use_flare_style = True):
    """
    :param z: array of redshifts for which model is drawn (should be numpy array).
    :param log10L: array of log10 luminosity (should be numpy array).
    :param beta_model: the initialised best fit beta model, e.g. beta_model = beta_fitter.Bouwens2014(beta_fitter.piecewise).
    :param model: the beta(L) evolution model, e.g. betale.piecewise.
    :param dataz: a list of redshifts for datapoints to be plotted.
    :param cmap: the model colormap (default is plasma).
    :param cmap_range: a tuple of (vmin, vmax) for colormap normalization.
    :param data_cmap: the data colormap (default is viridis).
    :param data_marker: the marker for datapoints str (default is 'x')
    :param print_fit_at_dataz: boolean, if True: plots dashed lines for the best fit at data z. (default is False)
    :param legend: boolean, if True: legend is shown for data and best fits (default is True)
    :param xlims: False or tuple for x limits for plotting (default is False)
    :param ylims: tuple for y limits for plotting (default is (-2.6, -1), doesn't take bool)
    :param colorbar: boolean, if True: draws colourmap on the right side of plot (default is True).
    :return: matplotlip.pyplot figure.
    """

    if use_flare_style:
        import flare.plt 

    fig = plt.figure(figsize=(5, 4))

    left = 0.15
    bottom = 0.15
    height = 0.8
    width = 0.7

    ax = fig.add_axes((left, bottom, width, height))
    ax_scale = fig.add_axes((left + width, bottom, 0.03, height))


    if cmap:
        cmap = cmap
    else:
        cmap = mpl.cm.plasma

    if data_cmap:
        data_cmap = data_cmap
    else:
        data_cmap = mpl.cm.plasma

    data_norm = mpl.colors.Normalize(vmin=4, vmax=9)    # Future improvement: include a handle for this

    if cmap_range:
        c_vmin = cmap_range[0]
        c_vmax = cmap_range[1]
    else:
        if type(z) == int or type(z) == float:
            # could do something more clever and aesthetically pleasing here
            c_vmin = 4
            c_vmax = 15
        else:
            c_vmin=z[0]
            c_vmax=z[-1]

    norm = mpl.colors.Normalize(vmin=c_vmin, vmax=c_vmax)

    zp = beta_model.zp
    a1 = zp['a1'][0] * (z - 6) + zp['a1'][1]
    a2 = zp['a2'][0] * (z - 6) + zp['a2'][1]
    b1 = zp['b1'][0] * (z - 6) + zp['b1'][1]
    b2 = zp['b2'][0] * (z - 6) + zp['b2'][1]

    for i, zz in enumerate(z):
        ax.plot(log10L, model(log10L, a1[i], a2[i], b1[i], b2[i]), c=cmap(norm(zz)), alpha=0.7, zorder=0)


    if type(dataz) == int or type(dataz) == float:
        try:
            if f'z{dataz}' in beta_model.lp and print_fit_at_dataz==True:
                lp = beta_model.lp[f'z{dataz}']
                ax.plot(log10L, model(log10L, *lp), 'k--', label=f'best fit at z = {dataz}')

            ax.errorbar(beta_model.data[f'z{dataz}'].log10L, beta_model.data[f'z{dataz}'].beta_biweight,
                 beta_model.data[f'z{dataz}'].beta_biweight_err[1], fmt=data_marker, c='k', label=f'{beta_model.ref}, z = {dataz}')
        except:
            print(f'No available data for z = {dataz}')

    else:
        if dataz:
            for z_ in dataz:
                if f'z{z_}' in beta_model.lp and print_fit_at_dataz==True:
                    lp = beta_model.lp[f'z{z_}']
                    ax.plot(log10L, model(log10L, *lp), ls='--', c = data_cmap(data_norm(z_)), label=f'best fit at z = {z_}')

                ax.errorbar(beta_model.data[f'z{z_}'].log10L, beta_model.data[f'z{z_}'].beta_biweight,
                             beta_model.data[f'z{z_}'].beta_biweight_err[1], fmt=data_marker, c = data_cmap(data_norm(z_)), label=f'{beta_model.ref}, z = {z_}')


    ax.set_ylim(*ylims)
    if xlims:
        ax.set_xlim(*xlims)
    ax.set_xlabel(r'$\rm log_{10}[L_{FUV}\;/\;erg \; s^{-1}]$')
    ax.set_ylabel(r'$\rm \beta$')

    if legend:
        ax.legend(loc='best', fontsize='small', frameon=False)

    if colorbar:
        cbar = cmap(np.arange(cmap.N)[::-1]).reshape(cmap.N, 1, 4)
        ax_scale.set_ylabel(r'$\rm z$', fontsize=10)
        ax_scale.imshow(cbar, extent=[0, 1, c_vmin, c_vmax], aspect='auto')
        ax_scale.yaxis.tick_right()
        ax_scale.yaxis.set_label_position("right")
        ax_scale.set_xticks([])

    return fig, ax, ax_scale


class analyse():

    def __init__(self, samples, parameters = False, truth = False):

        self.samples = samples
        self.truth = truth

        if not parameters:
            self.parameters = self.samples.keys()
        else:
            self.parameters = parameters

    def P(self):

        for k,v in self.samples.items():

            if not self.truth:
                print(f'{k}: {np.percentile(v, 16.):.2f} {np.median(v):.2f} {np.percentile(v, 84):.2f}')
            else:
                print(f'{k}: {np.percentile(v, 16.):.2f} {np.median(v):.2f} {np.percentile(v, 84):.2f} | {self.truth[k]}')


    def corner(self, filename = 'corner.pdf'):

        Samples = np.array([self.samples[k] for k in self.parameters]).T

        range = []
        for k in self.parameters:
            med = np.median(self.samples[k])
            c68 = np.percentile(self.samples[k], 84) - np.percentile(self.samples[k], 16)
            range.append([med - c68*4, med + c68*4])

        if not self.truth:
            figure = corner.corner(Samples, labels = self.parameters)
        else:
            figure = corner.corner(Samples, labels = self.parameters, truths = [self.truth[k] for k in self.parameters], range = range)



        figure.savefig(filename)
