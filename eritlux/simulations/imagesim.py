

import numpy as np

from astropy.modeling.models import Sersic2D
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import photutils

import FLARE.observatories


from . import psf



kron_params = [2.5, 1]




class empty(): pass

# --- return a normalised Sersic profile image for a given


class Image ():

    def __init__(self):
        pass

    def sn(self):
        return self.sci/self.noise


class CreateBackground():

    def __init__(self, filter, field, verbose = False):

        self.filter = filter
        self.pixel_scale = field.pixel_scale
        self.nJy_to_es = FLARE.observatories.get_nJy_to_es(filter) # conversion from nJy to e/s

        self.aperture = empty()
        self.aperture.depth = field.depths[filter]

        if field.depth_aperture_radius_pixel:
            self.aperture.radius = field.depth_aperture_radius_pixel
        else:
            self.aperture.radius = field.depth_aperture_radius_arcsec/field.pixel_scale

        self.aperture.significance = field.depth_aperture_significance
        self.aperture.noise = self.aperture.depth/self.aperture.significance # nJy
        self.aperture.background = self.aperture.noise**2
        self.aperture.area = np.pi * self.aperture.radius**2
        self.aperture.noise_es = self.aperture.noise * self.nJy_to_es # convert from nJy to e/s

        self.pixel = empty()
        self.pixel.background = self.aperture.background/self.aperture.area
        self.pixel.noise = np.sqrt(self.pixel.background) # nJy
        self.pixel.noise_es = self.pixel.noise * self.nJy_to_es # convert from nJy to e/s

        if verbose:
            print('assumed aperture radius: {0:.2f} pix'.format(self.aperture.radius))
            print('noise in aperture: {0:.2f} nJy'.format(self.aperture.noise))
            print('noise in pixel: {0:.2f} nJy'.format(self.pixel.noise))
            print('nJy_to_es: {0}'.format(self.nJy_to_es))

    def create_background_image(self, width_pixels):

        img = Image()
        img.nJy_to_es = self.nJy_to_es
        img.pixel_scale = self.pixel_scale
        img.noise = self.pixel.noise_es * np.ones((width_pixels, width_pixels))
        img.wht = 1./img.noise**2
        img.bkg = self.pixel.noise_es * np.random.randn(*img.noise.shape)
        img.sci = img.bkg.copy()

        return img



def create_PSFs(field, width_pixels):

    PSF_creator = psf.PSFs(field.filters)
    width_arcsec = width_pixels * field.pixel_scale
    PSFs = {}

    for f in field.filters:
        native_pixel_scale = FLARE.observatories.filter_info[f]['pixel_scale']
        xx = yy = np.linspace(-(width_arcsec/native_pixel_scale/2.), (width_arcsec/native_pixel_scale/2.), width_pixels)
        PSFs[f] = PSF_creator[f].f(xx, yy)
        PSFs[f] /= np.sum(PSFs[f])

    return PSFs


def sersic(width_arcsec, width_pixels, r_e_arcsec, n, ellip, theta):

    g = np.linspace(-width_arcsec/2., width_arcsec/2., width_pixels)

    xx, yy = np.meshgrid(g, g)

    mod = Sersic2D(amplitude = 1, r_eff = r_e_arcsec, n = n, x_0 = 0.0, y_0 = 0.0, ellip = ellip, theta = theta)

    img = mod(xx, yy)
    img /= np.sum(img)

    return img



def create_image(BackgroundCreator, field, p, width_pixels = 51, verbose = False, PSFs = None):

    width_arcsec = width_pixels * field.pixel_scale

    img = {f:BackgroundCreator[f].create_background_image(width_pixels) for f in field.filters}

    # --- create profile
    mod = sersic(width_arcsec, width_pixels, p['intrinsic/r_eff_arcsec'], p['intrinsic/n'], p['intrinsic/ellip'], p['intrinsic/theta'])

    for f in field.filters:
        flux_nJy = p[f'intrinsic/flux/{f}']
        if verbose:
            print(flux_nJy, BackgroundCreator[f].aperture.depth)

        img[f].mod_nopsf = mod * flux_nJy * img[f].nJy_to_es

        if PSFs:
            img[f].mod = convolve_fft(img[f].mod_nopsf, PSFs[f])
        else:
            img[f].mod = img[f].mod_nopsf

        img[f].sci += img[f].mod

    return img


def create_detection_image(imgs, detection_filters):

    detection_image = create_stack({f:imgs[f] for f in detection_filters})

    return detection_image


def create_stack(imgs):

    stack = Image()

    first_img = next(iter(imgs.values()))
    shape = first_img.sci.shape

    stack.sci = np.zeros(shape)
    stack.wht = np.zeros(shape)
    stack.pixel_scale = first_img.pixel_scale

    for filter, img in imgs.items():
        stack.sci += img.sci * img.wht
        stack.wht += img.wht

    stack.sci /= stack.wht

    stack.noise = 1./np.sqrt(img.wht)

    return stack



def detect_sources(detection_image, threshold = 2.5, npixels = 5, nlevels = 32):

    segm = photutils.detect_sources(detection_image.sn(), threshold, npixels = npixels)

    if segm:
        detected = True
        segm_deblended = photutils.deblend_sources(detection_image.sn(), segm, npixels = npixels, nlevels = nlevels)
        detection_cat = photutils.SourceCatalog(detection_image.sci, segm_deblended, error = detection_image.noise, kron_params = kron_params)
    else:
        detected = False
        segm_deblended = None
        detection_cat = None

    return detected, detection_cat, segm_deblended


def perform_photometry(detection_cat, segm_deblended, imgs):

    source_cat = {}

    for f, img in imgs.items():

        source_cat[f] = photutils.SourceCatalog(img.sci, segm_deblended, error = img.noise, kron_params = kron_params, detection_cat = detection_cat)

    return source_cat
