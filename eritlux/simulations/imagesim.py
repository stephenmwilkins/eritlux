

import numpy as np

from astropy.modeling.models import Sersic2D
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import photutils
from astropy.io import fits


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



class Idealised():

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
            print('-'*20)
            print(filter)
            print('assumed aperture radius: {0:.2f} pix'.format(self.aperture.radius))
            print('noise in aperture: {0:.2f} nJy'.format(self.aperture.noise))
            print('noise in pixel: {0:.2f} nJy'.format(self.pixel.noise))
            print('nJy_to_es: {0}'.format(self.nJy_to_es))

    def create_image(self, width_pixels, xy = None):

        img = Image()
        img.nJy_to_es = self.nJy_to_es
        img.pixel_scale = self.pixel_scale
        img.noise = self.pixel.noise_es * np.ones((width_pixels, width_pixels))
        img.wht = 1./img.noise**2
        img.bkg = self.pixel.noise_es * np.random.randn(*img.noise.shape)
        img.sci = img.bkg.copy()

        return img


    def get_random_location(self):

        """this doesn't do anything for the Idealised image creator but simplifies code later"""

        return None












class Real():

    def __init__(self, filter, field, verbose = False, sci_suffix = 'sci', wht_suffix = 'wht'):

        self.field = field
        self.verbose = verbose

        if field.mask_file:
            self.mask = fits.getdata(f'{field.data_dir}/{field.mask_file}')
        else:
            self.mask = None

        self.img = ImageFromFile(field.data_dir, filter, mask = self.mask, pixel_scale = field.pixel_scale, verbose = verbose, sci_suffix = sci_suffix, wht_suffix = wht_suffix)



    def make_cutout(self, x, y, width):

        """extract cut out"""

        sci = np.zeros((width, width))
        wht = np.zeros((width, width))

        x = int(np.round(x, 0))
        y = int(np.round(y, 0))

        xmin = x - width // 2
        xmax = x + width // 2
        ymin = y - width // 2
        ymax = y + width // 2

        xstart = 0
        ystart = 0
        xend = width
        yend = width

        if xmin < 0:
            xstart = -xmin
            xmin = 0
        if ymin < 0:
            ystart = -ymin
            ymin = 0
        if xmax > self.img.sci.shape[0]:
            xend -= xmax - self.img.sci.shape[0]
            xmax = self.img.sci.shape[0]
        if ymax > self.img.sci.shape[1]:
            yend -= ymax - self.img.sci.shape[1]
            ymax = self.img.sci.shape[1]

        if (width % 2) != 0:
            xmax += 1
            ymax += 1

        sci[xstart:xend,ystart:yend] = self.img.sci[xmin:xmax,ymin:ymax]
        wht[xstart:xend,ystart:yend] = self.img.wht[xmin:xmax,ymin:ymax]

        return ImageFromArrays(sci, wht, self.img.pixel_scale, zeropoint = self.img.zeropoint, nJy_to_es = self.img.nJy_to_es, verbose = self.verbose)


    def get_random_location(self):

        """get (single) random location on the image"""

        pos = np.random.choice(self.img.sci.count())
        return np.take((~self.img.sci.mask).nonzero(), pos, axis=1)


    def create_image(self, width_pixels, xy = None):

        """ cut-out an image at a random location """

        return self.make_cutout(*xy, width_pixels)








class ImageFromFile(Image):

    def __init__(self, data_dir, filter, mask = None, pixel_scale = 0.06, verbose = False, sci_suffix = 'sci', wht_suffix = 'wht'):

        """generate instance of image class from file"""

        if verbose:
            print('-'*40)
            print(f'filter: {filter}')
            print(f'reading image from: {data_dir}')


        f = filter.split('.')[-1]

        self.verbose = verbose

        self.filter = filter
        self.pixel_scale = pixel_scale

        self.sci = fits.getdata(f'{data_dir}/{f}_{sci_suffix}.fits')
        self.wht = fits.getdata(f'{data_dir}/{f}_{wht_suffix}.fits')

        if filter in FLARE.observatories.filter_info.keys():
            self.zeropoint = FLARE.observatories.filter_info[filter]['zeropoint'] # AB magnitude zeropoint
            self.nJy_to_es = FLARE.observatories.filter_info[filter]['nJy_to_es'] # conversion from nJy to e/s
        else:
            self.zeropoint = self.nJy_to_es = None

        self.mask = mask

        if type(mask) == np.ndarray:
            self.mask = mask
        else:
            self.mask = (self.wht == 0)

        self.sci = np.ma.masked_array(self.sci, mask = self.mask)
        self.wht = np.ma.masked_array(self.wht, mask = self.mask)

        if verbose:
            print(f'shape: ', self.sci.shape)

        self.noise = 1./np.sqrt(self.wht)
        # self.sig = self.sci/self.noise



class ImageFromArrays(Image):

    def __init__(self, sci, wht, pixel_scale, zeropoint = False, nJy_to_es = False,  verbose = False):

        """generate instance of image class from cutout"""

        self.verbose = verbose

        self.pixel_scale = pixel_scale
        self.zeropoint = zeropoint # AB magnitude zeropoint
        self.nJy_to_es = nJy_to_es # conversion from nJy to e/s

        self.sci = sci
        self.wht = wht
        self.noise = 1./np.sqrt(self.wht)
        # self.sig = self.sci/self.noise


































def create_PSFs(field, width_pixels, final_filter = True):

    if final_filter:
        filter = field.filters[-1]
        PSF = create_PSF(filter, field, width_pixels)
        PSFs = {f:PSF for f in field.filters}
    else:
        PSFs = {f:create_PSF(f, field, width_pixels) for f in field.filters}

    return PSFs


def create_PSF(filter, field, width_pixels):

    PSF_creator = psf.PSF(filter)
    width_arcsec = width_pixels * field.pixel_scale

    native_pixel_scale = FLARE.observatories.filter_info[filter]['pixel_scale']
    xx = yy = np.linspace(-(width_arcsec/native_pixel_scale/2.), (width_arcsec/native_pixel_scale/2.), width_pixels)
    PSF = PSF_creator.f(xx, yy)
    PSF /= np.sum(PSF)

    return PSF





def sersic(width_arcsec, width_pixels, r_e_arcsec, n, ellip, theta):

    g = np.linspace(-width_arcsec/2., width_arcsec/2., width_pixels)

    xx, yy = np.meshgrid(g, g)

    mod = Sersic2D(amplitude = 1, r_eff = r_e_arcsec, n = n, x_0 = 0.0, y_0 = 0.0, ellip = ellip, theta = theta)

    img = mod(xx, yy)
    img /= np.sum(img)

    return img



def create_image(image_creator, field, p, width_pixels = 51, verbose = False, PSFs = None):

    width_arcsec = width_pixels * field.pixel_scale

    xy = next(iter(image_creator.values())).get_random_location()
    if verbose: print(xy)

    img = {f:image_creator[f].create_image(width_pixels, xy) for f in field.filters}

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
