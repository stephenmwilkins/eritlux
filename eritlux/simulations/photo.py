


import numpy as np

from . import imagesim


import pysep.plots.image


def idealised(self, field, detection_filter = None, detection_threshold = 10, size_error = None):

    # return idealised photometry based simply on the depth

    # --- define the "observed" flux error in each band. In this case it is simply the depth.
    flux_err = {f: np.ones(self.N)*(field.depths[f]) for f in field.filters}

    # --- define the "observed" flux in each band. In this case it is simply the intrinsic flux + gaussian noise.
    flux = {f: self.o[f'intrinsic/flux/{f}'] + field.depths[f] * np.random.randn(self.N) for f in field.filters}

    # ---
    if not detection_filter:
        detection_filter = field.filters[-1]

    sn = flux[detection_filter]/flux_err[detection_filter]
    self.o['observed/sn'] = sn

    detected = sn > detection_threshold
    self.o['observed/detected'] = detected

    for k, v in flux.items():
        self.o[f'observed/flux/{k}'] = v

    for k, v in flux_err.items():
        self.o[f'observed/flux_err/{k}'] = v




def idealisedimage(self, field, detection_filters = None, width_pixels = 51, include_psf = False, verbose = False):

    # --- create idealised image creator
    image_creator = {f:imagesim.Idealised(f, field) for f in field.filters}

    # --- now call image
    image(self, field, image_creator, detection_filters = detection_filters, width_pixels = width_pixels, include_psf = include_psf, verbose = verbose)


def realimage(self, field, detection_filters = None, width_pixels = 51, include_psf = False, verbose = False):

    # --- create real image creator
    image_creator = {f:imagesim.Real(f, field) for f in field.filters}

    # --- now call image
    image(self, field, image_creator, detection_filters = detection_filters, width_pixels = width_pixels, include_psf = include_psf, verbose = verbose)




def image(self, field, image_creator, detection_filters = None, width_pixels = 51, include_psf = False, verbose = False):

    # --- create output quantities

    self.o[f'observed/detected'] = np.zeros(self.N, dtype=bool)

    quantities = ['r_segment_arcsec','r_eff_kron_arcsec']

    for q in ['sn'] + quantities:
        self.o[f'observed/{q}'] = np.zeros(self.N)

    for f in field.filters:
        for q in ['kron_flux','kron_fluxerr','segment_flux','segment_fluxerr']:
            self.o[f'observed/{q}/{f}'] = np.zeros(self.N)



    # --- create PSF
    if include_psf:
        PSFs = imagesim.create_PSFs(field, width_pixels, final_filter = True)
    else:
        PSFs = None


    for i in range(self.N):

        if verbose: print('-'*10, i)

        ob = self.i(i=i)

        # --- create image object
        imgs = imagesim.create_image(image_creator, field, ob, width_pixels = 51, PSFs = PSFs)

        # --- create detection image

        detection_image = imagesim.create_detection_image(imgs, detection_filters)

        if verbose: pysep.plots.image.make_significance_plot(detection_image)

        # --- detect sources
        detected, detection_cat, segm_deblended = imagesim.detect_sources(detection_image)

        if detected:

            if verbose: pysep.plots.image.make_segm_plot(segm_deblended)

            x, y = detection_cat.xcentroid, detection_cat.ycentroid

            # --- determine distance from the centre of the image
            r = np.sqrt((x-(width_pixels-1)/2)**2 + (y-(width_pixels-1)/2)**2)

            # --- determine closest object to the centre of the image
            j = np.where(r==np.min(r))[0]

            if verbose: print(r)

            # --- only count as detected if within 3 pixels of the centre. 3 is somewhat arbitrary here.
            if r[j]<3:

                self.o[f'observed/detected'][i] = True
                self.o[f'observed/sn'][i] = detection_cat.kron_flux[j]/detection_cat.kron_fluxerr[j]

                self.o[f'observed/r_segment_arcsec'][i] = detection_cat.equivalent_radius[j].value * field.pixel_scale
                self.o[f'observed/r_eff_kron_arcsec'][i] = detection_cat.fluxfrac_radius(0.5)[j] * field.pixel_scale

                # --- photometer sources
                source_cat = imagesim.perform_photometry(detection_cat, segm_deblended, imgs)

                for f in field.filters:
                    self.o[f'observed/kron_flux/{f}'][i] = source_cat[f].kron_flux[j]/imgs[f].nJy_to_es
                    self.o[f'observed/kron_fluxerr/{f}'][i] = source_cat[f].kron_fluxerr[j]/imgs[f].nJy_to_es
                    self.o[f'observed/segment_flux/{f}'][i] = source_cat[f].segment_flux[j]/imgs[f].nJy_to_es
                    self.o[f'observed/segment_fluxerr/{f}'][i] = source_cat[f].segment_fluxerr[j]/imgs[f].nJy_to_es

            else:

                detected = False # --- if objects, but none within 3 pixels of the centre then set to undetected


        if verbose: print(detected)




    # --- set kron photometry as default photometry
    for f in field.filters:
        self.o[f'observed/flux/{f}'] = self.o[f'observed/kron_flux/{f}']
        self.o[f'observed/flux_err/{f}'] = self.o[f'observed/kron_fluxerr/{f}']
