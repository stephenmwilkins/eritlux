


import numpy as np

from . import imagesim


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






def idealised_image(self, field, detection_filters = None, width_pixels = 51):

    # --- initialise background maker
    BackgroundCreator = {f:imagesim.CreateBackground(f, field) for f in field.filters}


    self.o[f'observed/detected'] = np.zeros(self.N, dtype=bool)


    quantities = ['r_segment_arcsec','r_eff_kron_arcsec']

    for q in ['sn'] + quantities:
        self.o[f'observed/{q}'] = np.zeros(self.N)

    for f in field.filters:
        for q in ['kron_flux','kron_fluxerr','segment_flux','segment_fluxerr']:
            self.o[f'observed/{q}/{f}'] = np.zeros(self.N)


    for i in range(self.N):

        print(i)

        ob = self.i(i=i)

        # --- create image object
        imgs = imagesim.create_image(BackgroundCreator, field, ob, width_pixels = 51)

        # --- create detection image

        detection_image = imagesim.create_detection_image(imgs, detection_filters)

        # --- detect sources
        detected, detection_cat, segm_deblended = imagesim.detect_sources(detection_image)

        if detected:

            self.o[f'observed/detected'][i] = True
            self.o[f'observed/sn'][i] = detection_cat.kron_flux[0]/detection_cat.kron_fluxerr[0]

            self.o[f'observed/r_segment_arcsec'][i] = detection_cat.equivalent_radius[0].value * field.pixel_scale
            self.o[f'observed/r_eff_kron_arcsec'][i] = detection_cat.fluxfrac_radius(0.5)[0] * field.pixel_scale

            # --- photometer sources
            source_cat = imagesim.perform_photometry(detection_cat, segm_deblended, imgs)

            for f in field.filters:
                self.o[f'observed/kron_flux/{f}'][i] = source_cat[f].kron_flux[0]
                self.o[f'observed/kron_fluxerr/{f}'][i] = source_cat[f].kron_fluxerr[0]
                self.o[f'observed/segment_flux/{f}'][i] = source_cat[f].segment_flux[0]
                self.o[f'observed/segment_fluxerr/{f}'][i] = source_cat[f].segment_fluxerr[0]


    # for q in quantities:
    #     self.o[f'observed/{q}'] = self.o[f'observed/{q}'][self.o[f'observed/detected']]

    for f in field.filters:
        # for q in ['kron_flux','kron_fluxerr','segment_flux','segment_fluxerr']:
        #     self.o[f'observed/{q}/{f}'] =  self.o[f'observed/{q}/{f}'][self.o[f'observed/detected']]

        self.o[f'observed/flux/{f}'] = self.o[f'observed/kron_flux/{f}']
        self.o[f'observed/flux_err/{f}'] = self.o[f'observed/kron_fluxerr/{f}']











def image():

    # use real images and run source extraction and photometry on them

    print('WARNING: Not yet implemented')
