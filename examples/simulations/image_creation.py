
import os
import sys

import numpy as np

import matplotlib.pyplot as plt

import FLARE.surveys
import FLARE.observatories



import pysep.plots.image






sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import eritlux.simulations.simulations
import eritlux.simulations.psf
import eritlux.simulations.imagesim as imagesim

# np.random.seed(38)


# --- create a single galaxy

survey_id, field_id = 'SimpleHubble', '10nJy'
field = FLARE.surveys.surveys[survey_id].fields[field_id]

detection_filters = [f'Hubble.WFC3.{f}' for f in ['f105w', 'f125w', 'f160w']]

output_dir, output_filename = 'output', 'image_test'

sim = eritlux.simulations.simulations.Simulation(profile_model = 'cSersic', sed_model = 'beta')
sim.n(1)
sim.intrinsic_beta(field.filters) # produce intrinsic photometry
# sim.photo_idealised(field) # produce observed photometry
# sim.pz_eazy() # measure photometric redshifts
# sim.export_to_HDF5(output_dir, output_filename)




BackgroundCreator = {f: imagesim.CreateBackground(f, field) for f in field.filters}


width_pixels = 51

PSFs = eritlux.simulations.imagesim.create_PSFs(field, width_pixels)



imgs = imagesim.create_image(BackgroundCreator, field, sim.i(), width_pixels = width_pixels, verbose = True, PSFs = PSFs)

img = imgs[field.filters[-1]]

plt.imshow(img.mod_nopsf)
plt.show()


plt.imshow(img.mod)
plt.show()











#
#
# imgs = imagesim.create_image(BackgroundCreator, field, p, width_pixels = width_pixels, verbose = True)
#
# # pysep.plots.image.make_flux_plots(imgs)
# # pysep.plots.image.make_significance_plots(imgs)
#
# detection_image = imagesim.create_detection_image(imgs, detection_filters)
#
# pysep.plots.image.make_significance_plot(detection_image)
#
# detected, detection_cat, segm_deblended = imagesim.detect_sources(detection_image)
#
# # pysep.plots.image.make_segm_plot(segm_deblended)
#
# source_cats = imagesim.perform_photometry(detection_cat, segm_deblended, imgs)
#
# for f in field.filters:
#
#     print(f, p[f'intrinsic/fluxes/{f}'], source_cats[f].kron_flux[0]/imgs[f].nJy_to_es, source_cats[f].segment_flux[0]/imgs[f].nJy_to_es, source_cats[f].kron_flux[0]/source_cats[f].kron_fluxerr[0])
