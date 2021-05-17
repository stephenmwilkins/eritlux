
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import eritlux.simulations.imagesim as imagesim

import flare.surveys
import flare.plots.image

survey_id = 'XDF' # the XDF (updated HUDF)
field_id = 'dXDF' # deepest sub-region of XDF (defined by a mask)

# --- get field info object. This contains the filters, depths, image location etc. (if real image)
field = flare.surveys.surveys[survey_id].fields[field_id]

# --- select a filter (or loop over all filters)
filter = 'Hubble.WFC3.f160w'

print(f'depth of {filter}: {field.depths[filter]} nJy')


# --- initialise ImageCreator object
image_creator = imagesim.Idealised(filter, field)


# --- create an Image object with the required size
width_pixels = 101
img = image_creator.create_image(width_pixels)


# --- show the current science image. At the moment this is identical to the background image
plt.imshow(img.sci)
plt.show()


# --- as an example make a Sersic profile
width_arcsec = width_pixels*img.pixel_scale
r_e_arcsec = 1
n = 1
ellip = 0.5
theta = 45.
source = imagesim.sersic(width_arcsec, width_pixels, r_e_arcsec, n, ellip, theta) # this is normalised to 1

plt.imshow(source)
plt.show()


# --- define total flux of the source
flux_nJy = 1000. #nJy
flux_es = flux_nJy * img.nJy_to_es

img.sci += flux_es * source

# --- show the new science image
plt.imshow(img.sci)
plt.show()


# --- create nice S/N map
flare.plots.image.make_significance_plot(img)
