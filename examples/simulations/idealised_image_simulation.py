

import sys
import os

import numpy as np
import FLARE.surveys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import eritlux.simulations.simulations

np.random.seed(42)

class empty(): pass

N = 10 # --- number of galaxies

# --- define survey and field. This in turn defines the depth, area, and filters but this could be done manually.

survey_id, field_id = 'SimpleHubble', '10nJy'
field = FLARE.surveys.surveys[survey_id].fields[field_id]

detection_filters = [f'Hubble.WFC3.{f}' for f in ['f105w', 'f125w', 'f160w']]

output_dir, output_filename = 'output', 'idealised_image'

sim = eritlux.simulations.simulations.Simulation(profile_model = 'cSersic', sed_model = 'beta')
sim.n(N)
sim.intrinsic_beta(field.filters) # produce intrinsic photometry
sim.photo_idealised_image(field, detection_filters = detection_filters) # produce observed photometry
sim.pz_eazy() # measure photometric redshifts
sim.export_to_HDF5(output_dir, output_filename)
