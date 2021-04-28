

import sys
import os

import numpy as np
import FLARE.surveys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import eritlux.simulations.simulations





np.random.seed(42)

args = sys.argv
print(args)

model = 'beta_idealisedimage_EAZY'

if len(args) == 1:
    N = 10
    output_dir = 'output'
    output_filename = f'{model}_{N}'

elif len(args) == 2:
    N = int(args[1])
    output_dir = 'output'
    output_filename = f'{model}_{N}'

elif len(args) == 3:
    N = int(args[1])
    i = int(args[2])
    output_dir = 'output'
    output_filename = f'{model}_{N}_{i}'

elif len(args) == 4:
    N = int(args[1])
    i = int(args[2])
    output_dir = args[3]
    output_filename = f'{model}_{N}_{i}'

else:
    print('ERROR: too many arguments')

survey_id, field_id = 'SimpleHubble', '10nJy'
field = FLARE.surveys.surveys[survey_id].fields[field_id]

detection_filters = [f'Hubble.WFC3.{f}' for f in ['f105w', 'f125w', 'f160w']]



sim = eritlux.simulations.simulations.Simulation(profile_model = 'cSersic', sed_model = 'beta', output_filename = output_filename)
sim.n(N)
sim.intrinsic_beta(field.filters) # produce intrinsic photometry
sim.photo_idealised_image(field, detection_filters = detection_filters) # produce observed photometry
sim.pz_eazy() # measure photometric redshifts
sim.export_to_HDF5(output_dir, output_filename)