

import sys
import os

import numpy as np
import FLARE.surveys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import eritlux.simulations.simulations



survey_id = 'SimpleHubble'
field_id = '10nJy'
sed_model = 'beta'
morph_model = 'cSersic'
phot_model = 'idealisedimage'
pz_model = 'eazy'

output_filename = '_'.join([survey_id, field_id, sed_model, morph_model, phot_model, pz_model])



args = sys.argv

field = FLARE.surveys.surveys[survey_id].fields[field_id]
detection_filters = [f'Hubble.WFC3.{f}' for f in ['f105w', 'f125w', 'f160w']] # --- should hard-code this into the survey module


if len(args) == 1:
    N = 10
    np.random.seed(42)
    run_id = 0
    output_dir = 'output'
    output_filename += f'_{N}'

elif len(args) == 2:
    N = int(args[1])
    np.random.seed(42)
    run_id = 0
    output_dir = 'output'
    output_filename += f'_{N}'

elif len(args) == 3:
    N = int(args[1])
    i = int(args[2])
    run_id = i
    np.random.seed(i)
    output_dir = 'output'
    output_filename += f'_{N}_{i}'

elif len(args) == 4:
    N = int(args[1])
    i = int(args[2])
    run_id = i
    np.random.seed(i)
    output_dir = args[3]
    output_filename += f'_{N}_{i}'

else:
    print('ERROR: too many arguments')


print(output_filename)


sim = eritlux.simulations.simulations.Simulation(morph_model = morph_model, sed_model = sed_model, run_id = run_id)
sim.n(N)

if sed_model == 'beta':
    sim.intrinsic_beta(field.filters) # produce intrinsic photometry
else:
    print('WARNING: SED model not implemented')

if morph_model == 'simple':
    sim.photo_idealised_image(field) # produce intrinsic photometry
else:
    if phot_model == 'idealisedimage':
        sim.photo_idealised_image(field, detection_filters = detection_filters, include_psf = False) # produce observed photometry
    elif phot_model == 'idealisedimagePSF':
        sim.photo_idealised_image(field, detection_filters = detection_filters, include_psf = True) # produce observed photometry
    else:
        print('WARNING: phot model not implemented')

if pz_model == 'eazy':
    sim.pz_eazy() # measure photometric redshifts
else:
    print('WARNING: pz model not implemented')

sim.export_to_HDF5(output_dir, output_filename)
