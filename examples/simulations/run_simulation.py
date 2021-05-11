

import sys
import os
import yaml
import numpy as np

import time

import FLARE.surveys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import eritlux.simulations.simulations


starting_time = time.time()

run = True
verbose = False

args = sys.argv

if len(args) == 1:
    scenario = 'default'
    N = 10
    np.random.seed(42)
    run_id = 0
    output_dir = 'output'
    output_suffix = f'{N}'

elif len(args) == 2:
    scenario = sys.argv[1]
    N = 10
    np.random.seed(42)
    run_id = 0
    output_dir = 'output'
    output_suffix = f'{N}'

elif len(args) == 3:
    scenario = sys.argv[1]
    N = int(args[2])
    np.random.seed(42)
    run_id = 0
    output_dir = 'output'
    output_suffix = f'{N}'

elif len(args) == 4:
    scenario = sys.argv[1]
    N = int(args[2])
    i = int(args[3])
    run_id = i
    np.random.seed(i)
    output_dir = 'output'
    output_suffix = f'{N}_{i}'

elif len(args) == 5:
    scenario = sys.argv[1]
    N = int(args[2])
    i = int(args[3])
    run_id = i
    np.random.seed(i)
    output_dir = args[4]
    output_suffix = f'{N}_{i}'

else:
    print('ERROR: too many arguments')


p = yaml.load(open(f'{scenario}.yaml','r'), Loader=yaml.FullLoader)

survey_id = p['survey_id']
field_id = p['field_id']
sed_model = p['sed_model']
morph_model = p['morph_model']
phot_model = p['phot_model']
pz_model = p['pz_model']
detection_filters = p['detection_filters']

print(detection_filters)

field = FLARE.surveys.surveys[survey_id].fields[field_id]


output_filename = '_'.join([survey_id, field_id, sed_model, morph_model, phot_model, pz_model, output_suffix])

print(output_filename)
print(field.depths)


if run:

    sim = eritlux.simulations.simulations.Simulation(morph_model = morph_model, sed_model = sed_model, run_id = run_id)
    if verbose: print(f'initialised: {np.round(time.time()-starting_time,1)}')
    sim.n(N)
    if verbose: print(f'input samples generated: {np.round(time.time()-starting_time,1)}')

    if sed_model == 'beta':
        sim.intrinsic_beta(field.filters) # produce intrinsic photometry
    else:
        print('WARNING: SED model not implemented')
    if verbose: print(f'fluxes calculated: {np.round(time.time()-starting_time,1)}')

    if morph_model == 'simple':
        sim.photo_idealised_image(field) # produce intrinsic photometry
    else:
        if phot_model == 'idealisedimage':
            sim.photo_idealisedimage(field, detection_filters = detection_filters, include_psf = False, verbose = verbose) # produce observed photometry
        elif phot_model == 'idealisedimagePSF':
            sim.photo_idealisedimage(field, detection_filters = detection_filters, include_psf = True, verbose = verbose) # produce observed photometry
        elif phot_model == 'realimage':
            sim.photo_realimage(field, detection_filters = detection_filters, include_psf = False, verbose = verbose) # produce observed photometry
        elif phot_model == 'realimagePSF':
            sim.photo_realimage(field, detection_filters = detection_filters, include_psf = True, verbose = verbose) # produce observed photometry
        else:
            print('WARNING: phot model not implemented')

    if pz_model == 'eazy':
        sim.pz_eazy() # measure photometric redshifts
    else:
        print('WARNING: pz model not implemented')

    sim.export_to_HDF5(output_dir, output_filename)
