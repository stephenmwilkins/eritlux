
import os
import sys

# Import CMasher to register colormaps
import cmasher as cmr


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import eritlux.simulations.plots as plots

# output_dir = 'output'
output_dir = '/Users/stephenwilkins/Dropbox/Research/data/eritlux'

output_filename = 'XDF_dXDF_beta_cSersic_idealisedimagePSF_eazy'
output_filename = 'XDF_dXDF_beta_cSersic_realimagePSF_eazy'



survey_id, field_id, sed_model, morph_model, phot_model, pz_model = output_filename.split('_')[:6]

analyser = plots.Analyser(output_dir, output_filename, save_plots = True, show_plots = False)

# analyser.transfer()
analyser.explore_hdf5()
