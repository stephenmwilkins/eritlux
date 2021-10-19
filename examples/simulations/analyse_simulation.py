
import os
import sys

# Import CMasher to register colormaps
import cmasher as cmr


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import eritlux.simulations.plots as plots

# output_dir = 'output'
output_dir = '/Users/stephenwilkins/Dropbox/Research/data/eritlux'


output_filename = 'SimpleHubble_10nJy_beta_simple_idealised_eazy'
output_filename = 'SimpleHubble_10nJy_beta_cSersic_idealisedimage_eazy'
# output_filename = 'SimpleHubble_10nJy_beta_cSersic_idealisedimagePSF_eazy'

output_filename = 'XDF_dXDF_beta_cSersic_idealisedimagePSF_eazy'
output_filename = 'XDF_dXDF_beta_cSersic_realimagePSF_eazy'







survey_id, field_id, sed_model, morph_model, phot_model, pz_model = output_filename.split('_')[:6]


analyser = plots.Analyser(output_dir, output_filename, save_plots = True, show_plots = False)

# analyser.explore_hdf5()


if phot_model == 'idealised':

    analyser.detection_grid(['intrinsic/z', 'intrinsic/log10L', 'intrinsic/beta'])

    if pz_model is not 'none':
        analyser.make_redshift_plot()

if phot_model in ['idealisedimage','idealisedimagePSF', 'realimage', 'realimagePSF']:

    # analyser.detection_grid_compact(['intrinsic/z', 'intrinsic/log10L', 'intrinsic/beta', 'intrinsic/log10r_eff_kpc'])
    analyser.detection_grid_compact(['intrinsic/z', 'intrinsic/log10L', 'intrinsic/beta', 'intrinsic/log10r_eff_kpc'], selected = True, cmap = cmr.torch  ) # selection grid

    # analyser.detection_grid(['intrinsic/z', 'intrinsic/log10L', 'intrinsic/beta', 'intrinsic/log10r_eff_kpc'])
    # analyser.detection_grid(['intrinsic/z', 'intrinsic/log10L', 'intrinsic/beta', 'intrinsic/log10r_eff_kpc'], selected = True) # selection grid
    # analyser.make_photometry_plot()
    # analyser.make_colour_plot()
    # analyser.make_size_plot()

    # if pz_model is not 'none':
        # analyser.make_redshift_plot()
        # analyser.pz_grid(['intrinsic/z', 'intrinsic/log10L', 'intrinsic/beta', 'intrinsic/log10r_eff_kpc'])
