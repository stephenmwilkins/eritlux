
import os
import sys



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import eritlux.simulations.plots as plots


output_dir = '/Users/stephenwilkins/Dropbox/Research/data/eritlux'

output_filename = 'SimpleHubble_10nJy_beta_simple_idealised_eazy'
output_filename = 'SimpleHubble_10nJy_beta_cSersic_idealisedimage_eazy'
# output_filename = 'SimpleHubble_10nJy_beta_cSersic_idealisedimagePSF_eazy'

output_filename = 'XDF_dXDF_beta_cSersic_realimagePSF_eazy'

survey_id, field_id, sed_model, morph_model, phot_model, pz_model, = output_filename.split('_')


analyser = plots.analyse(output_dir, output_filename, save_plots = True)

analyser.explore_hdf5()


if phot_model == 'idealised':

    analyser.detection_grid(['intrinsic/z', 'intrinsic/log10L', 'intrinsic/beta'])
    analyser.make_redshift_plot()

if phot_model in ['idealisedimage','idealisedimagePSF', 'realimage', 'realimagePSF']:

    analyser.detection_grid(['intrinsic/z', 'intrinsic/log10L', 'intrinsic/beta', 'intrinsic/log10r_eff_kpc'])
    analyser.make_photometry_plot()
    analyser.make_colour_plot()
    analyser.make_size_plot()
    analyser.make_redshift_plot()
    analyser.pz_grid(['intrinsic/z', 'intrinsic/log10L', 'intrinsic/beta', 'intrinsic/log10r_eff_kpc'])
