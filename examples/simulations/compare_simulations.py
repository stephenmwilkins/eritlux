
import os
import sys



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import eritlux.simulations.plots as plots


output_dir = '/Users/stephenwilkins/Dropbox/Research/data/eritlux'

output_filename_1 = 'XDF_dXDF_beta_cSersic_realimagePSF_none'
output_filename_2 = 'XDF_dXDF_beta_cSersic_idealisedimagePSF_none'


analyser = plots.ComparisonAnalyser(output_dir, output_filename_1, output_dir, output_filename_2, save_plots = True)

analyser.detection_ratio_grid(['intrinsic/z', 'intrinsic/log10L', 'intrinsic/beta', 'intrinsic/log10r_eff_kpc'])
