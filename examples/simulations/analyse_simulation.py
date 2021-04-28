
import os
import sys



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import eritlux.simulations.plots as plots



output_dir, output_filename = '/Users/stephenwilkins/Dropbox/Research/data/eritlux', 'beta_idealisedimage_EAZY'



analyser = plots.analyse(output_dir, output_filename)

analyser.explore_hdf5()

print(analyser.detected)

# analyser.detection_plot('intrinsic/z', 'intrinsic/log10L')
# analyser.detection_grid(['intrinsic/z', 'intrinsic/log10L', 'intrinsic/beta'])
analyser.detection_grid(['intrinsic/z', 'intrinsic/log10L', 'intrinsic/beta', 'intrinsic/log10r_eff_kpc'])
