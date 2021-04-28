
import os
import sys



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import eritlux.simulations.plots as plots



output_dir, output_filename = 'output', 'idealised_10k'



analyser = plots.analyse(output_dir, output_filename)

# analyser.detection_plot('intrinsic/z', 'intrinsic/log10L')
analyser.detection_grid(['intrinsic/z', 'intrinsic/log10L', 'intrinsic/beta'])
