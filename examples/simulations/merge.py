

import FLARE.io.hdf5 as flare_hdf5



output_dir = '/Users/stephenwilkins/Dropbox/Research/data/eritlux'

output_filename_pattern = '10nJy_beta_cSersic_idealisedimagePSF_eazy'
N_sim = 10000
N = 100

# output_filename_pattern = 'beta_idealised_EAZY'
# N_sim = 10000
# N = 100

flare_hdf5.merge([f'{output_dir}/{output_filename_pattern}_{N_sim}_{i+1}' for i in range(N)], f'{output_dir}/{output_filename_pattern}')
