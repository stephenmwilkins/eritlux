

import FLARE.io.hdf5 as flare_hdf5


flare_hdf5.merge([f'output/beta_idealisedimage_EAZY_100_{i+1}.h5' for i in range(2)], f'output/beta_idealisedimage_EAZY_merged.h5')
