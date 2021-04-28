


import numpy as np

import FLARE
import FLARE.filters
import FLARE.obs.EAZY as flareazy

def idealised(sim, pz_error = lambda x: 0.1):

    # return idealised photometry based simply on the depth

    detected = self.o['observed/detected']

    sn = self.o['observed/sn'][detected]

    pz = self.o['intrinsic/input/z'][detected] + pz_error(sn)*np.random.randn(self.N)

    self.o['observed/pz/z_a'] = pz
    self.o['observed/pz/z_m1'] = pz



def eazy(self, eazy_working_dir = '.', path_to_EAZY = f'{FLARE.FLARE_dir}/software/eazy-photoz'):

    id = 'test'

    # return idealised photometry based simply on the depth

    N = self.N
    filters = self.filters # filter list
    F = FLARE.filters.add_filters(filters) # make FLARE filter object

    # --- run EAZY
    flareazy.eazy(ID=id).run(self.o, F)

    zout, POFZ = flareazy.read_EAZY_output(f'{eazy_working_dir}/EAZY/outputs/{id}')

    for k in ['z_a','z_m1']:
        self.o[f'observed/pz/{k}'] = zout[k]
