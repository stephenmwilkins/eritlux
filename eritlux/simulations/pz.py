


import numpy as np

import FLARE
import FLARE.filters
import FLARE.obs.EAZY as flareazy

def idealised(sim, pz_error = lambda x: 0.1, best = True):

    # return idealised photometry based simply on the depth

    detected = self.o['observed/detected']

    sn = self.o['observed/sn'][detected]

    pz = self.o['intrinsic/input/z'][detected] + pz_error(sn)*np.random.randn(self.N)

    self.o['observed/idealpz'] = pz

    if best:
        self.o[f'observed/pz'] = self.o['observed/idealpz']



def eazy(self, eazy_working_dir = '.', path_to_EAZY = f'{FLARE.FLARE_dir}/software/eazy-photoz', best = True):

    id = self.run_id

    # return idealised photometry based simply on the depth

    N = self.N
    filters = self.filters # filter list
    F = FLARE.filters.add_filters(filters) # make FLARE filter object

    # --- run EAZY
    flareazy.eazy(ID=id).run(self.o, F, detected = self.o['observed/detected'], path = lambda f: f'observed/{f}')

    zout, POFZ = flareazy.read_EAZY_output(f'{eazy_working_dir}/EAZY/outputs/{id}')

    for k in ['z_a','z_m1','chi_a', 'l68', 'u68', 'l95', 'u95', 'l99', 'u99', 'q_z', 'z_peak']:
        self.o[f'observed/eazy/{k}'] = np.zeros(self.N)
        self.o[f'observed/eazy/{k}'][self.o['observed/detected']] = zout[k]

    if best:
        self.o[f'observed/pz'] = self.o[f'observed/eazy/z_a']
