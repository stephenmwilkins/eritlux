





import numpy as np


sn_cut = 7
l68_cut = 6


def apply_selection(self):

    # --- apply additional selection criteria to catalogue

    detected = self.o['observed/detected'][:]
    sn = self.o['observed/detection/sn'][:]
    l68 = self.o['observed/eazy/l68'][:]

    selected = (detected==True)&(sn>sn_cut)&(l68>l68_cut)

    self.o['observed/selected'] = selected
