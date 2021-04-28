
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import FLARE.plt as fplt


labels = {}
labels['intrinsic/z'] = 'z'
labels['intrinsic/log10L'] = r'log_{10}(L/erg\ s^{-1}\ Hz^{-1})'
labels['intrinsic/beta'] = r'\beta'
labels['intrinsic/log10r_eff_kpc'] = r'log_{10}(r_{eff}/kpc)'


range = {}
range['intrinsic/z'] = [6,10]
range['intrinsic/log10L'] = [28,30]
range['intrinsic/beta'] = [-3,0]
range['intrinsic/log10r_eff_kpc'] = [-0.3,0.0]

bins = 50



class analyse:

    def __init__(self, output_dir, output_filename, show_plots = True, save_plots = False):


        self.output_dir = output_dir
        self.output_filename = output_filename
        self.show_plots = show_plots
        self.save_plots = save_plots

        self.hf = h5py.File(f'{output_dir}/{output_filename}.h5', 'r')
        self.detected = self.hf['observed/detected'][:].astype('bool')

    # def explore_hdf5(self):
    #
    #     # --- show everything in the hdf5 file
    #     self.hf.visit(lambda x: print(x))

    def explore_hdf5(self):

        def get_name_shape(name, item):
            shape = ''
            if hasattr(item, 'value'):
                shape = item.shape
            print(name, shape)

        self.hf.visititems(get_name_shape)


    def default_plot(self):

        fig = plt.figure(figsize = (4,4))

        left  = 0.15
        bottom = 0.15
        height = 0.8
        width = 0.8

        ax = fig.add_axes((left, bottom, width, height))

        return fig, ax


    def detection_panel(self, ax, x, y, s = None, cmap = 'inferno'):

        X = self.hf[x]
        Y = self.hf[y]

        all, bin_edges_x, bin_edges_y = np.histogram2d(X[:],Y[:],bins=(bins,bins), range = [range[x],range[y]])

        d, bin_edges_x, bin_edges_y = np.histogram2d(X[self.detected],Y[self.detected],bins=(bins,bins), range = [range[x],range[y]])

        ax.imshow((d/all).T, origin='lower', extent = [*range[x], *range[y]], aspect = 'auto', cmap = cmap)

        return ax


    def detection_plot(self, x, y, s = None):

        fig, ax = self.default_plot()

        ax = self.detection_panel(ax, x, y, s = s)

        ax.set_xlabel(rf'$\rm {labels[x]}$')
        ax.set_ylabel(rf'$\rm {labels[y]}$')

        if self.save_plots: fig.savefig(f'{self.output_dir}/{self.output_filename}_detection.pdf')
        if self.show_plots: plt.show()


    def detection_grid(self, properties, s = None, save = False):

        N = len(properties) - 1

        fig, axes = plt.subplots(N, N, figsize = (6,6))
        plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.02, hspace=0.02)

        # axes = axes.T

        for i,x in enumerate(properties[:-1]):
            for j,y in enumerate(properties[1:][::-1]):

                jj = N-1-j
                ii = i

                ax = axes[jj, ii]

                if j+i<N:
                    ax = self.detection_panel(ax, x, y, s = s)
                    # ax.text(0.5, 0.5, f'x{i}-y{j}', transform = ax.transAxes)
                else:
                    ax.set_axis_off()
                    # ax.text(0.5, 0.5, f'x{i}-y{j}', transform = ax.transAxes)

                if i == 0: # first column
                    ax.set_ylabel(rf'$\rm {labels[y]}$')
                else:
                    ax.yaxis.set_ticklabels([])

                if j == 0: # first row
                    ax.set_xlabel(rf'$\rm {labels[x]}$')
                else:
                    ax.xaxis.set_ticklabels([])

        if self.save_plots: fig.savefig(f'{self.output_dir}/{self.output_filename}_detection_grid.pdf')
        if self.show_plots: plt.show()

    # --- photometric redshift

    # def make_redshift_plot(self):
    #
    #     fig, ax = default_plot()
    #
    #     cmap = cm.viridis
    #     norm = mpl.colors.Normalize(vmin=0, vmax=3)
    #
    #     ax.scatter(hf['intrinsic/z'][detected], hf['observed/pz/z_a'][detected], c = cmap(norm(np.log10(hf['observed/sn'][detected]))), s=5)
    #     ax.plot([3,13],[3,13],c='k', alpha=0.1)
    #
    #     ax.set_xlim([3,13])
    #     ax.set_ylim([3,13])
    #
    #     ax.set_xlabel(r'$\rm z$')
    #     ax.set_ylabel(r'$\rm z_{PZ}$')
    #
    #     # ax.legend(loc = 'lower left', fontsize = 8)
    #
    #     fig.savefig(f'{self.output_dir}/{self.output_filename}_zpz.pdf')
    #     fig.clf()


    def make_photometry_plot(self, cmap = 'viridis'):

        fluxes_input = self.hf['intrinsic/flux/']
        fluxes_observed = self.hf['observed/flux/']

        filters = list(fluxes_input.keys())


        fig, axes = fig, axes = plt.subplots(1, len(filters), figsize = (3*len(filters),3))
        plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.02, hspace=0.02)


        f_range = [-1.0, 3.0] # log10(nJy)
        R_range = [-0.5, 0.5]

        for f, ax in zip(filters, axes):

            f_input = np.log10(fluxes_input[f][self.detected])
            f_obseved = np.log10(fluxes_observed[f][self.detected])

            R = f_obseved - f_input

            print(np.min(f_input), np.max(f_input))
            print(np.min(R), np.max(R))

            H, bin_edges_x, bin_edges_y = np.histogram2d(f_input, R,bins=(bins*2,bins*2), range = [f_range,R_range])

            ax.imshow(H.T, origin='lower', extent = [*f_range, *R_range], aspect = 'auto', cmap = cmap)

            ax.set_xlim(f_range)
            ax.set_ylim(R_range)


        # ax.set_xlabel(r'$\rm z$')
        # ax.set_ylabel(r'$\rm z_{PZ}$')

        # ax.legend(loc = 'lower left', fontsize = 8)

        if self.show_plots: plt.show()
        if self.save_plots: fig.savefig(f'{self.output_dir}/{self.output_filename}_photometry.pdf')







    def make_size_scatter_plot(self):

        fig, ax = self.default_plot()

        cmap = cm.viridis
        norm = mpl.colors.Normalize(vmin=0, vmax=3)

        r_input = self.hf['intrinsic/r_eff_arcsec'][self.detected]
        r_obs = self.hf['observed/r_eff_kron_arcsec'][self.detected]
        log10sn = np.log10(self.hf['observed/sn'][self.detected])

        ax.scatter(r_input, r_obs, s, c = cmap(norm(log10sn)), s=5)

        r = range['intrinsic/log10r_eff_kpc']

        # ax.plot(r,r,c='k', alpha=0.1)
        #
        # ax.set_xlim(r)
        # ax.set_ylim(r)

        # ax.set_xlabel(r'$\rm z$')
        # ax.set_ylabel(r'$\rm z_{PZ}$')

        # ax.legend(loc = 'lower left', fontsize = 8)

        if self.show_plots: plt.show()
        if self.save_plots: fig.savefig(f'{self.output_dir}/{self.output_filename}_size.pdf')


    def make_size_plot(self):

        fig, ax = self.default_plot()

        cmap = cm.inferno

        r_input = np.log10(self.hf['intrinsic/r_eff_arcsec'][self.detected])
        r_obs = np.log10(self.hf['observed/r_eff_kron_arcsec'][self.detected])
        log10sn = np.log10(self.hf['observed/sn'][self.detected])

        r_range = [np.min(r_input), np.max(r_input)]



        H, bin_edges_x, bin_edges_y = np.histogram2d(r_input,r_obs,bins=(bins,bins), range = [r_range,r_range])

        ax.imshow(H.T, origin='lower', extent = [*r_range, *r_range], aspect = 'auto', cmap = cmap)

        ax.plot(r_range,r_range,c='w', alpha=0.5)

        ax.set_xlabel(r'$\rm log_{10}(r_{e}^{input}/arcsec)$')
        ax.set_ylabel(r'$\rm log_{10}(r_{e}^{obs}/arcsec)$')

        # ax.legend(loc = 'lower left', fontsize = 8)

        if show_plots: plt.show()
        if self.save_plots: fig.savefig(f'{self.output_dir}/{self.output_filename}_size.pdf')
