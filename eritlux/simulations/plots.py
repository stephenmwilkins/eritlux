
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import FLARE.plt as fplt


labels = {}
labels['intrinsic/z'] = 'z'
labels['intrinsic/log10L'] = r'log_{10}(L/erg s^{-1} Hz^{-1})'
labels['intrinsic/beta'] = r'\beta'

range = {}
range['intrinsic/z'] = [6,10]
range['intrinsic/log10L'] = [28,30]
range['intrinsic/beta'] = [-3,1]

bins = 20



class analyse:

    def __init__(self, output_dir, output_filename):

        self.output_dir = output_dir
        self.output_filename = output_filename
        self.hf = h5py.File(f'{output_dir}/{output_filename}.h5', 'r')
        self.detected = self.hf['observed/detected'][:]

    def visit(self):

        # --- show everything in the hdf5 file
        hf.visit(lambda x: print(x))


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

        fig.savefig(f'{self.output_dir}/{self.output_filename}_detection.pdf')
        plt.show()


    def detection_grid(self, properties, s = None, save = False):

        N = len(properties) - 1

        fig, axes = plt.subplots(N, N, figsize = (6,6))
        plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.0, hspace=0.0)

        # axes = axes.T

        for i,x in enumerate(properties[:-1]):
            for j,y in enumerate(properties[1:][::-1]):

                jj = N-1-j
                ii = i

                ax = axes[jj, ii]

                if j+i<2*(N-1):
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


        plt.show()
        fig.savefig(f'{self.output_dir}/{self.output_filename}_detection_grid.pdf')

    # --- photometric redshift

    def make_redshift_plot(self):

        fig, ax = default_plot()

        cmap = cm.viridis
        norm = mpl.colors.Normalize(vmin=0, vmax=3)

        ax.scatter(hf['intrinsic/z'][detected], hf['observed/pz/z_a'], c = cmap(norm(np.log10(hf['observed/sn'][detected]))), s=5)
        ax.plot([3,13],[3,13],c='k', alpha=0.1)

        ax.set_xlim([3,13])
        ax.set_ylim([3,13])

        ax.set_xlabel(r'$\rm z$')
        ax.set_ylabel(r'$\rm z_{PZ}$')

        # ax.legend(loc = 'lower left', fontsize = 8)

        fig.savefig(f'{self.output_dir}/{self.output_filename}_zpz.pdf')
        fig.clf()
