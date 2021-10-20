
import os
import numpy as np
import scipy.stats
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import flare.plt as fplt


labels = {}
labels['intrinsic/z'] = 'z'
labels['intrinsic/log10L'] = r'log_{10}(L/erg\ s^{-1}\ Hz^{-1})'
labels['intrinsic/beta'] = r'\beta'
labels['intrinsic/log10r_eff_kpc'] = r'log_{10}(r_{eff}/kpc)'


range = {}
# range['intrinsic/z'] = [6,10]
# range['intrinsic/log10L'] = [27,30]
# range['intrinsic/beta'] = [-3,1]
# range['intrinsic/log10r_eff_kpc'] = [-0.5,0.5]
range['intrinsic/z'] = [6.01,9.99]
range['intrinsic/log10L'] = [27.01,29.99]
range['intrinsic/beta'] = [-2.99,0.99]
range['intrinsic/log10r_eff_kpc'] = [-0.499,0.499]


bins = 50



class Analyser:

    def __init__(self, output_dir, output_filename, show_plots = True, save_plots = False):


        self.output_dir = output_dir
        self.output_filename = output_filename
        self.show_plots = show_plots
        self.save_plots = save_plots

        survey_id, field_id, sed_model, morph_model, phot_model, pz_model = output_filename.split('_')[:6]

        self.survey_id = survey_id
        self.field_id = field_id
        self.sed_model = sed_model
        self.morph_model = morph_model
        self.phot_model = phot_model
        self.pz_model = pz_model

        self.hf = h5py.File(f'{output_dir}/{output_filename}.h5', 'r')
        self.detected = self.hf['observed/detected'][:].astype('bool')
        print(f'fraction detected: {len(self.detected[self.detected])/len(self.detected)}')

        if 'observed/selected' in self.hf:
            self.selected = self.hf['observed/selected'][:].astype('bool')
            print(f'fraction selected: {len(self.selected[self.selected])/len(self.selected)}')

        # print(self.hf.attrs.keys())

        self.plot_dir = f'{output_dir}/{output_filename}'

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)



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

        d, bin_edges_x, bin_edges_y = np.histogram2d(X[s],Y[s],bins=(bins,bins), range = [range[x],range[y]])

        ax.imshow((d/all).T, origin='lower', extent = [*range[x], *range[y]], aspect = 'auto', cmap = cmap)

        return ax


    def detection_plot(self, x, y, s = None):

        fig, ax = self.default_plot()

        ax = self.detection_panel(ax, x, y, s = s)

        ax.set_xlabel(rf'$\rm {labels[x]}$')
        ax.set_ylabel(rf'$\rm {labels[y]}$')

        if self.save_plots: fig.savefig(f'{self.plot_dir}/detection.pdf')
        if self.show_plots: plt.show()


    def detection_grid(self, properties, selected = False, save = False, cmap = 'inferno'):

        if selected:
            s = self.selected
        else:
            s = self.detected

        N = len(properties)

        fig, axes = plt.subplots(N, N, figsize = (6,6))
        plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.02, hspace=0.02)

        # axes = axes.T

        for i in np.arange(N):
            for j in np.arange(N):
                axes[i, j].set_axis_off()

        for i,x in enumerate(properties):
            for j,y in enumerate(properties[1:][::-1]):

                jj = N-1-j
                ii = i

                ax = axes[jj, ii]

                if j+i<(N-1):
                    ax.set_axis_on()
                    ax = self.detection_panel(ax, x, y, s = s, cmap = cmap)

                if i == 0: # first column
                    ax.set_ylabel(rf'$\rm {labels[y]}$')
                else:
                    ax.yaxis.set_ticklabels([])

                if j == 0: # first row
                    ax.set_xlabel(rf'$\rm {labels[x]}$')
                else:
                    ax.xaxis.set_ticklabels([])


                # ax.text(0.5, 0.5, f'x{i}-y{j}', transform = ax.transAxes)


            # --- marginalised
            ax = axes[ii, ii]
            ax.set_axis_on()

            X = self.hf[x]

            H_all, bin_edges = np.histogram(X, bins = bins, range = range[x])
            H_detected, bin_edges = np.histogram(X[s], bins = bins, range = range[x])

            bin_centres = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])*0.5

            ax.plot(bin_centres, H_detected/H_all, c='0.5')

            # --- if selected also plot the detected fraction
            if selected:

                X = self.hf[x]

                H_all, bin_edges = np.histogram(X, bins = bins, range = range[x])
                H_detected, bin_edges = np.histogram(X[self.detected], bins = bins, range = range[x])

                bin_centres = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])*0.5

                ax.plot(bin_centres, H_detected/H_all, c='0.3', ls = ':')




            ax.set_ylim([-0.05,1.05])




            # ax.text(0.5, 0.5, f'x{i}', transform = ax.transAxes)


        # --- add colourbar
        cax = fig.add_axes([0.4, 0.87, 0.5, 0.03])
        norm = mpl.colors.Normalize(0,1)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')

        if selected:
            cax.set_xlabel(rf'$\rm selected\ fraction$')
        else:
            cax.set_xlabel(rf'$\rm detected\ fraction$')


        if self.save_plots:
            if selected:
                fig.savefig(f'{self.plot_dir}/selection_grid.pdf')
            else:
                fig.savefig(f'{self.plot_dir}/detection_grid.pdf')

        if self.show_plots: plt.show()


    def detection_grid_compact(self, properties, selected = False, cmap = 'inferno'):

        # a compact version of the above

        if selected:
            s = self.selected
        else:
            s = self.detected

        N = len(properties)

        fig, axes = plt.subplots(N-1, N-1, figsize = (4,4))
        plt.subplots_adjust(left=0.15, top=0.95, bottom=0.15, right=0.95, wspace=0.02, hspace=0.02)

        # axes = axes.T

        for i in np.arange(N-1):
            for j in np.arange(N-1):
                axes[i, j].set_axis_off()

        for i,x in enumerate(properties[:-1]):
            for j,y in enumerate(properties[1:][::-1]):

                jj = N-2-j
                ii = i

                ax = axes[jj, ii]

                if j+i<(N-1):
                    ax.set_axis_on()
                    ax = self.detection_panel(ax, x, y, s = s, cmap = cmap)

                if i == 0: # first column
                    ax.set_ylabel(rf'$\rm {labels[y]}$')
                else:
                    ax.yaxis.set_ticklabels([])

                if j == 0: # first row
                    ax.set_xlabel(rf'$\rm {labels[x]}$')
                else:
                    ax.xaxis.set_ticklabels([])


                # ax.text(0.5, 0.5, f'x{i}-y{j}', transform = ax.transAxes)




        # --- add colourbar
        cax = fig.add_axes([0.45, 0.87, 0.5, 0.03])
        norm = mpl.colors.Normalize(0,1)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')

        if selected:
            cax.set_xlabel(rf'$\rm selected\ fraction$')
        else:
            cax.set_xlabel(rf'$\rm detected\ fraction$')

        if self.save_plots:
            if selected:
                fig.savefig(f'{self.plot_dir}/selection_grid_compact.pdf')
            else:
                fig.savefig(f'{self.plot_dir}/detection_grid_compact.pdf')

        if self.show_plots: plt.show()




    def pz_panel(self, ax, x, y, s = None, statistic = 'median', cmap = 'coolwarm', vmnmx = [-0.1, 0.1]):

        X = self.hf[x][self.detected]
        Y = self.hf[y][self.detected]
        V = (self.hf['intrinsic/z'][self.detected] - self.hf['observed/pz'][self.detected])/self.hf['intrinsic/z'][self.detected] # the data on which the statistic is calculated

        stat, bin_edges_x, bin_edges_y, _ = scipy.stats.binned_statistic_2d(X, Y, V, statistic = statistic, bins = (bins,bins), range = [range[x],range[y]])

        ax.imshow(stat.T, origin='lower', extent = [*range[x], *range[y]], aspect = 'auto', cmap = cmap, vmin = vmnmx[0], vmax = vmnmx[1])

        return ax


    def pz_grid(self, properties, s = None, save = False):

        N = len(properties) - 1


        for statistic, vmnmx, cmap in zip(['median','std'], [[-0.1, 0.1],[0.0, 0.2]], ['coolwarm', 'viridis']):

            fig, axes = plt.subplots(N, N, figsize = (6,6))
            plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.02, hspace=0.02)

            # axes = axes.T

            for i,x in enumerate(properties[:-1]):
                for j,y in enumerate(properties[1:][::-1]):

                    jj = N-1-j
                    ii = i

                    ax = axes[jj, ii]

                    if j+i<N:
                        ax = self.pz_panel(ax, x, y, s = s, statistic = statistic, cmap = cmap, vmnmx = vmnmx)
                    else:
                        ax.set_axis_off()

                    if i == 0: # first column
                        ax.set_ylabel(rf'$\rm {labels[y]}$')
                    else:
                        ax.yaxis.set_ticklabels([])

                    if j == 0: # first row
                        ax.set_xlabel(rf'$\rm {labels[x]}$')
                    else:
                        ax.xaxis.set_ticklabels([])


            # --- add colourbar
            cax = fig.add_axes([0.4, 0.87, 0.5, 0.03])
            norm = mpl.colors.Normalize(*vmnmx)
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
            cax.set_xlabel(rf'$\rm {statistic}([z-z_{{PZ}}]/z)$')


            if self.save_plots: fig.savefig(f'{self.plot_dir}/pz_grid_{statistic}.pdf')
            if self.show_plots: plt.show()


    # --- photometric redshift

    def make_redshift_plot(self, s=None, cmap = 'inferno'):

        fig, ax = self.default_plot()

        X = self.hf['intrinsic/z'][self.detected]
        Y = (self.hf['intrinsic/z'][self.detected] - self.hf[f'observed/pz'][self.detected])/self.hf['intrinsic/z'][self.detected]

        x_range = range['intrinsic/z']
        y_range = [-0.3, 0.3]

        H, bin_edges_x, bin_edges_y = np.histogram2d(X, Y, bins=(bins*2,bins*2), range = [x_range, y_range])

        ax.imshow(H.T, origin='lower', extent = [*x_range, *y_range], aspect = 'auto', cmap = cmap)

        ax.axhline(0.0, c='w', alpha=0.2, lw=1)


        median, bin_edges, _ = scipy.stats.binned_statistic(X, Y, statistic='median', bins=100)
        P16, bin_edges, _ = scipy.stats.binned_statistic(X, Y, statistic = lambda x: np.percentile(x, 16.), bins=100)
        P84, bin_edges, _ = scipy.stats.binned_statistic(X, Y, statistic = lambda x: np.percentile(x, 84.), bins=100)

        bin_centres = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])/2

        ax.fill_between(bin_centres, P16, P84, color='w', alpha = 0.2, label = r'$\rm P_{84}-P_{16}$')
        ax.plot(bin_centres, median, c='w', label = r'$\rm median $')

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        ax.set_xlabel(r'$\rm z$')
        ax.set_ylabel(r'$\rm (z-z_{PZ})/z$')

        l = ax.legend(fontsize = 10)
        for text in l.get_texts():
            text.set_color('w')

        if self.save_plots: fig.savefig(f'{self.plot_dir}/zpz.pdf')
        if self.show_plots: plt.show()


    def make_photometry_plot(self, cmap = 'viridis'):

        phot_types = ['kron_flux', 'segment_flux']

        for phot_type in phot_types:

            fluxes_input = self.hf['intrinsic/flux/']
            fluxes_observed = self.hf[f'observed/']

            filters = list(fluxes_input.keys())

            fig, axes = fig, axes = plt.subplots(1, len(filters), figsize = (3*len(filters),3), sharey = True)
            plt.subplots_adjust(left=0.05, top=0.9, bottom=0.15, right=0.95, wspace=0.0, hspace=0.0)

            f_range = [0.01, 2.99] # log10(nJy)
            R_range = [-0.25, 0.25]

            for f, ax in zip(filters, axes):

                f_input = np.log10(fluxes_input[f][self.detected])
                f_obseved = np.log10(fluxes_observed[f'{f}/{phot_type}'][self.detected])

                R = f_obseved - f_input

                H, bin_edges_x, bin_edges_y = np.histogram2d(f_input, R,bins=(bins*2,bins*2), range = [f_range,R_range])

                ax.imshow(H.T, origin='lower', extent = [*f_range, *R_range], aspect = 'auto', cmap = cmap)

                ax.axhline(0.0, c='w', alpha=0.2, lw=1)

                ax.set_xlim(f_range)
                ax.set_ylim(R_range)

                ax.set_xlabel(rf'$\rm \log_{{10}}(f_{{ {f.split(".")[-1]} }}/nJy)$')

            axes[0].set_ylabel(r'$\rm  \log_{10}(f^{\ observed}/f^{\ input})$')

            if self.show_plots: plt.show()
            if self.save_plots: fig.savefig(f'{self.plot_dir}/photometry_{phot_type}.pdf')


    def make_colour_plot(self, cmap = 'viridis', s = None):

        phot_types = ['kron_flux', 'segment_flux']

        for phot_type in phot_types:

            fluxes_input = self.hf['intrinsic/flux/']
            fluxes_observed = self.hf[f'observed/']

            filters = list(fluxes_input.keys())

            fig, axes = fig, axes = plt.subplots(1, len(filters)-1, figsize = (3*(len(filters)-1),3), sharey = True)
            plt.subplots_adjust(left=0.05, top=0.9, bottom=0.15, right=0.95, wspace=0.0, hspace=0.0)

            x_range = [-2, 2] # log10(nJy)
            y_range = [-0.25, 0.25]

            for f1, f2, ax in zip(filters[:-1],filters[1:], axes):

                input = -2.5*np.log10(fluxes_input[f2][self.detected]/fluxes_input[f1][self.detected])
                observed = -2.5*np.log10(fluxes_observed[f'{f2}/{phot_type}'][self.detected]/fluxes_observed[f'{f1}/{phot_type}'][self.detected])


                X = input
                Y = observed - input

                H, bin_edges_x, bin_edges_y = np.histogram2d(X, Y,bins=(bins*2,bins*2), range = [x_range, y_range])

                ax.imshow(H.T, origin='lower', extent = [*x_range, *y_range], aspect = 'auto', cmap = cmap)

                ax.axhline(0.0, c='w', alpha=0.2, lw=1)

                ax.set_xlim(x_range)
                ax.set_ylim(y_range)

                ax.set_xlabel(rf"$\rm {f2.split('.')[-1]}-{f1.split('.')[-1]}$")

            axes[0].set_ylabel(r'$\rm (A-B)^{\ observed} - A-B)^{\ input})$')

            if self.show_plots: plt.show()
            if self.save_plots: fig.savefig(f'{self.plot_dir}/colour_{phot_type}.pdf')





    def make_size_plot(self, cmap = 'magma'):

        fig, ax = self.default_plot()

        X = np.log10(self.hf['intrinsic/r_eff_arcsec'][self.detected])
        Y = X - np.log10(self.hf['observed/r_eff_kron_arcsec'][self.detected])

        x_range = [np.min(X), np.max(X)]
        y_range = [-0.3, 0.3]

        H, bin_edges_x, bin_edges_y = np.histogram2d(X,Y,bins=(2*bins,2*bins), range = [x_range,y_range])

        ax.imshow(H.T, origin='lower', extent = [*x_range, *y_range], aspect = 'auto', cmap = cmap)

        ax.axhline(0.0, c='w', alpha=0.5)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        ax.set_xlabel(r'$\rm log_{10}(r_{e}^{input}/arcsec)$')
        ax.set_ylabel(r'$\rm log_{10}(r_{e}^{input}/r_{e}^{obs})$')

        if self.show_plots: plt.show()
        if self.save_plots: fig.savefig(f'{self.plot_dir}/size.pdf')


    def make_transfer_plot(self, properties, selected = False, cmap = 'inferno'):

        # a compact version of the above

        if selected:
            s = self.selected
        else:
            s = self.detected

        N = len(properties)

        fig, axes = plt.subplots(N, 1, figsize = (1,N))
        plt.subplots_adjust(left=0.15, top=0.95, bottom=0.15, right=0.95, wspace=0.02, hspace=0.02)



        for x, ax in zip(properties, axes):

            ax.scatter()


            if j == 0: # first row
                ax.set_xlabel(rf'$\rm {labels[x]}$')
            else:
                ax.xaxis.set_ticklabels([])


            # ax.text(0.5, 0.5, f'x{i}-y{j}', transform = ax.transAxes)




        # --- add colourbar
        cax = fig.add_axes([0.45, 0.87, 0.5, 0.03])
        norm = mpl.colors.Normalize(0,1)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')

        if selected:
            cax.set_xlabel(rf'$\rm selected\ fraction$')
        else:
            cax.set_xlabel(rf'$\rm detected\ fraction$')

        if self.save_plots:
            if selected:
                fig.savefig(f'{self.plot_dir}/transfer_plot.pdf')
            else:
                fig.savefig(f'{self.plot_dir}/transfer_plot.pdf')

        if self.show_plots: plt.show()



class ComparisonAnalyser:

    def __init__(self, output_dir_1, output_filename_1, output_dir_2, output_filename_2, show_plots = True, save_plots = False, plot_dir = 'comparison_plots'):


        self.output_dir_1 = output_dir_1
        self.output_filename_1 = output_filename_1
        self.output_dir_2 = output_dir_2
        self.output_filename_2 = output_filename_2

        self.show_plots = show_plots
        self.save_plots = save_plots

        survey_id, field_id, sed_model, morph_model, phot_model, pz_model, = output_filename_1.split('_')

        self.survey_id_1 = survey_id
        self.field_id_1 = field_id
        self.sed_model_1 = sed_model
        self.morph_model_ = morph_model
        self.phot_model_ = phot_model
        self.pz_model_ = pz_model

        self.hf_1 = h5py.File(f'{output_dir_1}/{output_filename_1}.h5', 'r')
        self.detected_1 = self.hf_1['observed/detected'][:].astype('bool')

        survey_id, field_id, sed_model, morph_model, phot_model, pz_model, = output_filename_2.split('_')

        self.survey_id_2 = survey_id
        self.field_id_2 = field_id
        self.sed_model_2 = sed_model
        self.morph_model_2 = morph_model
        self.phot_model_2 = phot_model
        self.pz_model_2 = pz_model

        self.hf_2 = h5py.File(f'{output_dir_2}/{output_filename_2}.h5', 'r')
        self.detected_2 = self.hf_2['observed/detected'][:].astype('bool')

        self.plot_dir = plot_dir

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)


    def default_plot(self):

        fig = plt.figure(figsize = (4,4))

        left  = 0.15
        bottom = 0.15
        height = 0.8
        width = 0.8

        ax = fig.add_axes((left, bottom, width, height))

        return fig, ax


    def detection_ratio_panel(self, ax, x, y, s = None, cmap = 'Spectral'):

        X = self.hf_1[x]
        Y = self.hf_1[y]

        all, bin_edges_x, bin_edges_y = np.histogram2d(X[:],Y[:],bins=(bins,bins), range = [range[x],range[y]])
        d, bin_edges_x, bin_edges_y = np.histogram2d(X[self.detected_1],Y[self.detected_1],bins=(bins,bins), range = [range[x],range[y]])

        d_f_1 = (d/all).T

        X = self.hf_2[x]
        Y = self.hf_2[y]

        all, bin_edges_x, bin_edges_y = np.histogram2d(X[:],Y[:],bins=(bins,bins), range = [range[x],range[y]])
        d, bin_edges_x, bin_edges_y = np.histogram2d(X[self.detected_2],Y[self.detected_2],bins=(bins,bins), range = [range[x],range[y]])

        d_f_2 = (d/all).T

        R = d_f_1-d_f_2

        # v = np.max([np.fabs(np.min(R)), np.max(R)])
        # vmin = -v
        # vmax = v
        vmin = -0.15
        vmax = 0.15

        ax.imshow(R, origin='lower', extent = [*range[x], *range[y]], aspect = 'auto', cmap = cmap, vmin = vmin, vmax = vmax)

        return ax


    def detection_ratio_plot(self, x, y, s = None):

        fig, ax = self.default_plot()

        ax = self.detection_ratio_panel(ax, x, y, s = s)

        ax.set_xlabel(rf'$\rm {labels[x]}$')
        ax.set_ylabel(rf'$\rm {labels[y]}$')

        if self.save_plots: fig.savefig(f'{self.plot_dir}/detection.pdf')
        if self.show_plots: plt.show()





    def detection_ratio_grid(self, properties, s = None, save = False, cmap = 'RdBu'):

        N = len(properties)

        fig, axes = plt.subplots(N, N, figsize = (6,6))
        plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.02, hspace=0.02)

        # axes = axes.T

        for i in np.arange(N):
            for j in np.arange(N):
                axes[i, j].set_axis_off()

        for i,x in enumerate(properties[:-1]):
            for j,y in enumerate(properties[1:][::-1]):

                jj = N-1-j
                ii = i

                ax = axes[jj, ii]

                if j+i<(N-1):
                    ax.set_axis_on()
                    ax = self.detection_ratio_panel(ax, x, y, s = s, cmap = cmap)

                if i == 0: # first column
                    ax.set_ylabel(rf'$\rm {labels[y]}$')
                else:
                    ax.yaxis.set_ticklabels([])

                if j == 0: # first row
                    ax.set_xlabel(rf'$\rm {labels[x]}$')
                else:
                    ax.xaxis.set_ticklabels([])


            # --- marginalised
            ax = axes[ii, ii]
            ax.set_axis_on()

            X = self.hf_1[x]
            H_all, bin_edges = np.histogram(X, bins = bins, range = range[x])
            H_detected, bin_edges = np.histogram(X[self.detected_1], bins = bins, range = range[x])
            bin_centres = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])*0.5
            ax.plot(bin_centres, H_detected/H_all, c='0.5', label = 'A')

            X = self.hf_2[x]
            H_all, bin_edges = np.histogram(X, bins = bins, range = range[x])
            H_detected, bin_edges = np.histogram(X[self.detected_2], bins = bins, range = range[x])
            bin_centres = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])*0.5
            ax.plot(bin_centres, H_detected/H_all, c='0.5', ls = ':', label = 'B')

            ax.legend(fontsize = 6)
            ax.set_ylim([-0.05,1.05])



        # --- add colourbar
        cax = fig.add_axes([0.4, 0.87, 0.5, 0.03])
        norm = mpl.colors.Normalize(-0.15,0.15)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
        cax.set_xlabel(r'$\rm detected_{A}-detected_{B}$')

        fig.text(0.4, 0.78, f'A={self.output_filename_1}', fontsize = 8)
        fig.text(0.4, 0.75, f'B={self.output_filename_2}', fontsize = 8)

        if self.save_plots: fig.savefig(f'{self.plot_dir}/detection_grid.pdf')
        if self.show_plots: plt.show()



    # def detection_ratio_grid(self, properties, s = None, save = False, cmap = 'RdBu'):
    #
    #     N = len(properties) - 1
    #
    #     fig, axes = plt.subplots(N, N, figsize = (6,6))
    #     plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.02, hspace=0.02)
    #
    #     # axes = axes.T
    #
    #     for i,x in enumerate(properties[:-1]):
    #         for j,y in enumerate(properties[1:][::-1]):
    #
    #             jj = N-1-j
    #             ii = i
    #
    #             ax = axes[jj, ii]
    #
    #             if j+i<N:
    #                 ax = self.detection_ratio_panel(ax, x, y, s = s, cmap = cmap)
    #                 # ax.text(0.5, 0.5, f'x{i}-y{j}', transform = ax.transAxes)
    #             else:
    #                 ax.set_axis_off()
    #                 # ax.text(0.5, 0.5, f'x{i}-y{j}', transform = ax.transAxes)
    #
    #             if i == 0: # first column
    #                 ax.set_ylabel(rf'$\rm {labels[y]}$')
    #             else:
    #                 ax.yaxis.set_ticklabels([])
    #
    #             if j == 0: # first row
    #                 ax.set_xlabel(rf'$\rm {labels[x]}$')
    #             else:
    #                 ax.xaxis.set_ticklabels([])
    #
    #
    #     # --- add colourbar
    #     cax = fig.add_axes([0.4, 0.87, 0.5, 0.03])
    #     norm = mpl.colors.Normalize(-1,1)
    #     fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    #     cax.set_xlabel(rf'$\rm detected\ fraction\ difference$')
    #
    #     if self.save_plots: fig.savefig(f'{self.plot_dir}/detection_grid.pdf')
    #     if self.show_plots: plt.show()
