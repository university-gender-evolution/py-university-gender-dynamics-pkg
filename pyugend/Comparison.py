__author__ = 'krishnab'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# CONSTANTS

# line_colors = ['#7fc97f', '#beaed4', '#fdc086','#386cb0','#f0027f','#ffff99']

class Comparison():
    def __init__(self, model_list):
        self.name = 'All Models'
        self.label = 'All Models'
        self.mlist = model_list

    def plot_comparison_spec_parameter(self, parm, title,
                                       xlabel, ylabel, number_of_runs=10):

        ## This function will execute gender proportion comparisons for all models

        ## Color list

        line_colors = ['#7fc97f', '#beaed4', '#fdc086', '#386cb0', '#f0027f',
                       '#ffff99']

        for mod in self.mlist:
            mod.run_multiple(number_of_runs)

        ## Create plot array and execute plot

        for k, v in enumerate(self.mlist):
            plt.plot(range(self.mlist[k].duration),
                     self.mlist[k].mean_matrix[parm],
                     color=line_colors[k], label=self.mlist[k].label,
                     linewidth=2.0)

            plt.plot(range(self.mlist[k].duration),
                     self.mlist[k].mean_matrix[parm])
            plt.fill_between(range(self.mlist[k].duration),
                             self.mlist[k].mean_matrix[
                                 parm] +
                             1.96 * self.mlist[k].std_matrix[
                                 parm],
                             self.mlist[k].mean_matrix[parm] - 1.96 *
                             self.mlist[k].std_matrix[
                                 parm], color=line_colors[k], alpha=0.5)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='upper right', shadow=True)
        plt.show()


    def plot_parameter_sweep_gender_proportion(self, number_of_runs, param,
                                               llim, ulim, number_of_steps):

        ## This function will execute gender proportion comparisons for all models

        ## Color list

        line_colors = ['#7fc97f', '#beaed4', '#fdc086', '#386cb0', '#f0027f',
                       '#ffff99']
        for mod in self.mlist:
            mod.run_parameter_sweep(number_of_runs, param, llim, ulim,
                                    number_of_steps)

        ## Create plot array and execute plot

        for k, v in enumerate(self.mlist):
            plot_array = self.mlist[k].parameter_sweep_array[0]
            plt.plot(plot_array[:, 0], plot_array[:, 1],
                     label=self.mlist[k].label, linewidth=2.0,
                     color=line_colors[k])
            plt.fill_between(plot_array[:, 0], plot_array[:, 1] +
                             1.96 * plot_array[:, 2], plot_array[:, 1] -
                             1.96 * plot_array[:, 2], alpha=0.5,
                             color=line_colors[k], facecolor=line_colors[k])

        plt.title('Parameter Sweep for Gender Proportion over ' + str(
            self.mlist[0].duration) + ' years')
        plt.xlabel(param)
        plt.ylabel('Percentage of the Department that is Women')
        plt.legend(loc='upper right', shadow=True)
        plt.show()

    def plot_parameter_sweep_detail(self,
                                    number_of_runs,
                                    param,
                                    llim,
                                    ulim,
                                    number_of_steps):

        # create color list

        line_colors = ['#7fc97f', '#beaed4', '#fdc086', '#386cb0', '#f0027f',
                       '#ffff99']
        plot_titles = ['extra', 'Female Level 1', 'Female level 2', 'Female '
                                                                    'level 3',
                       'Male level 1', 'Male level 2', 'Male level 3']
        plot_y_axis_titles = ['extra', 'Number of Females', 'Number of Females',
                              'Number of Females', 'Number of Males',
                              'Number of Males', 'Number of Males']

        # run all models and generate result matrices. I run each model so I can guarantee that they all have the same duration.

        for mod in self.mlist:
            mod.run_parameter_sweep(number_of_runs, param, llim, ulim,
                                    number_of_steps)

        f, axarr = plt.subplots(nrows=2, ncols=3)
        f.suptitle("Parameter sweep")

        for k, v in enumerate(self.mlist):
            array_list = self.mlist[k].parameter_sweep_array

            axarr[0, 0].plot(array_list[1][:, 0], array_list[1][:, 1],
                             label=self.mlist[k].label, linewidth=2.0,
                             color=line_colors[k])
            axarr[0, 0].set_title('Female level 1')
            axarr[0, 0].set_xlabel(array_list[7])
            axarr[0, 0].set_ylabel('Number of Females')
            axarr[0, 0].fill_between(array_list[1][:, 0], array_list[1][:, 1] +
                                     1.96 * array_list[1][:, 2],
                                     array_list[1][:, 1] -
                                     1.96 * array_list[1][:, 2], alpha=0.5,
                                     color=line_colors[k],
                                     facecolor=line_colors[k])
            axarr[0, 0].legend(loc='upper right', shadow=True)

            axarr[0, 1].plot(array_list[2][:, 0], array_list[2][:, 1],
                             color=line_colors[k])
            axarr[0, 1].set_title('Female level 2')
            axarr[0, 1].set_xlabel(array_list[7])
            axarr[0, 1].set_ylabel('Number of Females')
            axarr[0, 1].fill_between(array_list[2][:, 0], array_list[2][:, 1] +
                                     1.96 * array_list[2][:, 2],
                                     array_list[2][:, 1] -
                                     1.96 * array_list[2][:, 2], alpha=0.5,
                                     color=line_colors[k],
                                     facecolor=line_colors[k])

            axarr[0, 2].plot(array_list[3][:, 0], array_list[3][:, 1],
                             color=line_colors[k])
            axarr[0, 2].set_title('Female level 3')
            axarr[0, 2].set_xlabel(array_list[7])
            axarr[0, 2].set_ylabel('Number of Females')
            axarr[0, 2].fill_between(array_list[3][:, 0], array_list[3][:, 1] +
                                     1.96 * array_list[3][:, 2],
                                     array_list[3][:, 1] -
                                     1.96 * array_list[3][:, 2], alpha=0.5,
                                     color=line_colors[k],
                                     facecolor=line_colors[k])

            axarr[1, 0].plot(array_list[4][:, 0], array_list[4][:, 1],
                             color=line_colors[k])
            axarr[1, 0].set_title('Male level 1')
            axarr[1, 0].set_xlabel(array_list[7])
            axarr[1, 0].set_ylabel('Number of Males')
            axarr[1, 0].fill_between(array_list[4][:, 0], array_list[4][:, 1] +
                                     1.96 * array_list[4][:, 2],
                                     array_list[4][:, 1] -
                                     1.96 * array_list[4][:, 2], alpha=0.5,
                                     color=line_colors[k],
                                     facecolor=line_colors[k])

            axarr[1, 1].plot(array_list[5][:, 0], array_list[5][:, 1],
                             color=line_colors[k])
            axarr[1, 1].set_title('Male level 2')
            axarr[1, 1].set_xlabel(array_list[7])
            axarr[1, 1].set_ylabel('Number of Males')
            axarr[1, 1].fill_between(array_list[5][:, 0], array_list[5][:, 1] +
                                     1.96 * array_list[5][:, 2],
                                     array_list[5][:, 1] -
                                     1.96 * array_list[5][:, 2], alpha=0.5,
                                     color=line_colors[k],
                                     facecolor=line_colors[k])

            axarr[1, 2].plot(array_list[6][:, 0], array_list[6][:, 1],
                             color=line_colors[k])
            axarr[1, 2].set_title('Male level 3')
            axarr[1, 2].set_xlabel(array_list[7])
            axarr[1, 2].set_ylabel('Number of Males')
            axarr[1, 2].fill_between(array_list[6][:, 0], array_list[6][:, 1] +
                                     1.96 * array_list[6][:, 2],
                                     array_list[6][:, 1] -
                                     1.96 * array_list[6][:, 2], alpha=0.5,
                                     color=line_colors[k],
                                     facecolor=line_colors[k])
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()


    def plot_comparison_overall_chart(self,
                           plottype,
                           number_of_runs,
                           target,
                           caption,
                           xlabel,
                           ylabel,
                           title,
                           line_width,
                           xmin,
                           ymin,
                           xmax,
                           ymax,
                           transparency,
                           marker_shape=None,
                           linecolor='g',
                           target_plot=False,
                           legend_location='upper right',
                           color_target='r',
                           percent_line_plot=False,
                           percent_line_value=0.5,
                           color_percent_line='r',
                           target_plot_line_style='--',
                           percent_line_style='-.',
                           target_plot_linewidth=2,
                           percent_linewidth=2,
                           model_legend_label='model',
                           target_plot_legend_label='target',
                           percent_legend_label='percent',
                           male_female_numbers_plot=False,
                           mf_male_color='k',
                           mf_target_color='r',
                           mf_male_label='Male',
                           mf_target_label='Target',
                           mf_male_linestyle=None,
                           mf_target_linestyle=None,
                           mf_male_linewidth=2,
                           mf_target_linewidth=2
                           ):
        # generate data for the plot.

        for mod in self.mlist:
            mod.run_multiple(number_of_runs)

        # set default plot parameters. The xaxis is generally duration,
        # though I have the option--depending on the plot, to put in a
        # different x-axis.

        xval = min([m.duration for m in self.mlist])

        if plottype == 'probability proportion':

            for mod in self.mlist:
                mod.run_probability_analysis_gender_proportion(number_of_runs,
                                                            target)

            yval = [m.probability_matrix['Probability'] for m in self.mlist]
            fill_matrix = np.zeros(len(self.mlist)).tolist()

        if plottype == 'gender proportion':

            yval = [m.mean_matrix['gendprop'] for m in self.mlist]

            fill_matrix = [m.std_matrix['gendprop'] for m in self.mlist]

        if plottype == 'gender numbers':
            pass

        if plottype == 'unfilled vacancies':
            yval = [m.mean_matrix['unfilled'] for m in self.mlist]

            fill_matrix = [m.std_matrix['unfilled'] for m in self.mlist]

        if plottype == 'department size':
            yval = [m.dept_size_matrix['mean'] for m in self.mlist]

            fill_matrix = [m.dept_size_matrix['std'] for m in self.mlist]

        if plottype == 'male female numbers':

            yval = [sum(list([m.mean_matrix['f1'],
                             m.mean_matrix['f2'],
                             m.mean_matrix['f3']])) for m in self.mlist]

            fill_matrix = np.zeros(len(self.mlist)).tolist()

            yval2 = [sum(list([m.mean_matrix['m1'],
                             m.mean_matrix['m2'],
                             m.mean_matrix['m3']])) for m in self.mlist]


            total_faculty = [sum(list([m.mean_matrix['m1'],
                             m.mean_matrix['m2'],
                             m.mean_matrix['m3'],
                             m.mean_matrix['f1'],
                             m.mean_matrix['f2'],
                             m.mean_matrix['f3']])) for m in self.mlist]

            yval3 = [np.round(target * dept) for dept in total_faculty]

        # Execute plots

        for k,v in enumerate(self.mlist):
            plt.plot(range(xval),
                     yval[k],
                     linewidth=line_width,
                     marker=marker_shape[k],
                     color=linecolor[k],
                     label=model_legend_label[k])

            plt.fill_between(range(xval),
                             yval[k] + 1.96 * fill_matrix[k],
                             yval[k] - 1.96 * fill_matrix[k],
                             alpha=transparency[k],
                             facecolor=linecolor[k])

            if male_female_numbers_plot:
                plt.plot(range(xval),
                         yval2[k],
                         color=mf_male_color[k],
                         label=mf_male_label[k],
                         linestyle=mf_male_linestyle,
                         linewidth=mf_male_linewidth)

                plt.plot(range(xval),
                         yval3[k],
                         color=mf_target_color,
                         label=mf_target_label[k],
                         linestyle=mf_target_linestyle[k],
                         linewidth=mf_target_linewidth)

        if target_plot:
            plt.axhline(target,
                        color=color_target,
                        linestyle=target_plot_line_style,
                        label=target_plot_legend_label,
                        linewidth=target_plot_linewidth)

        if percent_line_plot:
            plt.axhline(y=percent_line_value,
                        color=color_percent_line,
                        linestyle=percent_line_style,
                        label=percent_legend_label,
                        linewidth=percent_linewidth)



        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc=legend_location, shadow=True)
        plt.show()


    def plot_comparison_level_chart(self,
                         plottype,
                         number_of_runs,
                         target,
                         caption,
                         xlabel_f1,
                         ylabel_f1,
                         xlabel_f2,
                         ylabel_f2,
                         xlabel_f3,
                         ylabel_f3,
                         xlabel_m1,
                         ylabel_m1,
                         xlabel_m2,
                         ylabel_m2,
                         xlabel_m3,
                         ylabel_m3,
                         group_title,
                         title_f1,
                         title_f2,
                         title_f3,
                         title_m1,
                         title_m2,
                         title_m3,
                         line_width,
                         xmin_f1,
                         ymin_f1,
                         xmax_f1,
                         ymax_f1,
                         xmin_f2,
                         ymin_f2,
                         xmax_f2,
                         ymax_f2,
                         xmin_f3,
                         ymin_f3,
                         xmax_f3,
                         ymax_f3,
                         xmin_m1,
                         ymin_m1,
                         xmax_m1,
                         ymax_m1,
                         xmin_m2,
                         ymin_m2,
                         xmax_m2,
                         ymax_m2,
                         xmin_m3,
                         ymin_m3,
                         xmax_m3,
                         ymax_m3,
                         legend_location='upper right',
                         model_legend_label='model',
                         transparency = 0.25,
                         marker_shape=None,
                         linecolor='g',
                         target_plot=False,
                         target_color='r',
                         target_plot_line_style='--',
                         target_plot_linewidth=2,
                         target_plot_legend_label='target',
                         percent_line_plot=False,
                         percent_line_value=0.5,
                         color_percent_line='r',
                         percent_line_style='-.',
                         percent_linewidth=2,
                         percent_legend_label='percent'):

        # generate data for the plot.

        for mod in self.mlist:
            mod.run_multiple(number_of_runs)

        # set default plot parameters. The xaxis is generally duration,
        # though I have the option--depending on the plot, to put in a
        # different x-axis.

        xval = min([m.duration for m in self.mlist])

        if plottype == 'probability proportion':
            for mod in self.mlist:
                mod.run_probability_analysis_gender_by_level(number_of_runs,
                                                             target)


            # Not a very finesse way to do these assignments, but it makes
            # the code more readable.
            yval_f1 = [m.probability_by_level['pf1'] for m in self.mlist]
            yval_f2 = [m.probability_by_level['pf2'] for m in self.mlist]
            yval_f3 = [m.probability_by_level['pf3'] for m in self.mlist]
            yval_m1 = [m.probability_by_level['pm1'] for m in self.mlist]
            yval_m2 = [m.probability_by_level['pm2'] for m in self.mlist]
            yval_m3 = [m.probability_by_level['pm3'] for m in self.mlist]

            fill_f1 = np.zeros(len(self.mlist)).tolist()
            fill_f2 = np.zeros(len(self.mlist)).tolist()
            fill_f3 = np.zeros(len(self.mlist)).tolist()
            fill_m1 = np.zeros(len(self.mlist)).tolist()
            fill_m2 = np.zeros(len(self.mlist)).tolist()
            fill_m3 = np.zeros(len(self.mlist)).tolist()

        if plottype == 'gender proportion':

            female_pct_matrices = [m.pct_female_matrix for m in self.mlist]

            yval_f1 = [m['mpct_f1'] for m in female_pct_matrices]
            yval_f2 = [m['mpct_f2'] for m in female_pct_matrices]
            yval_f3 = [m['mpct_f3'] for m in female_pct_matrices]
            yval_m1 = [m['mpct_m1'] for m in female_pct_matrices]
            yval_m2 = [m['mpct_m2'] for m in female_pct_matrices]
            yval_m3 = [m['mpct_m3'] for m in female_pct_matrices]

            fill_f1 = [m['spct_f1'] for m in female_pct_matrices]
            fill_f2 = [m['spct_f2'] for m in female_pct_matrices]
            fill_f3 = [m['spct_f3'] for m in female_pct_matrices]
            fill_m1 = [m['spct_m1'] for m in female_pct_matrices]
            fill_m2 = [m['spct_m2'] for m in female_pct_matrices]
            fill_m3 = [m['spct_m3'] for m in female_pct_matrices]

        if plottype == 'gender number':

            mean_matrices = [m.mean_matrix for m in self.mlist]
            yval_f1 = [m['f1'] for m in mean_matrices]
            yval_f2 = [m['f2'] for m in mean_matrices]
            yval_f3 = [m['f3'] for m in mean_matrices]
            yval_m1 = [m['m1'] for m in mean_matrices]
            yval_m2 = [m['m2'] for m in mean_matrices]
            yval_m3 = [m['m3'] for m in mean_matrices]

            std_matrices = [m.std_matrix for m in self.mlist]
            fill_f1 = [m['f1'] for m in std_matrices]
            fill_f2 = [m['f2'] for m in std_matrices]
            fill_f3 = [m['f3'] for m in std_matrices]
            fill_m1 = [m['m1'] for m in std_matrices]
            fill_m2 = [m['m2'] for m in std_matrices]
            fill_m3 = [m['m3'] for m in std_matrices]


        f, axarr = plt.subplots(nrows=2, ncols=3)
        f.suptitle(group_title)
        for k, v in enumerate(self.mlist):
            axarr[0, 0].plot(range(xval),
                             np.minimum(1,
                                        np.maximum(0,
                                                   yval_f1[k])),
                             label=model_legend_label[k],
                             linewidth=line_width,
                             color=linecolor[k],
                             marker=marker_shape[k])
            axarr[0, 0].set_xlim([xmin_f1, xmax_f1])
            axarr[0, 0].set_ylim([ymin_f1, ymax_f1])
            axarr[0, 0].set_title(title_f1)
            axarr[0, 0].set_xlabel(xlabel_f1)
            axarr[0, 0].set_ylabel(ylabel_f1)
            axarr[0, 0].fill_between(range(xval),
                                     np.minimum(1,
                                                yval_f1[k] + 1.96 * fill_f1[k]),
                                     np.maximum(0,
                                                yval_f1[k] - 1.96 * fill_f1[k]),
                                     alpha=transparency[k],
                                     facecolor=linecolor[k])
            axarr[0, 0].legend(loc=legend_location, shadow=True)

            axarr[0, 1].plot(range(xval),
                             np.minimum(1,
                                        np.maximum(0,
                                                   yval_f2[k])),
                             label=model_legend_label[k],
                             linewidth=line_width,
                             color=linecolor[k],
                             marker=marker_shape[k]
                             )
            axarr[0, 1].set_xlim([xmin_f2, xmax_f2])
            axarr[0, 1].set_ylim([ymin_f2, ymax_f2])
            axarr[0, 1].set_title(title_f2)
            axarr[0, 1].set_xlabel(xlabel_f2)
            axarr[0, 1].set_ylabel(ylabel_f2)
            axarr[0, 1].fill_between(range(xval),
                                     np.minimum(1,
                                                yval_f2[k] + 1.96 * fill_f2[k]),
                                     np.maximum(0,
                                                yval_f2[k] - 1.96 * fill_f2[k]),
                                     alpha=transparency[k],
                                     facecolor=linecolor[k])

            axarr[0, 2].plot(range(xval),
                             np.minimum(1,
                                        np.maximum(0,
                                                   yval_f3[k])),
                             label=model_legend_label[k],
                             linewidth=line_width,
                             color=linecolor[k],
                             marker=marker_shape[k]
                             )
            axarr[0, 2].set_xlim([xmin_f3, xmax_f3])
            axarr[0, 2].set_ylim([ymin_f3, ymax_f3])
            axarr[0, 2].set_title(title_f3)
            axarr[0, 2].set_xlabel(xlabel_f3)
            axarr[0, 2].set_ylabel(ylabel_f3)
            axarr[0, 2].fill_between(range(xval),
                                     np.minimum(1,
                                                yval_f3[k] + 1.96 * fill_f3[k]),
                                     np.maximum(0,
                                                yval_f3[k] - 1.96 * fill_f3[k]),
                                     alpha=transparency[k],
                                     facecolor=linecolor[k])

            axarr[1, 0].plot(range(xval),
                             np.minimum(1,
                                        np.maximum(0,
                                                   yval_m1[k])),
                             label=self.label[k],
                             linewidth=line_width,
                             color=linecolor[k],
                             marker=marker_shape[k]
                             )
            axarr[1, 0].set_xlim([xmin_m1, xmax_m1])
            axarr[1, 0].set_ylim([ymin_m1, ymax_m1])
            axarr[1, 0].set_title(title_m1)
            axarr[1, 0].set_xlabel(xlabel_m1)
            axarr[1, 0].set_ylabel(ylabel_m1)
            axarr[1, 0].fill_between(range(xval),
                                     np.minimum(1,
                                                yval_m1[k] + 1.96 * fill_m1[k]),
                                     np.maximum(0,
                                                yval_m1[k] - 1.96 * fill_m1[k]),
                                     alpha=transparency[k],
                                     facecolor=linecolor[k])

            axarr[1, 1].plot(range(xval),
                             np.minimum(1,
                                        np.maximum(0,
                                                   yval_m2[k])),
                             label=self.label[k],
                             linewidth=line_width,
                             color=linecolor[k],
                             marker=marker_shape[k]
                             )
            axarr[1, 1].set_xlim([xmin_m2, xmax_m2])
            axarr[1, 1].set_ylim([ymin_m2, ymax_m2])
            axarr[1, 1].set_title(title_m2)
            axarr[1, 1].set_xlabel(xlabel_m2)
            axarr[1, 1].set_ylabel(ylabel_m2)
            axarr[1, 1].fill_between(range(xval),
                                     np.minimum(1,
                                                yval_m2[k] + 1.96 * fill_m2[k]),
                                     np.maximum(0,
                                                yval_m2[k] - 1.96 * fill_m2[k]),
                                     alpha=transparency[k],
                                     facecolor=linecolor[k])

            axarr[1, 2].plot(range(xval),
                             np.minimum(1,
                                        np.maximum(0,
                                                   yval_m3[k])),
                             label=self.label[k],
                             linewidth=line_width,
                             color=linecolor[k],
                             marker=marker_shape[k]
                             )
            axarr[1, 2].set_xlim([xmin_m3, xmax_m3])
            axarr[1, 2].set_ylim([ymin_m3, ymax_m3])
            axarr[1, 2].set_title(title_m3)
            axarr[1, 2].set_xlabel(xlabel_m3)
            axarr[1, 2].set_ylabel(ylabel_m3)
            axarr[1, 2].fill_between(range(xval),
                                     np.minimum(1,
                                                yval_m3[k] + 1.96 * fill_m3[k]),
                                     np.maximum(0,
                                                yval_m3[k] - 1.96 * fill_m3[k]),
                                     alpha=transparency[k],
                                     facecolor=linecolor[k])

        if target_plot == True:
            axarr[0, 0].axhline(y=target,
                                color=target_color,
                                linestyle = target_plot_line_style,
                                linewidth = target_plot_linewidth,
                                label = target_plot_legend_label)
            axarr[0, 0].legend(loc=legend_location, shadow=True)

            axarr[0, 1].axhline(y=target,
                                color=target_color,
                                linestyle = target_plot_line_style,
                                linewidth = target_plot_linewidth,
                                label = target_plot_legend_label)
            axarr[0, 2].axhline(y=target,
                                color=target_color,
                                linestyle = target_plot_line_style,
                                linewidth = target_plot_linewidth,
                                label = target_plot_legend_label)
            axarr[1, 0].axhline(y=1 - target,
                                color=target_color,
                                linestyle = target_plot_line_style,
                                linewidth = target_plot_linewidth,
                                label = target_plot_legend_label)
            axarr[1, 1].axhline(y=1 - target,
                                color=target_color,
                                linestyle = target_plot_line_style,
                                linewidth = target_plot_linewidth,
                                label = target_plot_legend_label)
            axarr[1, 2].axhline(y=1 - target,
                                color=target_color,
                                linestyle = target_plot_line_style,
                                linewidth = target_plot_linewidth,
                                label = target_plot_legend_label)

        if percent_line_plot == True:
            axarr[0, 0].axhline(y=percent_line_value,
                                color=color_percent_line,
                                linestyle = percent_line_style,
                                linewidth = percent_linewidth,
                                label = percent_legend_label)
            axarr[0, 0].legend(loc=legend_location, shadow=True)

            axarr[0, 1].axhline(y=percent_line_value,
                                color=color_percent_line,
                                linestyle=percent_line_style,
                                linewidth=percent_linewidth,
                                label=percent_legend_label)
            axarr[0, 2].axhline(y=percent_line_value,
                                color=color_percent_line,
                                linestyle=percent_line_style,
                                linewidth=percent_linewidth,
                                label=percent_legend_label)
            axarr[1, 0].axhline(y=percent_line_value,
                                color=color_percent_line,
                                linestyle=percent_line_style,
                                linewidth=percent_linewidth,
                                label=percent_legend_label)
            axarr[1, 1].axhline(y=percent_line_value,
                                color=color_percent_line,
                                linestyle=percent_line_style,
                                linewidth=percent_linewidth,
                                label=percent_legend_label)
            axarr[1, 2].axhline(y=percent_line_value,
                                color=color_percent_line,
                                linestyle=percent_line_style,
                                linewidth=percent_linewidth,
                                label=percent_legend_label)


        plt.show()





    def testfunc(self):

        for mod in self.mlist:
            mod.run_multiple(5)

        yval = [m.mean_matrix['gendprop'] for m in self.mlist]
        return(yval)



