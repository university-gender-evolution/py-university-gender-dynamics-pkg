__author__ = 'krishnab'
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.charts import defaults
from bokeh.layouts import gridplot
from operator import add, sub

defaults.height = 800
defaults.width = 800
# CONSTANTS

# line_colors = ['#7fc97f', '#beaed4', '#fdc086','#386cb0','#f0027f','#ffff99']

class Comparison():
    def __init__(self, model_list):
        self.name = 'All Models'
        self.label = 'All Models'
        self.mlist = model_list

    # def plot_comparison_spec_parameter(self, parm, title,
    #                                    xlabel, ylabel, number_of_runs=10):
    #
    #     ## This function will execute gender proportion comparisons for all models
    #
    #     ## Color list
    #
    #     line_colors = ['#7fc97f', '#beaed4', '#fdc086', '#386cb0', '#f0027f',
    #                    '#ffff99']
    #
    #     for mod in self.mlist:
    #         mod.run_multiple(number_of_runs)
    #
    #     ## Create plot array and execute plot
    #
    #     for k, v in enumerate(self.mlist):
    #         plt.plot(range(self.mlist[k].duration),
    #                  self.mlist[k].mean_matrix[parm],
    #                  color=line_colors[k], label=self.mlist[k].label,
    #                  linewidth=2.0)
    #
    #         plt.plot(range(self.mlist[k].duration),
    #                  self.mlist[k].mean_matrix[parm])
    #         plt.fill_between(range(self.mlist[k].duration),
    #                          self.mlist[k].mean_matrix[
    #                              parm] +
    #                          1.96 * self.mlist[k].std_matrix[
    #                              parm],
    #                          self.mlist[k].mean_matrix[parm] - 1.96 *
    #                          self.mlist[k].std_matrix[
    #                              parm], color=line_colors[k], alpha=0.5)
    #
    #     plt.title(title)
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.legend(loc='upper right', shadow=True)
    #     plt.show()


    # def plot_parameter_sweep_gender_proportion(self, number_of_runs, param,
    #                                            llim, ulim, number_of_steps):
    #
    #     ## This function will execute gender proportion comparisons for all models
    #
    #     ## Color list
    #
    #     line_colors = ['#7fc97f', '#beaed4', '#fdc086', '#386cb0', '#f0027f',
    #                    '#ffff99']
    #     for mod in self.mlist:
    #         mod.run_parameter_sweep(number_of_runs, param, llim, ulim,
    #                                 number_of_steps)
    #
    #     ## Create plot array and execute plot
    #
    #     for k, v in enumerate(self.mlist):
    #         plot_array = self.mlist[k].parameter_sweep_array[0]
    #         plt.plot(plot_array[:, 0], plot_array[:, 1],
    #                  label=self.mlist[k].label, linewidth=2.0,
    #                  color=line_colors[k])
    #         plt.fill_between(plot_array[:, 0], plot_array[:, 1] +
    #                          1.96 * plot_array[:, 2], plot_array[:, 1] -
    #                          1.96 * plot_array[:, 2], alpha=0.5,
    #                          color=line_colors[k], facecolor=line_colors[k])
    #
    #     plt.title('Parameter Sweep for Gender Proportion over ' + str(
    #         self.mlist[0].duration) + ' years')
    #     plt.xlabel(param)
    #     plt.ylabel('Percentage of the Department that is Women')
    #     plt.legend(loc='upper right', shadow=True)
    #     plt.show()
    #
    # def plot_parameter_sweep_detail(self,
    #                                 number_of_runs,
    #                                 param,
    #                                 llim,
    #                                 ulim,
    #                                 number_of_steps):
    #
    #     # create color list
    #
    #     line_colors = ['#7fc97f', '#beaed4', '#fdc086', '#386cb0', '#f0027f',
    #                    '#ffff99']
    #     plot_titles = ['extra', 'Female Level 1', 'Female level 2', 'Female '
    #                                                                 'level 3',
    #                    'Male level 1', 'Male level 2', 'Male level 3']
    #     plot_y_axis_titles = ['extra', 'Number of Females', 'Number of Females',
    #                           'Number of Females', 'Number of Males',
    #                           'Number of Males', 'Number of Males']
    #
    #     # run all models and generate result matrices. I run each model so I can guarantee that they all have the same duration.
    #
    #     for mod in self.mlist:
    #         mod.run_parameter_sweep(number_of_runs, param, llim, ulim,
    #                                 number_of_steps)
    #
    #     f, axarr = plt.subplots(nrows=2, ncols=3)
    #     f.suptitle("Parameter sweep")
    #
    #     for k, v in enumerate(self.mlist):
    #         array_list = self.mlist[k].parameter_sweep_array
    #
    #         axarr[0, 0].plot(array_list[1][:, 0], array_list[1][:, 1],
    #                          label=self.mlist[k].label, linewidth=2.0,
    #                          color=line_colors[k])
    #         axarr[0, 0].set_title('Female level 1')
    #         axarr[0, 0].set_xlabel(array_list[7])
    #         axarr[0, 0].set_ylabel('Number of Females')
    #         axarr[0, 0].fill_between(array_list[1][:, 0], array_list[1][:, 1] +
    #                                  1.96 * array_list[1][:, 2],
    #                                  array_list[1][:, 1] -
    #                                  1.96 * array_list[1][:, 2], alpha=0.5,
    #                                  color=line_colors[k],
    #                                  facecolor=line_colors[k])
    #         axarr[0, 0].legend(loc='upper right', shadow=True)
    #
    #         axarr[0, 1].plot(array_list[2][:, 0], array_list[2][:, 1],
    #                          color=line_colors[k])
    #         axarr[0, 1].set_title('Female level 2')
    #         axarr[0, 1].set_xlabel(array_list[7])
    #         axarr[0, 1].set_ylabel('Number of Females')
    #         axarr[0, 1].fill_between(array_list[2][:, 0], array_list[2][:, 1] +
    #                                  1.96 * array_list[2][:, 2],
    #                                  array_list[2][:, 1] -
    #                                  1.96 * array_list[2][:, 2], alpha=0.5,
    #                                  color=line_colors[k],
    #                                  facecolor=line_colors[k])
    #
    #         axarr[0, 2].plot(array_list[3][:, 0], array_list[3][:, 1],
    #                          color=line_colors[k])
    #         axarr[0, 2].set_title('Female level 3')
    #         axarr[0, 2].set_xlabel(array_list[7])
    #         axarr[0, 2].set_ylabel('Number of Females')
    #         axarr[0, 2].fill_between(array_list[3][:, 0], array_list[3][:, 1] +
    #                                  1.96 * array_list[3][:, 2],
    #                                  array_list[3][:, 1] -
    #                                  1.96 * array_list[3][:, 2], alpha=0.5,
    #                                  color=line_colors[k],
    #                                  facecolor=line_colors[k])
    #
    #         axarr[1, 0].plot(array_list[4][:, 0], array_list[4][:, 1],
    #                          color=line_colors[k])
    #         axarr[1, 0].set_title('Male level 1')
    #         axarr[1, 0].set_xlabel(array_list[7])
    #         axarr[1, 0].set_ylabel('Number of Males')
    #         axarr[1, 0].fill_between(array_list[4][:, 0], array_list[4][:, 1] +
    #                                  1.96 * array_list[4][:, 2],
    #                                  array_list[4][:, 1] -
    #                                  1.96 * array_list[4][:, 2], alpha=0.5,
    #                                  color=line_colors[k],
    #                                  facecolor=line_colors[k])
    #
    #         axarr[1, 1].plot(array_list[5][:, 0], array_list[5][:, 1],
    #                          color=line_colors[k])
    #         axarr[1, 1].set_title('Male level 2')
    #         axarr[1, 1].set_xlabel(array_list[7])
    #         axarr[1, 1].set_ylabel('Number of Males')
    #         axarr[1, 1].fill_between(array_list[5][:, 0], array_list[5][:, 1] +
    #                                  1.96 * array_list[5][:, 2],
    #                                  array_list[5][:, 1] -
    #                                  1.96 * array_list[5][:, 2], alpha=0.5,
    #                                  color=line_colors[k],
    #                                  facecolor=line_colors[k])
    #
    #         axarr[1, 2].plot(array_list[6][:, 0], array_list[6][:, 1],
    #                          color=line_colors[k])
    #         axarr[1, 2].set_title('Male level 3')
    #         axarr[1, 2].set_xlabel(array_list[7])
    #         axarr[1, 2].set_ylabel('Number of Males')
    #         axarr[1, 2].fill_between(array_list[6][:, 0], array_list[6][:, 1] +
    #                                  1.96 * array_list[6][:, 2],
    #                                  array_list[6][:, 1] -
    #                                  1.96 * array_list[6][:, 2], alpha=0.5,
    #                                  color=line_colors[k],
    #                                  facecolor=line_colors[k])
    #     plt.tight_layout()
    #     plt.subplots_adjust(top=0.85)
    #     plt.show()


    def plot_comparison_overall_chart(self,
                           plottype,
                           intervals,
                           number_of_runs,
                           target,
                           xlabel,
                           ylabel,
                           title,
                           line_width=2,
                           width_=defaults.width,
                           height_=defaults.height,
                           transparency = 0.25,
                           linecolor=['#018571', '#a6611a', '#e66101'],
                           target_plot=False,
                           legend_location='top_right',
                           color_target='#ca0020',
                           percent_line_plot=False,
                           percent_line_value=0.5,
                           color_percent_line='#ca0020',
                           target_plot_linewidth=2,
                           percent_linewidth=2,
                           model_legend_label=['model'],
                           target_plot_legend_label='target',
                           percent_legend_label='percent',
                           male_female_numbers_plot=False,
                           mf_male_color=['#a6611a', '#2c7bb6'],
                           mf_target_color=['#fdae61'],
                           mf_male_label=['Male model 1', 'Male model 2'],
                           mf_target_label='Target',
                           mf_male_linewidth=2,
                           mf_target_linewidth=2,
                           parameter_sweep_param = None,
                           parameter_ubound = 0,
                           parameter_lbound = 0,
                           number_of_steps = 0
                           ):


        # Choose plot type. This block will initialize the data for the
        # plots. If the plot is a parameter sweep, then it will run the
        # Models.run_parameter_sweep() function. If this is not a parameter
        # sweep, then the block will run the Models.run_multiple() function.

        # BEGIN BLOCK
        if plottype in ['parameter sweep percentage',
                        'parameter sweep probability']:

            for mod in self.mlist:
                mod.run_parameter_sweep(number_of_runs,
                                        parameter_sweep_param,
                                        parameter_lbound,
                                        parameter_ubound,
                                        number_of_steps)

            # xval, so I need to directly feed this range in.

            xval = self.mlist[0].parameter_sweep_results.loc[:,
                           'increment']

        else:

            for mod in self.mlist:
                mod.run_multiple(number_of_runs)

            xval = list(range(min([m.duration for m in self.mlist])))

        # END OF BLOCK



        # BEGIN BLOCK
        # This section  just initializes the bokeh figure with labels and
        # sizing.
        p = figure(title=title,
               x_axis_label=xlabel,
               y_axis_label=ylabel,
               width=width_,
               height=height_)

        # END BLOCK

        # BEGIN BLOCK
        # Generate main plot values.
        # This section will take the data from the model simulation and
        # convert it to the appropriate y-variables for the plot.


        if plottype == 'probability proportion':

            for mod in self.mlist:
                mod.run_probability_analysis_gender_proportion(number_of_runs,
                                                            target)

            yval = [m.probability_matrix['Probability'] for m in self.mlist]

        if plottype == 'gender proportion':

            yval = [m.results_matrix['mean_gendprop'] for m in self.mlist]

        if plottype == 'gender numbers':
            pass

        if plottype == 'unfilled vacancies':
            yval = [m.results_matrix['mean_unfilled'] for m in self.mlist]


        if plottype == 'department size':
            yval = [m.results_matrix['mean_dept_size'] for m in self.mlist]

        if plottype == 'male female numbers':

            yval = [sum(list([m.results_matrix['mean_f1'],
                             m.results_matrix['mean_f2'],
                             m.results_matrix['mean_f3']])) for m in self.mlist]



            yval2 = [sum(list([m.results_matrix['mean_m1'],
                             m.results_matrix['mean_m2'],
                             m.results_matrix['mean_m3']])) for m in self.mlist]


            total_faculty = [sum(list([m.results_matrix['mean_m1'],
                             m.results_matrix['mean_m2'],
                             m.results_matrix['mean_m3'],
                             m.results_matrix['mean_f1'],
                             m.results_matrix['mean_f2'],
                             m.results_matrix['mean_f3']])) for m in self.mlist]

            yval3 = [np.round(target * dept) for dept in total_faculty]

        if plottype == 'parameter sweep percentage':

            yval = [m.parameter_sweep_results['mean_gendprop'] for m in
                    self.mlist]

        if plottype == 'parameter sweep probability':

            for mod in self.mlist:
                mod.run_probability_parameter_sweep_overall(number_of_runs,
                                                            parameter_sweep_param,
                                                            parameter_lbound,
                                                            parameter_ubound,
                                                            number_of_steps,
                                                            target)

            yval = [m.probability_sweep_results['Probability'] for m in
                    self.mlist]


        # Set confidence bounds using empirical results

        if intervals == 'empirical':
            if plottype == 'probability proportion':

                upper_band = yval
                lower_band = yval


            if plottype == 'gender proportion':

                upper_band = [m.results_matrix['gendprop_975'] for m in
                              self.mlist]
                lower_band = [m.results_matrix['gendprop_025'] for m in
                              self.mlist]

            if plottype == 'unfilled vacancies':
                upper_band = [m.results_matrix['unfilled_975'] for m in
                              self.mlist]
                lower_band = [m.results_matrix['unfilled_025'] for m in
                              self.mlist]

            if plottype == 'department size':

                upper_band = [m.results_matrix['dept_size_975'] for m in
                              self.mlist]
                lower_band = [m.results_matrix['dept_size_025'] for m in
                              self.mlist]

            if plottype == 'male female numbers':
                upper_band = yval
                lower_band = yval

            if plottype == 'parameter sweep percentage':
                upper_band = [m.parameter_sweep_results['gendprop_975'] for m in
                              self.mlist]
                lower_band = [m.parameter_sweep_results['gendprop_025'] for m in
                              self.mlist]

            if plottype == 'parameter sweep probability':
                upper_band = yval
                lower_band = yval


        # Set confidence bounds using 2 standard deviations

        if intervals == 'standard':

            if plottype == 'probability proportion':

                upper_band = yval
                lower_band = yval

            if plottype == 'gender proportion':

                fill_matrix = [m.results_matrix['std_gendprop'] for m in
                               self.mlist]

                upper_band = list(map(add, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))

                lower_band = list(map(sub, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))

            if plottype == 'unfilled vacancies':
                fill_matrix = [m.results_matrix['std_unfilled'] for m in
                               self.mlist]
                upper_band = list(map(add, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))
                lower_band = list(map(sub, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))

            if plottype == 'department size':
                fill_matrix = [m.results_matrix['std_dept_size'] for m in
                               self.mlist]
                upper_band = list(map(add, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))
                lower_band = list(map(sub, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))

            if plottype == 'male female numbers':

                upper_band = yval
                lower_band = yval

            if plottype == 'parameter sweep percentage':

                fill_matrix = [m.parameter_sweep_results['std_gendprop'] for m in
                               self.mlist]

                upper_band = list(map(add, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))

                lower_band = list(map(sub, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))

            if plottype == 'parameter sweep probability':
                upper_band = yval
                lower_band = yval

        # Execute plots

        for k, v in enumerate(self.mlist):

            p.line(xval, yval[k],
                   line_width=line_width,
                   line_color=linecolor[k],
                   legend=model_legend_label[k])

            p.circle(xval, yval[k], size=3)

            x_data = np.asarray(xval)
            band_x = np.append(x_data, x_data[::-1])
            band_y = np.append(lower_band[k], upper_band[k][::-1])

            p.patch(band_x,
                    band_y,
                    color=linecolor[k],
                    fill_alpha=transparency)

            if male_female_numbers_plot:
                p.line(xval,
                       yval2[k],
                       line_color=mf_male_color[k],
                       legend=mf_male_label[k],
                       line_width=mf_male_linewidth)

                p.circle(xval, yval2[k], size=3)

                p.line(xval,
                       yval3[k],
                       line_color=mf_target_color,
                       legend=mf_target_label,
                       line_width=mf_target_linewidth)

        if target_plot:
            p.line(xval, target,
                   line_color=color_target,
                   legend=target_plot_legend_label,
                   line_width=target_plot_linewidth)

        if percent_line_plot:
            p.line(xval, percent_line_value,
                   line_color=color_percent_line,
                   legend=percent_legend_label,
                   line_width=percent_linewidth)

        return(p)


    def plot_comparison_level_chart(self,
                         plottype,
                         intervals,
                         number_of_runs,
                         target,
                         xlabels,
                         ylabels,
                         titles = ['f1', 'f2', 'f3', 'm1', 'm2', 'm3'],
                         line_width=2,
                         height_= defaults.height // 2,
                         width_ = defaults.width // 2,
                         legend_location='top right',
                         model_legend_label='model',
                         transparency = 0.25,
                         linecolor=['#018571', '#a6611a', '#e66101'],
                         target_plot=False,
                         target_color='#ca0020',
                         target_plot_linewidth=2,
                         target_plot_legend_label='target',
                         percent_line_plot=False,
                         percent_line_value=0.5,
                         color_percent_line='#ca0020',
                         percent_linewidth=2,
                         percent_legend_label='percent',
                         parameter_sweep_param=None,
                         parameter_ubound=0,
                         parameter_lbound=0,
                         number_of_steps=0
                        ):

        # Choose plot type. This block will initialize the data for the
        # plots. If the plot is a parameter sweep, then it will run the
        # Models.run_parameter_sweep() function. If this is not a parameter
        # sweep, then the block will run the Models.run_multiple() function.

        # BEGIN BLOCK

        if plottype in ['parameter sweep gender percentage',
                        'parameter sweep probability',
                        'parameter sweep gender number']:

            for mod in self.mlist:
                mod.run_parameter_sweep(number_of_runs,
                                        parameter_sweep_param,
                                        parameter_lbound,
                                        parameter_ubound,
                                        number_of_steps)

            # xval, so I need to directly feed this range in.

            xval = self.mlist[0].parameter_sweep_results.loc[:, 'increment']

        else:

            for mod in self.mlist:
                mod.run_multiple(number_of_runs)

            xval = list(range(min([m.duration for m in self.mlist])))

        # END BLOCK


        # Generate main plot values.
        # This section will take the data from the model simulation and
        # convert it to the appropriate y-variables for the plot.
        # BEGIN BLOCK

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


        if plottype == 'gender proportion':

            female_pct_matrices = [m.pct_female_matrix for m in self.mlist]

            yval_f1 = [m['mpct_f1'] for m in female_pct_matrices]
            yval_f2 = [m['mpct_f2'] for m in female_pct_matrices]
            yval_f3 = [m['mpct_f3'] for m in female_pct_matrices]
            yval_m1 = [m['mpct_m1'] for m in female_pct_matrices]
            yval_m2 = [m['mpct_m2'] for m in female_pct_matrices]
            yval_m3 = [m['mpct_m3'] for m in female_pct_matrices]



        if plottype == 'gender number':

            mean_matrices = [m.results_matrix.loc[:,'mean_f1':'mean_m3'] for
                             m in self.mlist]
            yval_f1 = [m['mean_f1'] for m in mean_matrices]
            yval_f2 = [m['mean_f2'] for m in mean_matrices]
            yval_f3 = [m['mean_f3'] for m in mean_matrices]
            yval_m1 = [m['mean_m1'] for m in mean_matrices]
            yval_m2 = [m['mean_m2'] for m in mean_matrices]
            yval_m3 = [m['mean_m3'] for m in mean_matrices]


        if plottype == 'parameter sweep gender percentage':

            female_sweep_matrices = [m.parameter_sweep_results for m in
                                   self.mlist]

            yval_f1 = [m['mpct_f1'] for m in female_sweep_matrices]
            yval_f2 = [m['mpct_f2'] for m in female_sweep_matrices]
            yval_f3 = [m['mpct_f3'] for m in female_sweep_matrices]
            yval_m1 = [m['mpct_m1'] for m in female_sweep_matrices]
            yval_m2 = [m['mpct_m2'] for m in female_sweep_matrices]
            yval_m3 = [m['mpct_m3'] for m in female_sweep_matrices]

        if plottype == 'parameter sweep gender number':

            female_sweep_matrices = [m.parameter_sweep_results for m in
                                   self.mlist]

            yval_f1 = [m['mean_f1'] for m in female_sweep_matrices]
            yval_f2 = [m['mean_f2'] for m in female_sweep_matrices]
            yval_f3 = [m['mean_f3'] for m in female_sweep_matrices]
            yval_m1 = [m['mean_m1'] for m in female_sweep_matrices]
            yval_m2 = [m['mean_m2'] for m in female_sweep_matrices]
            yval_m3 = [m['mean_m3'] for m in female_sweep_matrices]

        if plottype == 'parameter sweep probability':
            pass

        #END BLOCK


        # setup empirical bounds

        if intervals == 'empirical':

            if plottype == 'probability proportion':

                upper_f1 = yval_f1
                upper_f2 = yval_f2
                upper_f3 = yval_f3
                upper_m1 = yval_m1
                upper_m2 = yval_m2
                upper_m3 = yval_m3

                lower_f1 = yval_f1
                lower_f2 = yval_f2
                lower_f3 = yval_f3
                lower_m1 = yval_m1
                lower_m2 = yval_m2
                lower_m3 = yval_m3

            if plottype == 'gender proportion':

                upper_f1 = [m['pf1_975'] for m in female_pct_matrices]
                upper_f2 = [m['pf2_975'] for m in female_pct_matrices]
                upper_f3 = [m['pf3_975'] for m in female_pct_matrices]
                upper_m1 = [m['pm1_975'] for m in female_pct_matrices]
                upper_m2 = [m['pm2_975'] for m in female_pct_matrices]
                upper_m3 = [m['pm3_975'] for m in female_pct_matrices]

                # print('band', upper_f2)
                # print('\n')
                lower_f1 = [m['pf1_025'] for m in female_pct_matrices]
                lower_f2 = [m['pf2_025'] for m in female_pct_matrices]
                lower_f3 = [m['pf3_025'] for m in female_pct_matrices]
                lower_m1 = [m['pm1_025'] for m in female_pct_matrices]
                lower_m2 = [m['pm2_025'] for m in female_pct_matrices]
                lower_m3 = [m['pm3_025'] for m in female_pct_matrices]


            if plottype == 'gender number':
                u_matrices = [m.results_matrix.loc[:, 'f1_975':'m3_975'] for
                              m in
                            self.mlist]

                upper_f1 = [m['f1_975'] for m in u_matrices]
                upper_f2 = [m['f2_975'] for m in u_matrices]
                upper_f3 = [m['f3_975'] for m in u_matrices]
                upper_m1 = [m['m1_975'] for m in u_matrices]
                upper_m2 = [m['m2_975'] for m in u_matrices]
                upper_m3 = [m['m3_975'] for m in u_matrices]

                l_matrices = [m.results_matrix.loc[:, 'f1_025':'m3_025'] for
                              m in
                            self.mlist]

                lower_f1 = [m['f1_025'] for m in l_matrices]
                lower_f2 = [m['f2_025'] for m in l_matrices]
                lower_f3 = [m['f3_025'] for m in l_matrices]
                lower_m1 = [m['m1_025'] for m in l_matrices]
                lower_m2 = [m['m2_025'] for m in l_matrices]
                lower_m3 = [m['m3_025'] for m in l_matrices]

            if plottype == 'parameter sweep gender percentage':

                upper_f1 = [m['pf1_975'] for m in female_sweep_matrices]
                upper_f2 = [m['pf2_975'] for m in female_sweep_matrices]
                upper_f3 = [m['pf3_975'] for m in female_sweep_matrices]
                upper_m1 = [m['pm1_975'] for m in female_sweep_matrices]
                upper_m2 = [m['pm2_975'] for m in female_sweep_matrices]
                upper_m3 = [m['pm3_975'] for m in female_sweep_matrices]

                lower_f1 = [m['pf1_025'] for m in female_sweep_matrices]
                lower_f2 = [m['pf2_025'] for m in female_sweep_matrices]
                lower_f3 = [m['pf3_025'] for m in female_sweep_matrices]
                lower_m1 = [m['pm1_025'] for m in female_sweep_matrices]
                lower_m2 = [m['pm2_025'] for m in female_sweep_matrices]
                lower_m3 = [m['pm3_025'] for m in female_sweep_matrices]

            if plottype == 'parameter sweep gender number':

                upper_f1 = [m['f1_975'] for m in female_sweep_matrices]
                upper_f2 = [m['f2_975'] for m in female_sweep_matrices]
                upper_f3 = [m['f3_975'] for m in female_sweep_matrices]
                upper_m1 = [m['m1_975'] for m in female_sweep_matrices]
                upper_m2 = [m['m2_975'] for m in female_sweep_matrices]
                upper_m3 = [m['m3_975'] for m in female_sweep_matrices]

                lower_f1 = [m['f1_025'] for m in female_sweep_matrices]
                lower_f2 = [m['f2_025'] for m in female_sweep_matrices]
                lower_f3 = [m['f3_025'] for m in female_sweep_matrices]
                lower_m1 = [m['m1_025'] for m in female_sweep_matrices]
                lower_m2 = [m['m2_025'] for m in female_sweep_matrices]
                lower_m3 = [m['m3_025'] for m in female_sweep_matrices]



        # setup standard bounds


        if intervals == 'standard':

            if plottype == 'probability proportion':

                upper_f1 = yval_f1
                upper_f2 = yval_f2
                upper_f3 = yval_f3
                upper_m1 = yval_m1
                upper_m2 = yval_m2
                upper_m3 = yval_m3
                lower_f1 = yval_f1
                lower_f2 = yval_f2
                lower_f3 = yval_f3
                lower_m1 = yval_m1
                lower_m2 = yval_m2
                lower_m3 = yval_m3

            if plottype == 'gender proportion':

                upper_f1 = list(map(add, [y for y in yval_f1], [1.96 * m[
                    'spct_f1'] for m in female_pct_matrices]))

                upper_f1 = np.where(np.array(upper_f1)>1,1,np.array(upper_f1))

                upper_f2 = list(map(add, [y for y in yval_f2], [1.96 * m[
                    'spct_f2'] for m in female_pct_matrices]))

                upper_f2 = np.where(np.array(upper_f2)>1,1,np.array(upper_f2))

                upper_f3 = list(map(add, [y for y in yval_f3], [1.96 * m[
                    'spct_f3'] for m in female_pct_matrices]))

                upper_f3 = np.where(np.array(upper_f3)>1,1,np.array(upper_f3))

                upper_m1 = list(map(add, [y for y in yval_m1], [1.96 * m[
                    'spct_m1']for m in female_pct_matrices]))

                upper_m1 = np.where(np.array(upper_m1)>1,1,np.array(upper_m1))

                upper_m2 = list(map(add, [y for y in yval_m2], [1.96 * m[
                    'spct_m2']for m in female_pct_matrices]))

                upper_m2 = np.where(np.array(upper_m2)>1,1,np.array(upper_m2))

                upper_m3 = list(map(add, [y for y in yval_m3], [1.96 * m[
                    'spct_m3']for m in female_pct_matrices]))

                upper_m3 = np.where(np.array(upper_m3)>1,1,np.array(upper_m3))


                lower_f1 = list(map(sub, [y for y in yval_f1], [1.96 * m[
                    'spct_f1'] for m in female_pct_matrices]))

                lower_f1 = np.where(np.array(lower_f1)<0,0,np.array(lower_f1))

                lower_f2 = list(map(sub, [y for y in yval_f2], [1.96 * m[
                    'spct_f2'] for m in female_pct_matrices]))

                lower_f2 = np.where(np.array(lower_f2)<0,0,np.array(lower_f2))

                lower_f3 = list(map(sub, [y for y in yval_f3], [1.96 * m[
                    'spct_f3'] for m in female_pct_matrices]))

                lower_f3 = np.where(np.array(lower_f3)<0,0,np.array(lower_f3))


                lower_m1 = list(map(sub, [y for y in yval_m1], [1.96 * m[
                    'spct_m1'] for m in female_pct_matrices]))

                lower_m1 = np.where(np.array(lower_m1)<0,0,np.array(lower_m1))

                lower_m2 = list(map(sub, [y for y in yval_m2], [1.96 * m[
                    'spct_m2'] for m in female_pct_matrices]))

                lower_m2 = np.where(np.array(lower_m2)<0,0,np.array(lower_m2))

                lower_m3 = list(map(sub, [y for y in yval_m3], [1.96 * m[
                    'spct_m3'] for m in female_pct_matrices]))

                lower_m3 = np.where(np.array(lower_m3)<0,0,np.array(lower_m3))

            if plottype == 'gender number':

                std_matrices = [m.results_matrix[:, 'std_f1':'std_m3'] for m in
                                self.mlist]

                upper_f1 = list(map(add, [y for y in yval_f1], [1.96 * m[
                    'spct_f1'] for m in std_matrices]))

                upper_f2 = list(map(add, [y for y in yval_f2], [1.96 * m[
                    'spct_f2'] for m in std_matrices]))

                upper_f3 = list(map(add, [y for y in yval_f3], [1.96 * m[
                    'spct_f3'] for m in std_matrices]))

                upper_m1 = list(map(add, [y for y in yval_m1], [1.96 * m[
                    'spct_m1'] for m in std_matrices]))

                upper_m2 = list(map(add, [y for y in yval_m2], [1.96 * m[
                    'spct_m2'] for m in std_matrices]))

                upper_m3 = list(map(add, [y for y in yval_m3], [1.96 * m[
                    'spct_m3'] for m in std_matrices]))

                lower_f1 = list(map(sub, [y for y in yval_f1], [1.96 * m[
                    'spct_f1'] for m in std_matrices]))

                lower_f2 = list(map(sub, [y for y in yval_f2], [1.96 * m[
                    'spct_f2'] for m in std_matrices]))

                lower_f3 = list(map(sub, [y for y in yval_f3], [1.96 * m[
                    'spct_f3'] for m in std_matrices]))

                lower_m1 = list(map(sub, [y for y in yval_m1], [1.96 * m[
                    'spct_m1'] for m in std_matrices]))

                lower_m2 = list(map(sub, [y for y in yval_m2], [1.96 * m[
                    'spct_m2'] for m in std_matrices]))

                lower_m3 = list(map(sub, [y for y in yval_m3], [1.96 * m[
                    'spct_m3'] for m in std_matrices]))

            if plottype == 'parameter sweep gender percentage ':

                upper_f1 = list(map(add, [y for y in yval_f1], [1.96 * m[
                    'spct_f1'] for m in female_sweep_matrices]))

                upper_f1 = np.where(np.array(upper_f1)>1,1,np.array(upper_f1))

                upper_f2 = list(map(add, [y for y in yval_f2], [1.96 * m[
                    'spct_f2'] for m in female_sweep_matrices]))

                upper_f2 = np.where(np.array(upper_f2)>1,1,np.array(upper_f2))

                upper_f3 = list(map(add, [y for y in yval_f3], [1.96 * m[
                    'spct_f3'] for m in female_sweep_matrices]))

                upper_f3 = np.where(np.array(upper_f3)>1,1,np.array(upper_f3))

                upper_m1 = list(map(add, [y for y in yval_m1], [1.96 * m[
                    'spct_m1']for m in female_sweep_matrices]))

                upper_m1 = np.where(np.array(upper_m1)>1,1,np.array(upper_m1))

                upper_m2 = list(map(add, [y for y in yval_m2], [1.96 * m[
                    'spct_m2']for m in female_sweep_matrices]))

                upper_m2 = np.where(np.array(upper_m2)>1,1,np.array(upper_m2))

                upper_m3 = list(map(add, [y for y in yval_m3], [1.96 * m[
                    'spct_m3']for m in female_sweep_matrices]))

                upper_m3 = np.where(np.array(upper_m3)>1,1,np.array(upper_m3))


                lower_f1 = list(map(sub, [y for y in yval_f1], [1.96 * m[
                    'spct_f1'] for m in female_sweep_matrices]))

                lower_f1 = np.where(np.array(lower_f1)<0,0,np.array(lower_f1))

                lower_f2 = list(map(sub, [y for y in yval_f2], [1.96 * m[
                    'spct_f2'] for m in female_sweep_matrices]))

                lower_f2 = np.where(np.array(lower_f2)<0,0,np.array(lower_f2))

                lower_f3 = list(map(sub, [y for y in yval_f3], [1.96 * m[
                    'spct_f3'] for m in female_sweep_matrices]))

                lower_f3 = np.where(np.array(lower_f3)<0,0,np.array(lower_f3))


                lower_m1 = list(map(sub, [y for y in yval_m1], [1.96 * m[
                    'spct_m1'] for m in female_sweep_matrices]))

                lower_m1 = np.where(np.array(lower_m1)<0,0,np.array(lower_m1))

                lower_m2 = list(map(sub, [y for y in yval_m2], [1.96 * m[
                    'spct_m2'] for m in female_sweep_matrices]))

                lower_m2 = np.where(np.array(lower_m2)<0,0,np.array(lower_m2))

                lower_m3 = list(map(sub, [y for y in yval_m3], [1.96 * m[
                    'spct_m3'] for m in female_sweep_matrices]))

                lower_m3 = np.where(np.array(lower_m3)<0,0,np.array(lower_m3))


            if plottype == 'parameter sweep gender number':

                upper_f1 = list(map(add, [y for y in yval_f1], [1.96 * m[
                    'std_f1'] for m in female_sweep_matrices]))

                upper_f2 = list(map(add, [y for y in yval_f2], [1.96 * m[
                    'std_f2'] for m in female_sweep_matrices]))

                upper_f3 = list(map(add, [y for y in yval_f3], [1.96 * m[
                    'std_f3'] for m in female_sweep_matrices]))

                upper_m1 = list(map(add, [y for y in yval_m1], [1.96 * m[
                    'std_m1']for m in female_sweep_matrices]))

                upper_m2 = list(map(add, [y for y in yval_m2], [1.96 * m[
                    'std_m2']for m in female_sweep_matrices]))

                upper_m3 = list(map(add, [y for y in yval_m3], [1.96 * m[
                    'std_m3']for m in female_sweep_matrices]))


                lower_f1 = list(map(sub, [y for y in yval_f1], [1.96 * m[
                    'std_f1'] for m in female_sweep_matrices]))

                lower_f1 = np.where(np.array(lower_f1)<0,0,np.array(lower_f1))

                lower_f2 = list(map(sub, [y for y in yval_f2], [1.96 * m[
                    'std_f2'] for m in female_sweep_matrices]))

                lower_f2 = np.where(np.array(lower_f2)<0,0,np.array(lower_f2))

                lower_f3 = list(map(sub, [y for y in yval_f3], [1.96 * m[
                    'std_f3'] for m in female_sweep_matrices]))

                lower_f3 = np.where(np.array(lower_f3)<0,0,np.array(lower_f3))


                lower_m1 = list(map(sub, [y for y in yval_m1], [1.96 * m[
                    'std_m1'] for m in female_sweep_matrices]))

                lower_m1 = np.where(np.array(lower_m1)<0,0,np.array(lower_m1))

                lower_m2 = list(map(sub, [y for y in yval_m2], [1.96 * m[
                    'std_m2'] for m in female_sweep_matrices]))

                lower_m2 = np.where(np.array(lower_m2)<0,0,np.array(lower_m2))

                lower_m3 = list(map(sub, [y for y in yval_m3], [1.96 * m[
                    'std_m3'] for m in female_sweep_matrices]))

                lower_m3 = np.where(np.array(lower_m3)<0,0,np.array(lower_m3))


        levels = ['f1', 'f2', 'f3', 'm1', 'm2', 'm3']
        yvals = [yval_f1, yval_f2, yval_f3, yval_m1, yval_m2, yval_m3]
        upper_fill = [upper_f1, upper_f2, upper_f3, upper_m1, upper_m2,
                      upper_m3]
        lower_fill = [lower_f1, lower_f2, lower_f3, lower_m1, lower_m2,
                      lower_m3]

        plots = []

        x_data = np.asarray(xval)
        band_x = np.append(x_data, x_data[::-1])

        for key, val in enumerate(levels):

            plots.append(figure(title=titles[key],
                                x_axis_label=xlabels[key],
                                y_axis_label = ylabels[key],
                                width=width_,
                                height=height_))


        for k,v in enumerate(self.mlist):

            for i, p in enumerate(plots):
                p.line(xval, yvals[i][k],
                       line_width=line_width,
                       line_color=linecolor[k])

                band_y = np.append(lower_fill[i][k], upper_fill[i][k][::-1])

                p.patch(band_x,
                        band_y,
                        color=linecolor[k],
                        fill_alpha=transparency)

        if target_plot == True:

            for i, p in enumerate(plots):

                p.line(xval,
                       target,
                       line_color=target_color,
                       line_width=target_plot_linewidth,
                       legend=target_plot_legend_label,
                       line_dash=[6,6])

        if percent_line_plot == True:

            for i, p in enumerate(plots):
                p.line(xval, percent_line_value,
                       line_color=color_percent_line,
                       line_width=percent_linewidth,
                       line_dash=[2,2],
                       legend=percent_legend_label)

        grid = gridplot([[plots[0], plots[1], plots[2]],
                         [plots[3], plots[4], plots[5]]])
        return(grid)


    def testfunc(self):

        for mod in self.mlist:
            mod.run_multiple(5)

        yval = [m.mean_matrix['gendprop'] for m in self.mlist]
        return(yval)



