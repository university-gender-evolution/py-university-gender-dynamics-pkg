__author__ = 'krishnab'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# CONSTANTS

#line_colors = ['#7fc97f', '#beaed4', '#fdc086','#386cb0','#f0027f','#ffff99']

class Comparison():

    def __init__(self, model_list):
        self.name = 'All Models'
        self.label = 'All Models'
        self.mlist = model_list

    def plot_comparison_gender_proportion(self, xlabel, ylabel, title, txt,
                                          target,
                                          number_of_runs=10):

        ## This function will execute gender proportion comparisons for all models

        ## Color list

        line_colors = ['#7fc97f', '#beaed4', '#fdc086','#386cb0','#f0027f','#ffff99']

        for mod in self.mlist:
            mod.run_multiple(number_of_runs)

        ## Create plot array and execute plot

        for k,v in enumerate(self.mlist):

            plt.plot(range(self.mlist[k].duration), self.mlist[k].mean_matrix['gendprop'],label = self.mlist[k].label, linewidth=2.0, color=line_colors[k])

            plt.plot(range(self.mlist[k].duration), self.mlist[k].mean_matrix['gendprop'])
            plt.fill_between(range(self.mlist[k].duration), self.mlist[k].mean_matrix[
                'gendprop'] +
                                1.96*self.mlist[k].std_matrix[
            'gendprop'], self.mlist[k].mean_matrix['gendprop'] - 1.96*self.mlist[k].std_matrix[
            'gendprop'], color = line_colors[k], alpha=0.5)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axhline(target, color='r')
        plt.legend(loc='upper right', shadow=True)
        plt.text(0.2, 0.2, txt)
        plt.show()

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

    def plot_comparison_department_size(self, number_of_runs=10):

        ## This function will execute gender proportion comparisons for all models

        ## Color list

        line_colors = ['#7fc97f', '#beaed4', '#fdc086','#386cb0','#f0027f','#ffff99']

        for mod in self.mlist:
            mod.run_multiple(number_of_runs)

        ## Create plot array and execute plot

        for k,v in enumerate(self.mlist):

            plt.plot(range(self.mlist[k].duration), self.mlist[k].dept_size_matrix['mean'], color=line_colors[k],label = self.mlist[k].label, linewidth=2.0)

            plt.plot(range(self.mlist[k].duration), self.mlist[k].dept_size_matrix['mean'])
            plt.fill_between(range(self.mlist[k].duration), self.mlist[k].dept_size_matrix[
                'mean'] +
                                1.96*self.mlist[k].dept_size_matrix[
            'std'], self.mlist[k].dept_size_matrix['mean'] - 1.96*self.mlist[k].dept_size_matrix[
            'std'], color = line_colors[k], alpha=0.5)

        plt.title('Department Size over Time: ' + self.name)
        plt.xlabel('Years')
        plt.ylabel('Total Department Size')
        plt.legend(loc='upper right', shadow=True)
        plt.show()



    def plot_comparison_detail(self, number_of_runs):


        # create color list

        line_colors = ['#7fc97f', '#beaed4', '#fdc086','#386cb0','#f0027f','#ffff99']
        plot_titles = ['extra', 'Female Level 1', 'Female level 2', 'Female level 3', 'Male level 1', 'Male level 2', 'Male level 3']
        plot_y_axis_titles = ['extra', 'Number of Females', 'Number of Females', 'Number of Females', 'Number of Males', 'Number of Males', 'Number of Males']

        # run all models and generate result matrices. I run each model so I can guarantee that they all have the same duration.

        for mod in self.mlist:
            mod.run_multiple(number_of_runs)



        f, axarr = plt.subplots(nrows = 2, ncols=3)
        f.suptitle("Department Male/Female Counts by Level")

        for k,v in enumerate(self.mlist):

            axarr[0,0].plot(range(self.mlist[k].duration), self.mlist[k].mean_matrix['f1'], color=line_colors[k], label = self.mlist[k].label, linewidth=2.0)
            axarr[0,0].set_title('Female level 1')
            axarr[0,0].set_xlabel('Years')
            axarr[0,0].set_ylabel('Number of Females')
            axarr[0,0].fill_between(range(self.mlist[k].duration), self.mlist[k].mean_matrix['f1'] +
                                    1.96*self.mlist[k].std_matrix[
                'f1'], self.mlist[k].mean_matrix['f1'] - 1.96*self.mlist[k].std_matrix[
                'f1'], alpha=0.5, color = line_colors[k], facecolor= line_colors[k])
            axarr[0,0].legend(loc='upper right', shadow=True)


            axarr[0,1].plot(range(self.mlist[k].duration), self.mlist[k].mean_matrix['f2'], color=line_colors[k])
            axarr[0,1].set_title('Female level 2')
            axarr[0,1].set_xlabel('Years')
            axarr[0,1].set_ylabel('Number of Females')
            axarr[0,1].fill_between(range(self.mlist[k].duration), self.mlist[k].mean_matrix['f2'] +
                                    1.96*self.mlist[k].std_matrix[
                'f2'], self.mlist[k].mean_matrix['f2'] - 1.96*self.mlist[k].std_matrix[
                'f2'], alpha=0.5, color = line_colors[k], facecolor= line_colors[k] )

            axarr[0,2].plot(range(self.mlist[k].duration), self.mlist[k].mean_matrix['f3'], color=line_colors[k])
            axarr[0,2].set_title('Female level 3')

            axarr[0,2].set_xlabel('Years')
            axarr[0,2].set_ylabel('Number of Females')
            axarr[0,2].fill_between(range(self.mlist[k].duration), self.mlist[k].mean_matrix['f3'] +
                                    1.96*self.mlist[k].std_matrix[
                'f3'], self.mlist[k].mean_matrix['f3'] - 1.96*self.mlist[k].std_matrix[
                'f3'], alpha=0.5,  color = line_colors[k], facecolor= line_colors[k])

            axarr[1,0].plot(range(self.mlist[k].duration), self.mlist[k].mean_matrix['m1'],color=line_colors[k])
            axarr[1,0].set_title('Male level 1')
            axarr[1,0].set_xlabel('Years')
            axarr[1,0].set_ylabel('Number of Males')
            axarr[1,0].fill_between(range(self.mlist[k].duration), self.mlist[k].mean_matrix['m1'] +
                                    1.96*self.mlist[k].std_matrix[
                'm1'], self.mlist[k].mean_matrix['m1'] - 1.96*self.mlist[k].std_matrix[
                'm1'], alpha=0.5, color = line_colors[k], facecolor= line_colors[k])

            axarr[1,1].plot(range(self.mlist[k].duration), self.mlist[k].mean_matrix['m2'], color=line_colors[k])
            axarr[1,1].set_title('Male level 2')
            axarr[1,1].set_xlabel('Years')
            axarr[1,1].set_ylabel('Number of Males')
            axarr[1,1].fill_between(range(self.mlist[k].duration), self.mlist[k].mean_matrix['m2'] +
                                    1.96*self.mlist[k].std_matrix[
                'm2'], self.mlist[k].mean_matrix['m2'] - 1.96*self.mlist[k].std_matrix[
                'm2'], alpha=0.5,  color = line_colors[k], facecolor= line_colors[k] )

            axarr[1,2].plot(range(self.mlist[k].duration), self.mlist[k].mean_matrix['m3'], color=line_colors[k])
            axarr[1,2].set_title('Male level 3')
            axarr[1,2].set_xlabel('Years')
            axarr[1,2].set_ylabel('Number of Males')
            axarr[1,2].fill_between(range(self.mlist[k].duration), self.mlist[k].mean_matrix['m3'] +
                                    1.96*self.mlist[k].std_matrix[
                'm3'], self.mlist[k].mean_matrix['m3'] - 1.96*self.mlist[k].std_matrix[
                'm3'], alpha=0.5,  color = line_colors[k], facecolor= line_colors[k] )
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        plt.show()





    def plot_parameter_sweep_gender_proportion(self, number_of_runs, param, llim, ulim, number_of_steps):

        ## This function will execute gender proportion comparisons for all models

        ## Color list



        for mod in self.mlist:
            mod.run_parameter_sweep(number_of_runs, param, llim, ulim, number_of_steps)

        ## Create plot array and execute plot

        for k,v in enumerate(self.mlist):

            plot_array = self.mlist[k].parameter_sweep_array[0]
            plt.plot(plot_array[:,0], plot_array[:,1], label = self.mlist[k].label, linewidth=2.0, color=line_colors[k])
            plt.fill_between(plot_array[:,0], plot_array[:,1] +
                                    1.96*plot_array[:,2], plot_array[:,1] -
                             1.96*plot_array[:,2], alpha=0.5,  color = line_colors[k], facecolor= line_colors[k])

        plt.title('Parameter Sweep for Gender Proportion over ' + str(self.mlist[0].duration) + ' years')
        plt.xlabel(param)
        plt.ylabel('Percentage of the Department that is Women')
        plt.legend(loc='upper right', shadow=True)
        plt.show()





    def plot_parameter_sweep_detail(self, number_of_runs, param, llim, ulim, number_of_steps):



        # create color list

        line_colors = ['#7fc97f', '#beaed4', '#fdc086','#386cb0','#f0027f','#ffff99']
        plot_titles = ['extra', 'Female Level 1', 'Female level 2', 'Female level 3', 'Male level 1', 'Male level 2', 'Male level 3']
        plot_y_axis_titles = ['extra', 'Number of Females', 'Number of Females', 'Number of Females', 'Number of Males', 'Number of Males', 'Number of Males']

        # run all models and generate result matrices. I run each model so I can guarantee that they all have the same duration.

        for mod in self.mlist:
            mod.run_parameter_sweep(number_of_runs, param, llim, ulim, number_of_steps)

        f, axarr = plt.subplots(nrows = 2, ncols=3)
        f.suptitle("Parameter sweep")



        for k,v in enumerate(self.mlist):

            array_list = self.mlist[k].parameter_sweep_array

            axarr[0,0].plot(array_list[1][:,0], array_list[1][:,1],label=self.mlist[k].label, linewidth=2.0, color=line_colors[k])
            axarr[0,0].set_title('Female level 1')
            axarr[0,0].set_xlabel(array_list[7])
            axarr[0,0].set_ylabel('Number of Females')
            axarr[0,0].fill_between(array_list[1][:,0], array_list[1][:,1] +
                                    1.96*array_list[1][:,2], array_list[1][:,1] -
                                    1.96*array_list[1][:,2], alpha=0.5,color = line_colors[k], facecolor= line_colors[k])
            axarr[0,0].legend(loc='upper right', shadow=True)

            axarr[0,1].plot(array_list[2][:,0], array_list[2][:,1], color=line_colors[k])
            axarr[0,1].set_title('Female level 2')
            axarr[0,1].set_xlabel(array_list[7])
            axarr[0,1].set_ylabel('Number of Females')
            axarr[0,1].fill_between(array_list[2][:,0], array_list[2][:,1] +
                                    1.96*array_list[2][:,2], array_list[2][:,1] -
                                    1.96*array_list[2][:,2], alpha=0.5,color = line_colors[k], facecolor= line_colors[k])

            axarr[0,2].plot(array_list[3][:,0], array_list[3][:,1], color=line_colors[k])
            axarr[0,2].set_title('Female level 3')
            axarr[0,2].set_xlabel(array_list[7])
            axarr[0,2].set_ylabel('Number of Females')
            axarr[0,2].fill_between(array_list[3][:,0], array_list[3][:,1] +
                                    1.96*array_list[3][:,2], array_list[3][:,1] -
                                    1.96*array_list[3][:,2], alpha=0.5,color = line_colors[k], facecolor= line_colors[k])

            axarr[1,0].plot(array_list[4][:,0], array_list[4][:,1], color=line_colors[k])
            axarr[1,0].set_title('Male level 1')
            axarr[1,0].set_xlabel(array_list[7])
            axarr[1,0].set_ylabel('Number of Males')
            axarr[1,0].fill_between(array_list[4][:,0], array_list[4][:,1] +
                                    1.96*array_list[4][:,2], array_list[4][:,1] -
                                    1.96*array_list[4][:,2], alpha=0.5,color = line_colors[k], facecolor= line_colors[k])

            axarr[1,1].plot(array_list[5][:,0], array_list[5][:,1], color=line_colors[k])
            axarr[1,1].set_title('Male level 2')
            axarr[1,1].set_xlabel(array_list[7])
            axarr[1,1].set_ylabel('Number of Males')
            axarr[1,1].fill_between(array_list[5][:,0], array_list[5][:,1] +
                                    1.96*array_list[5][:,2], array_list[5][:,1] -
                                    1.96*array_list[5][:,2], alpha=0.5,color = line_colors[k], facecolor= line_colors[k])

            axarr[1,2].plot(array_list[6][:,0], array_list[6][:,1], color=line_colors[k])
            axarr[1,2].set_title('Male level 3')
            axarr[1,2].set_xlabel(array_list[7])
            axarr[1,2].set_ylabel('Number of Males')
            axarr[1,2].fill_between(array_list[6][:,0], array_list[6][:,1] +
                                    1.96*array_list[6][:,2], array_list[6][:,1] -
                                    1.96*array_list[6][:,2], alpha=0.5,color = line_colors[k], facecolor= line_colors[k])
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()



    # def plot_comparison_empirical_probability_gender_proportion(self, number_of_runs, param, prof_group, llim,
    #                 ulim, num_of_steps, target):
    #
    #
    #     for mod in self.mlist:
    #         mod.run_probability_analysis_parameter_sweep_gender_detail(number_of_runs, param, prof_group, llim,
    #                 ulim, num_of_steps, target)
    #
    #
    #     for k,v in enumerate(self.mlist):
    #
    #         plot_array = self.mlist[k].last_empirical_probability_detail
    #         plt.plot(plot_array['param'], plot_array['probability'], label = self.mlist[k].label)
    #
    #
    #     plt.title('Probability plot for Parameter Sweep on ' + param + ' for ' + str(self.mlist[0].duration) + ' years')
    #     plt.xlabel(param)
    #     plt.ylabel('Probability of Target Value or Greater')
    #     plt.legend(loc='upper right', shadow=True)
    #     plt.show()


    def plot_comparison_empirical_probability_gender_proportion(self,xlabel,
                                                                ylabel,
                                                                title,
                                                                txt,
                                                                target,
                                                                number_of_runs):

        ## This function will execute gender proportion comparisons for all models

        ## Color list

        line_colors = ['#7fc97f', '#beaed4', '#fdc086','#386cb0','#f0027f','#ffff99']

        for mod in self.mlist:
            mod.run_probability_analysis_gender_proportion(number_of_runs, target)

        ## Create plot array and execute plot

        for k,v in enumerate(self.mlist):

            plt.plot(self.mlist[k].probability_matrix['Year'], self.mlist[k].probability_matrix['Probability'], color=line_colors[k],label = self.mlist[k].label, linewidth=2.0)


        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='upper right', shadow=True)
        plt.axhline(0.5, color='r')
        plt.text(0.2, 0.2, txt)
        plt.show()

    def plot_comparison_female_male_numbers(self, xlabel, ylabel, title, txt,
                                          target, number_of_runs=10):


        ## This function will execute gender proportion comparisons for all models

        ## Color list

        colors_women = ['#7fc97f', '#beaed4', '#fdc086']

        colors_men = ['#386cb0', '#f69967', '#ffff99']



        for mod in self.mlist:
            mod.run_multiple(number_of_runs)

        ## Create plot array and execute plot

        for k, v in enumerate(self.mlist):
            total_faculty = 0
            plt.plot(range(self.mlist[k].duration),
                     sum(list([self.mlist[k].mean_matrix['f1'],
                               self.mlist[k].mean_matrix['f2'],
                               self.mlist[k].mean_matrix['f3']])),
                     color= colors_women[k], label = self.mlist[k].label +
                     ' female', linewidth=3)

            plt.plot(range(self.mlist[k].duration),
                     sum(list([self.mlist[k].mean_matrix['m1'],
                               self.mlist[k].mean_matrix['m2'],
                               self.mlist[k].mean_matrix['m3']])),
                     color=colors_men[k], label = self.mlist[k].label +
                     ' male', linewidth=3)

            total_faculty = self.mlist[k].mean_matrix['f1'] \
                            + self.mlist[k].mean_matrix['f2'] \
                            + self.mlist[k].mean_matrix['f3'] \
                            + self.mlist[k].mean_matrix['m1'] \
                            + self.mlist[k].mean_matrix['m2'] \
                            + self.mlist[k].mean_matrix['m3']

        plt.plot(range(self.mlist[1].duration), np.round(target *
                                                         total_faculty),
                 color='r', label='Target for women', linewidth=3)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='upper left', shadow=True)
        plt.text(0.2, 0.2, txt)
        plt.show()
