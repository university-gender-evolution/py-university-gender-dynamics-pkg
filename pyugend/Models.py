"""
Base Model Module
-----------------

This is the base class for all model modules. This class does not contain an particular model but it does include all of the functions to run a model, capture model statistics, and visualize model data.

"""

__author__ = 'krishnab'
__version__ = '0.1.0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.palettes import Viridis3
from bokeh.plotting import figure
from bokeh.charts import defaults

## Initialize Constants

PROFESSOR_LEVEL_NAMES = list(['f1n', 'f2n', 'f3n', 'm1n', 'm2n', 'm3n'])

PROBABILITY_ARRAY_COLUMN_NAMES = list(
    ['param', 'prof_group_mean', 'probability'])

LEVELS = list(['f1',
               'f2',
               'f3',
               'm1',
               'm2',
               'm3'])

MODEL_RUN_COLUMNS = list(['f1',
                           'f2',
                           'f3',
                           'm1',
                           'm2',
                           'm3',
                           'vac_3',
                           'vac_2',
                           'vac_1',
                           'prom1',
                           'prom2',
                           'gendprop',
                           'unfilled',
                           'dept_size',
                           'f_hire_3',
                           'm_hire_3',
                           'f_hire_2',
                           'm_hire_2',
                           'f_hire_1',
                           'm_hire_1',
                           'f_prom_3',
                           'm_prom_3',
                           'f_prom_2',
                           'm_prom_2',
                           'f_prom_1',
                           'm_prom_1'])

RESULTS_COLUMNS = list([ 'year', 'mean_f1', 'mean_f2',
                                   'mean_f3', 'mean_m1',
                                   'mean_m2', 'mean_m3',
                                   'mean_vac_3', 'mean_vac_2',
                                   'mean_vac_1', 'mean_prom1',
                                   'mean_prom2', 'mean_gendprop',
                                   'mean_unfilled','mean_dept_size',
                                   'mean_f_hire_3','mean_m_hire_3',
                                   'mean_f_hire_2','mean_m_hire_2',
                                   'mean_f_hire_1','mean_m_hire_1',
                                   'mean_f_prom_3','mean_m_prom_3',
                                   'mean_f_prom_2','mean_m_prom_2',
                                   'mean_f_prom_1','mean_m_prom_1',
                                   'std_f1', 'std_f2',
                                   'std_f3', 'std_m1',
                                   'std_m2', 'std_m3',
                                   'std_vac_3', 'std_vac_2',
                                   'std_vac_1', 'std_prom1',
                                   'std_prom2', 'std_gendprop',
                                   'std_unfilled','std_dept_size',
                                   'std_f_hire_3','std_m_hire_3',
                                   'std_f_hire_2','std_m_hire_2',
                                   'std_f_hire_1','std_m_hire_1',
                                   'std_f_prom_3','std_m_prom_3',
                                   'std_f_prom_2','std_m_prom_2',
                                   'std_f_prom_1','std_m_prom_1',
                                   'f1_025', 'f2_025',
                                   'f3_025', 'm1_025',
                                   'm2_025', 'm3_025', 'vac_3_025',
                                    'vac_2_025', 'vac_1_025',
                                    'prom1_025', 'prom2_025',
                                    'gendprop_025','unfilled_025',
                                    'dept_size_025', 'f_hire_3_025',
                                    'm_hire_3_025', 'f_hire_2_025',
                                    'm_hire_2_025','f_hire_1_025',
                                    'm_hire_1_025', 'f_prom_3_025',
                                    'm_prom_3_025', 'f_prom_2_025',
                                    'm_prom_2_025','f_prom_1_025',
                                    'm_prom_1_025', 'f1_975',
                                    'f2_975','f3_975',
                                    'm1_975', 'm2_975',
                                    'm3_975', 'vac_3_975',
                                    'vac_2_975', 'vac_1_975',
                                    'prom_1_975', 'prom2_975',
                                    'genderprop_975', 'unfilled_975',
                                    'dept_size_975','f_hire_3_975',
                                    'm_hire_3_975', 'f_hire_2_975',
                                    'm_hire_2_975', 'f_hire_1_975',
                                    'm_hire_1_975','f_prom_3_975',
                                    'm_prom_3_975', 'f_prom_2_975',
                                    'm_prom_2_975', 'f_prom_1_975',
                                    'm_prom_1_975'
                         ])

FEMALE_MATRIX_COLUMNS = list(['year',
                              'mpct_f1',
                              'spct_f1',
                              'mpct_f2',
                              'spct_f2',
                              'mpct_f3',
                              'spct_f3',
                              'mpct_m1',
                              'spct_m1',
                              'mpct_m2',
                              'spct_m2',
                              'mpct_m3',
                              'spct_m3',
                              'f1_025',
                              'f1_975',
                              'f2_025',
                              'f2_975',
                              'f3_025',
                              'f3_975',
                              'm1_025',
                              'm1_975',
                              'm2_025',
                              'm2_975',
                              'm3_025',
                              'm3_975'])



defaults.width = 400
defaults.height = 400


class Base_model():
    def __init__(self, number_of_females_1,
                 number_of_females_2,
                 number_of_females_3,
                 number_of_males_1,
                 number_of_males_2,
                 number_of_males_3,
                 number_of_initial_vacancies_1,
                 number_of_initial_vacancies_2,
                 number_of_initial_vacancies_3,
                 hiring_rate_women_1,
                 hiring_rate_women_2,
                 hiring_rate_women_3,
                 attrition_rate_women_1,
                 attrition_rate_women_2,
                 attrition_rate_women_3,
                 attrition_rate_men_1,
                 attrition_rate_men_2,
                 attrition_rate_men_3,
                 probablity_of_outside_hire_1,
                 probability_of_outside_hire_2,
                 probability_of_outside_hire_3,
                 duration,
                 female_promotion_probability_1,
                 female_promotion_probability_2,
                 male_promotion_probability_1,
                 male_promotion_probability_2,
                 upperbound,
                 lowerbound,
                 variation_range):
        self.name = 'replication m'
        self.label = 'replication m'
        self.nf1 = number_of_females_1
        self.nf2 = number_of_females_2
        self.nf3 = number_of_females_3
        self.nm1 = number_of_males_1
        self.nm2 = number_of_males_2
        self.nm3 = number_of_males_3
        self.vac3 = number_of_initial_vacancies_3
        self.vac2 = number_of_initial_vacancies_2
        self.vac1 = number_of_initial_vacancies_1
        self.bf1 = hiring_rate_women_1
        self.bf2 = hiring_rate_women_2
        self.bf3 = hiring_rate_women_3
        self.df1 = attrition_rate_women_1
        self.df2 = attrition_rate_women_2
        self.df3 = attrition_rate_women_3
        self.dm1 = attrition_rate_men_1
        self.dm2 = attrition_rate_men_2
        self.dm3 = attrition_rate_men_3
        self.phire2 = probability_of_outside_hire_2
        self.phire3 = probability_of_outside_hire_3
        self.duration = duration
        self.female_promotion_probability_1 = female_promotion_probability_1
        self.female_promotion_probability_2 = female_promotion_probability_2
        self.male_promotion_probability_1 = male_promotion_probability_1
        self.male_promotion_probability_2 = male_promotion_probability_2
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.variation_range = variation_range
        self.run = 0
        self.runarray = 0
        self.mean_matrix = 0
        self.std_matrix = 0
        self.pd_last_row_data = 0
        self.pct_female_matrix = 0
        self.probability_matrix = 0
        self.probability_by_level = 0

    def load_baseline_data_mgmt(self):
        '''
        This function will load the original management data to the object and remove any previous data.
        '''

        self.nf1 = 3
        self.nf2 = 3
        self.nf3 = 2
        self.nm1 = 11
        self.nm2 = 12
        self.nm3 = 43
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.172
        self.bf2 = 0.4
        self.bf3 = 0.167
        self.df1 = 0.056
        self.df2 = 0.00
        self.df3 = 0.074
        self.dm1 = 0.069
        self.dm2 = 0.057
        self.dm3 = 0.040
        self.phire2 = 0.125
        self.phire3 = 0.150
        self.duration = 40
        self.female_promotion_probability_1 = 0.0555
        self.female_promotion_probability_2 = 0.1905
        self.male_promotion_probability_1 = 0.0635
        self.male_promotion_probability_2 = 0.1149
        self.upperbound = 84
        self.lowerbound = 64
        self.variation_range = 3

    def load_optimistic_data_mgmt(self):
        '''
        This function will load the original management data to the object and remove any previous data.
        '''

        self.nf1 = 3
        self.nf2 = 3
        self.nf3 = 2
        self.nm1 = 11
        self.nm2 = 12
        self.nm3 = 43
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.300
        self.bf2 = 0.4
        self.bf3 = 0.300
        self.df1 = 0.056
        self.df2 = 0.00
        self.df3 = 0.146
        self.dm1 = 0.069
        self.dm2 = 0.057
        self.dm3 = 0.112
        self.phire2 = 0.125
        self.phire3 = 0.150
        self.duration = 40
        self.female_promotion_probability_1 = 0.0555
        self.female_promotion_probability_2 = 0.1905
        self.male_promotion_probability_1 = 0.0635
        self.male_promotion_probability_2 = 0.1149
        self.upperbound = 84
        self.lowerbound = 64
        self.variation_range = 3

    def load_most_optimistic_data_mgmt(self):
        '''
        This function will load the original management data to the object and remove any previous data.
        '''

        self.nf1 = 3
        self.nf2 = 3
        self.nf3 = 2
        self.nm1 = 11
        self.nm2 = 12
        self.nm3 = 43
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.400
        self.bf2 = 0.400
        self.bf3 = 0.300
        self.df1 = 0.036
        self.df2 = 0.00
        self.df3 = 0.054
        self.dm1 = 0.069
        self.dm2 = 0.057
        self.dm3 = 0.112
        self.phire2 = 0.125
        self.phire3 = 0.150
        self.duration = 40
        self.female_promotion_probability_1 = 0.0555
        self.female_promotion_probability_2 = 0.1905
        self.male_promotion_probability_1 = 0.0635
        self.male_promotion_probability_2 = 0.1149
        self.upperbound = 84
        self.lowerbound = 64
        self.variation_range = 3

    def load_pessimistic_data_mgmt(self):
        '''
        This function will load the original management data to the object and remove any previous data.
        '''

        self.nf1 = 3
        self.nf2 = 3
        self.nf3 = 2
        self.nm1 = 11
        self.nm2 = 12
        self.nm3 = 43
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.172
        self.bf2 = 0.4
        self.bf3 = 0.167
        self.df1 = 0.106
        self.df2 = 0.050
        self.df3 = 0.124
        self.dm1 = 0.069
        self.dm2 = 0.057
        self.dm3 = 0.076
        self.phire2 = 0.125
        self.phire3 = 0.150
        self.duration = 40
        self.female_promotion_probability_1 = 0.0555
        self.female_promotion_probability_2 = 0.1905
        self.male_promotion_probability_1 = 0.0635
        self.male_promotion_probability_2 = 0.1149
        self.upperbound = 84
        self.lowerbound = 64
        self.variation_range = 3

    def load_baseline_data_science(self):
        '''
        This function will load the original management data to the object and remove any previous data.
        '''

        self.nf1 = 14
        self.nf2 = 13
        self.nf3 = 9
        self.nm1 = 37
        self.nm2 = 28
        self.nm3 = 239
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.310
        self.bf2 = 0.222
        self.bf3 = 0.0
        self.df1 = 0.0
        self.df2 = 0.0
        self.df3 = 0.017
        self.dm1 = 0.009
        self.dm2 = 0.017
        self.dm3 = 0.033
        self.phire2 = 0.158
        self.phire3 = 0.339
        self.duration = 40
        self.female_promotion_probability_1 = 0.122
        self.female_promotion_probability_2 = 0.188
        self.male_promotion_probability_1 = 0.19
        self.male_promotion_probability_2 = 0.19
        self.upperbound = 350
        self.lowerbound = 330
        self.variation_range = 3

    def load_optimistic_data_science(self):
        '''
        This function will load the original management data to the object and remove any previous data.
        '''

        self.nf1 = 14
        self.nf2 = 13
        self.nf3 = 9
        self.nm1 = 37
        self.nm2 = 28
        self.nm3 = 239
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.310
        self.bf2 = 0.322
        self.bf3 = 0.050
        self.df1 = 0.0
        self.df2 = 0.0
        self.df3 = 0.069
        self.dm1 = 0.009
        self.dm2 = 0.017
        self.dm3 = 0.085
        self.phire2 = 0.158
        self.phire3 = 0.339
        self.duration = 40
        self.female_promotion_probability_1 = 0.122
        self.female_promotion_probability_2 = 0.188
        self.male_promotion_probability_1 = 0.19
        self.male_promotion_probability_2 = 0.19
        self.upperbound = 350
        self.lowerbound = 330
        self.variation_range = 3

    def load_most_optimistic_data_science(self):
        '''
        This function will load the original management data to the object and remove any previous data.
        '''

        self.nf1 = 14
        self.nf2 = 13
        self.nf3 = 9
        self.nm1 = 37
        self.nm2 = 28
        self.nm3 = 239
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.400
        self.bf2 = 0.322
        self.bf3 = 0.100
        self.df1 = 0.0
        self.df2 = 0.0
        self.df3 = 0.0
        self.dm1 = 0.009
        self.dm2 = 0.017
        self.dm3 = 0.085
        self.phire2 = 0.158
        self.phire3 = 0.339
        self.duration = 40
        self.female_promotion_probability_1 = 0.122
        self.female_promotion_probability_2 = 0.188
        self.male_promotion_probability_1 = 0.19
        self.male_promotion_probability_2 = 0.19
        self.upperbound = 350
        self.lowerbound = 330
        self.variation_range = 3

    def load_pessimistic_data_science(self):
        '''
        This function will load the original management data to the object and remove any previous data.
        '''

        self.nf1 = 14
        self.nf2 = 13
        self.nf3 = 9
        self.nm1 = 37
        self.nm2 = 28
        self.nm3 = 239
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.310
        self.bf2 = 0.222
        self.bf3 = 0.0
        self.df1 = 0.050
        self.df2 = 0.050
        self.df3 = 0.043
        self.dm1 = 0.009
        self.dm2 = 0.017
        self.dm3 = 0.059
        self.phire2 = 0.158
        self.phire3 = 0.339
        self.duration = 40
        self.female_promotion_probability_1 = 0.122
        self.female_promotion_probability_2 = 0.188
        self.male_promotion_probability_1 = 0.19
        self.male_promotion_probability_2 = 0.19
        self.upperbound = 350
        self.lowerbound = 330
        self.variation_range = 3

    def run_model(self):

        self.res = np.zeros([self.duration, 26], dtype=np.float32)
        df_ = pd.DataFrame(self.res)
        df_.columns = MODEL_RUN_COLUMNS

        recarray_results = df_.to_records(index=True)
        self.res = recarray_results
        return recarray_results

    def run_multiple(self, number_of_runs):

        res_array = np.recarray((number_of_runs,), dtype=[('run', object)])

        ## Then I need to run a loop to run multiple models and return their values to the record array

        for idx in range(number_of_runs):
            res_array['run'][idx] = self.run_model()

        ##Create empty arrays to hold the results

        self.results_matrix = pd.DataFrame(np.zeros([self.duration,
                                            len(RESULTS_COLUMNS)]))
        self.results_matrix.columns = RESULTS_COLUMNS

        self.pct_female_matrix = pd.DataFrame(np.zeros([self.duration,
                                                        len(FEMALE_MATRIX_COLUMNS)]))
        self.pct_female_matrix.columns = FEMALE_MATRIX_COLUMNS

        ## Collect mean and standard deviations for each column/row across all
        # iterations of the model.


        for idx in range(self.duration):


            # Set the year in the results matrix

            self.results_matrix.loc[idx,0] = idx

            ## This section will iterate over all of the values in the results
            ## matrix for a year, and it will get the mean and average values
            ## for each statistic for that year. This info contains the raw
            ## numbers for each grouping and goes to the gender numbers plots.

            for k, f in enumerate(MODEL_RUN_COLUMNS):
                _t = np.array([r['run'][f][idx] for r in res_array])

                self.results_matrix.loc[idx,
                        RESULTS_COLUMNS[k+1]] = np.array(np.mean(_t))
                self.results_matrix.loc[idx,
                        RESULTS_COLUMNS[k+27]] = np.array(np.std(_t))

                self.results_matrix.loc[idx,
                        RESULTS_COLUMNS[k+53]] = np.percentile(_t,2.5)
                self.results_matrix.loc[idx,
                        RESULTS_COLUMNS[k+79]] = np.percentile(_t,97.5)


            # Calculate the mean and standard deviation/percentiles
            # for each grouping.

            for k, lev in enumerate(LEVELS):
                if k <=2:
                    _u = np.array([r['run'][LEVELS[k]][idx] / (r['run'][LEVELS[k]][idx]
                                  + r['run'][LEVELS[k + 3]][idx]) for r in
                                   res_array])
                else:
                    _u = np.array([r['run'][LEVELS[k]][idx] / (r['run'][LEVELS[k]][idx]
                                  + r['run'][LEVELS[k - 3]][idx]) for r in
                                   res_array])

                self.pct_female_matrix.loc[idx, 'year'] = idx
                self.pct_female_matrix.loc[idx, FEMALE_MATRIX_COLUMNS[2*k+1]]\
                    = _u.mean()
                self.pct_female_matrix.loc[idx, FEMALE_MATRIX_COLUMNS[2*k+2]]\
                    = _u.std()

                self.pct_female_matrix.loc[idx, 'year'] = idx
                self.pct_female_matrix.loc[idx,
                    FEMALE_MATRIX_COLUMNS[2*k+1]] = np.percentile(_u,2.5)
                self.pct_female_matrix.loc[idx,
                    FEMALE_MATRIX_COLUMNS[2*k+2]] = np.percentile(_u,97.5)


        self.res_array = res_array


    def run_parameter_sweep(self, number_of_runs, param, llim,
                            ulim, num_of_steps):

        # First I will create a structured array to hold the results of the
        # simulation. The size of the array should be one for each step in the
        # parameter sweep. To calculate that,

        parameter_sweep_increments = np.linspace(llim, ulim, num_of_steps)

        parameter_sweep_results = np.zeros([len(parameter_sweep_increments),
                                            ])

        # Make arrays to hold the values for each key value for plotting.
        paramater_sweep_plot_array = np.zeros(
            [len(parameter_sweep_increments), 3])

        paramater_sweep_plot_array_f1 = np.zeros([len(
            parameter_sweep_increments), 3])

        paramater_sweep_plot_array_f2 = np.zeros([len(
            parameter_sweep_increments), 3])

        paramater_sweep_plot_array_f3 = np.zeros([len(
            parameter_sweep_increments), 3])

        paramater_sweep_plot_array_m1 = np.zeros([len(
            parameter_sweep_increments), 3])

        paramater_sweep_plot_array_m2 = np.zeros([len(
            parameter_sweep_increments), 3])

        paramater_sweep_plot_array_m3 = np.zeros([len(
            parameter_sweep_increments), 3])

        # Run simulations with parameter increments and collect into a container.

        for i, val in enumerate(parameter_sweep_increments):
            setattr(self, param, val)
            # setattr(self, 'duration', 20)
            self.run_multiple(number_of_runs)
            # print(getattr(self, param))
            model_final_year_results = self.pd_last_row_data

            paramater_sweep_plot_array[i, 0] = val
            paramater_sweep_plot_array_f1[i, 0] = val
            paramater_sweep_plot_array_f2[i, 0] = val
            paramater_sweep_plot_array_f3[i, 0] = val
            paramater_sweep_plot_array_m1[i, 0] = val
            paramater_sweep_plot_array_m2[i, 0] = val
            paramater_sweep_plot_array_m3[i, 0] = val

            paramater_sweep_plot_array[i, 1] = np.mean(model_final_year_results[
                                                           'gendprop'])
            paramater_sweep_plot_array[i, 2] = np.std(model_final_year_results[
                                                          'gendprop'])

            paramater_sweep_plot_array_f1[i, 1] = np.mean(
                model_final_year_results[
                    'f1'])
            paramater_sweep_plot_array_f1[i, 2] = np.std(
                model_final_year_results[
                    'f1'])

            paramater_sweep_plot_array_f2[i, 1] = np.mean(
                model_final_year_results[
                    'f2'])
            paramater_sweep_plot_array_f2[i, 2] = np.std(
                model_final_year_results[
                    'f2'])

            paramater_sweep_plot_array_f3[i, 1] = np.mean(
                model_final_year_results[
                    'f3'])
            paramater_sweep_plot_array_f3[i, 2] = np.std(
                model_final_year_results[
                    'f3'])

            paramater_sweep_plot_array_m1[i, 1] = np.mean(
                model_final_year_results[
                    'm1'])
            paramater_sweep_plot_array_m1[i, 2] = np.std(
                model_final_year_results[
                    'm1'])
            paramater_sweep_plot_array_m2[i, 1] = np.mean(
                model_final_year_results[
                    'm2'])
            paramater_sweep_plot_array_m2[i, 2] = np.std(
                model_final_year_results[
                    'm2'])

            paramater_sweep_plot_array_m3[i, 1] = np.mean(
                model_final_year_results[
                    'm3'])
            paramater_sweep_plot_array_m3[i, 2] = np.std(
                model_final_year_results[
                    'm3'])

            # print(getattr(self, param))

        # return parameter sweep results as list

        self.parameter_sweep_array = list([paramater_sweep_plot_array,
                                           paramater_sweep_plot_array_f1,
                                           paramater_sweep_plot_array_f2,
                                           paramater_sweep_plot_array_f3,
                                           paramater_sweep_plot_array_m1,
                                           paramater_sweep_plot_array_m2,
                                           paramater_sweep_plot_array_m3,
                                           param])

        # print(paramater_sweep_plot_array_f1)
        # print(paramater_sweep_plot_array_f2)
        # print(model_final_year_results['f2'])
        if hasattr(self, 'parameter_sweep_array'):
            return (0)

    def run_probability_analysis_gender_proportion(self, num_runs, target):

        ## First run the model multiple times to generate the mean and standard deviation matrices. This will also create the res_array attribute for the stored simulation data.

        self.run_multiple(num_runs)

        ## Then I have to create an array to hold the probability for the target value given the array at that time. So I would take in the target value and a vector of current values.

        probability_matrix = pd.DataFrame(np.zeros([self.duration, 5]))
        probability_matrix.columns = ['Year', 'Probability', 'Mean', 'Min',
                                      'Max']
        probability_matrix['Year'] = list(range(self.duration))

        ## Pull the gender ratio data from the sims and extract probability of reaching the target.

        for idx in range(self.duration):
            _s = np.array([sum(list([r['run']['f1'][idx],
                                     r['run']['f2'][idx],
                                     r['run']['f3'][idx]])) / sum(
                list([r['run']['f1'][idx],
                      r['run']['f2'][idx],
                      r['run']['f3'][idx],
                      r['run']['m1'][idx],
                      r['run']['m2'][idx],
                      r['run']['m3'][idx]])) for r in self.res_array])
            probability_matrix.loc[idx, 'Probability'] = \
                calculate_empirical_probability_of_value(target, _s)
            probability_matrix.loc[idx, 'Mean'] = _s.mean()
            probability_matrix.loc[idx, 'Min'] = _s.min()
            probability_matrix.loc[idx, 'Max'] = _s.max()

        self.probability_matrix = probability_matrix

    def run_probability_analysis_gender_by_level(self, num_runs, target):

        ## First run the model multiple times to generate the mean and standard deviation matrices. This will also create the res_array attribute for the stored simulation data.

        self.run_multiple(num_runs)

        probability_by_level_data = pd.DataFrame(np.zeros([self.duration, 7]))
        probability_by_level_data.columns = ['year', 'pf1', 'pf2', 'pf3', 'pm1',
                                             'pm2', 'pm3']

        for idx in range(self.duration):
            _u1 = np.array([r['run']['f1'][idx] / (r['run']['f1'][idx] + r[
                'run']['m1'][idx]) for r in self.res_array])

            _u2 = np.array([r['run']['f2'][idx] / (r['run']['f2'][idx] + r[
                'run']['m2'][idx]) for r in self.res_array])

            _u3 = np.array([r['run']['f3'][idx] / (r['run']['f3'][idx] + r[
                'run']['m3'][idx]) for r in self.res_array])

            probability_by_level_data['year'] = idx

            probability_by_level_data.loc[idx, 'pf1'] = \
                calculate_empirical_probability_of_value(target, _u1)

            probability_by_level_data.loc[idx, 'pf2'] = \
                calculate_empirical_probability_of_value(target, _u2)

            probability_by_level_data.loc[idx, 'pf3'] = \
                calculate_empirical_probability_of_value(target, _u3)

            probability_by_level_data.loc[idx, 'pm1'] = \
                1 - probability_by_level_data['pf1'][idx]

            probability_by_level_data.loc[idx, 'pm2'] = \
                1 - probability_by_level_data['pf2'][idx]

            probability_by_level_data.loc[idx, 'pm3'] = \
                1 - probability_by_level_data['pf3'][idx]

        self.probability_by_level = probability_by_level_data

        return (probability_by_level_data)

    def run_probability_analysis_parameter_sweep_gender_proportion(self,
                                                                   number_of_runs,
                                                                   param, llim,
                                                                   ulim,
                                                                   num_of_steps,
                                                                   target):

        pass

    def run_probability_analysis_parameter_sweep_gender_detail(self,
                                                               number_of_runs,
                                                               param,
                                                               prof_group, llim,
                                                               ulim,
                                                               num_of_steps,
                                                               target):

        ## This is similar to the parameter sweep, except that I am capturing the probability instead of
        # the mean and standard deviation

        ## Setup the sweep increments

        parameter_sweep_increments = np.linspace(llim, ulim, num_of_steps)

        ## Now I create a container for the data. In this case I am only looking at the probability a certain count
        ## is equal to or greater than a particular target value.

        empirical_probability_param_sweep_df = pd.DataFrame(
            np.zeros([len(parameter_sweep_increments),
                      len(PROBABILITY_ARRAY_COLUMN_NAMES)]),
            columns=PROBABILITY_ARRAY_COLUMN_NAMES)

        ## Loop over all increments and get the results for the final year. Then pass these results to the probability
        ## calculation function to get the empirical probability that the value is the target or greater.

        for i, val in enumerate(parameter_sweep_increments):
            setattr(self, param, val)
            self.run_multiple(number_of_runs)
            model_final_year_results = self.pd_last_row_data

            empirical_probability_param_sweep_df['param'][i] = val

            empirical_probability_param_sweep_df['prof_group_mean'][i] = \
                model_final_year_results[prof_group].mean()

            empirical_probability_param_sweep_df['probability'][
                i] = calculate_empirical_probability_of_value(target,
                                                              model_final_year_results[
                                                                  prof_group])

        self.last_empirical_probability_detail = empirical_probability_param_sweep_df

        return (empirical_probability_param_sweep_df)

    def plot_parameter_sweep(self, title, xlabel, ylabel):

        if not hasattr(self, 'parameter_sweep_array'):
            print("please run parameter sweep function first.")
            return (0)
        plot_array = self.parameter_sweep_array[0]
        plt.plot(plot_array[:, 0], plot_array[:, 1], label=self.label)
        plt.fill_between(plot_array[:, 0], plot_array[:, 1] +
                         1.96 * plot_array[:, 2], plot_array[:, 1] -
                         1.96 * plot_array[:, 2], alpha=0.5)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='upper right', shadow=True)
        plt.show()

    def plot_parameter_sweep_all_counts(self):

        # This function will generate graphs of the mean and standard deviation
        # matrices.

        if not hasattr(self, 'parameter_sweep_array'):
            print("please run parameter sweep function first.")
            return (0)

        array_list = self.parameter_sweep_array

        f, axarr = plt.subplots(nrows=2, ncols=3)
        f.suptitle('Model: ' + self.name)
        axarr[0, 0].plot(array_list[1][:, 0], array_list[1][:, 1],
                         label=self.label)
        axarr[0, 0].set_title('Female level 1')
        axarr[0, 0].set_xlabel('parameter: ' + array_list[7])
        axarr[0, 0].set_ylabel('Number of Females')
        axarr[0, 0].fill_between(array_list[1][:, 0], array_list[1][:, 1] +
                                 1.96 * array_list[1][:, 2],
                                 array_list[1][:, 1] -
                                 1.96 * array_list[1][:, 2], alpha=0.5)
        axarr[0, 0].legend(loc='upper right', shadow=True)

        axarr[0, 1].plot(array_list[2][:, 0], array_list[2][:, 1])
        axarr[0, 1].set_title('Female level 2')
        axarr[0, 1].set_xlabel('parameter: ' + array_list[7])
        axarr[0, 1].set_ylabel('Number of Females')
        axarr[0, 1].fill_between(array_list[2][:, 0], array_list[2][:, 1] +
                                 1.96 * array_list[2][:, 2],
                                 array_list[2][:, 1] -
                                 1.96 * array_list[2][:, 2], alpha=0.5)

        axarr[0, 2].plot(array_list[3][:, 0], array_list[3][:, 1])
        axarr[0, 2].set_title('Female level 3')
        axarr[0, 2].set_xlabel('parameter: ' + array_list[7])
        axarr[0, 2].set_ylabel('Number of Females')
        axarr[0, 2].fill_between(array_list[3][:, 0], array_list[3][:, 1] +
                                 1.96 * array_list[3][:, 2],
                                 array_list[3][:, 1] -
                                 1.96 * array_list[3][:, 2], alpha=0.5)

        axarr[1, 0].plot(array_list[4][:, 0], array_list[4][:, 1])
        axarr[1, 0].set_title('Male level 1')
        axarr[1, 0].set_xlabel('parameter: ' + array_list[7])
        axarr[1, 0].set_ylabel('Number of Males')
        axarr[1, 0].fill_between(array_list[4][:, 0], array_list[4][:, 1] +
                                 1.96 * array_list[4][:, 2],
                                 array_list[4][:, 1] -
                                 1.96 * array_list[4][:, 2], alpha=0.5)

        axarr[1, 1].plot(array_list[5][:, 0], array_list[5][:, 1])
        axarr[1, 1].set_title('Male level 2')
        axarr[1, 1].set_xlabel('parameter: ' + array_list[7])
        axarr[1, 1].set_ylabel('Number of Males')
        axarr[1, 1].fill_between(array_list[5][:, 0], array_list[5][:, 1] +
                                 1.96 * array_list[5][:, 2],
                                 array_list[5][:, 1] -
                                 1.96 * array_list[5][:, 2], alpha=0.5)

        axarr[1, 2].plot(array_list[6][:, 0], array_list[6][:, 1])
        axarr[1, 2].set_title('Male level 3')
        axarr[1, 2].set_xlabel('parameter: ' + array_list[7])
        axarr[1, 2].set_ylabel('Number of Males')
        axarr[1, 2].fill_between(array_list[6][:, 0], array_list[6][:, 1] +
                                 1.96 * array_list[6][:, 2],
                                 array_list[6][:, 1] -
                                 1.96 * array_list[6][:, 2], alpha=0.5)

        plt.show()

    def export_model_run(self, model_label, number_of_runs):

        if not hasattr(self, 'res'):
            self.run_multiple(number_of_runs)

        # first I will allocate the memory by creating an empty dataframe.
        # then I will iterate over the res_array matrix and write to the
        # correct rows of the dataframe. This is more memory efficient compared
        # to appending to a dataframe.

        # print(pd.DataFrame(self.res_array['run'][3]))

        print_array = np.zeros([self.duration * number_of_runs,
                                28])

        for idx in range(number_of_runs):
            print_array[(idx * self.duration):(idx * self.duration +
                                               self.duration), 0] = idx

            print_array[(idx * self.duration):(idx * self.duration +
                                               self.duration),
            1:] = pd.DataFrame(self.res_array['run'][idx])

        # work with barbara to craft the filename
        # model_label + 160114_HH:MM(24hour) +
        filename = model_label + "_" + str(datetime.datetime.now()) + "_iter" \
                   + str(number_of_runs) + ".csv"
        pd.DataFrame(print_array, columns=['run',
                                           'year',
                                           'women_1',
                                           'women_2',
                                           'women_3',
                                           'men_1',
                                           'men_2',
                                           'men_3',
                                           'attrition_3',
                                           'attrition_2',
                                           'attrition_1',
                                           'women_promotion_rate_1',
                                           'women_promotion_rate_2',
                                           'gender_proportion_overall',
                                           'unfilled_vacancies',
                                           'department_size',
                                           'women_hires_3',
                                           'men_hired_3',
                                           'women_hired_2',
                                           'men_hired_2',
                                           'women_hired_1',
                                           'men_hired_1',
                                           'women_promoted_3',
                                           'men_promoted_3',
                                           'women_promoted_2',
                                           'men_promoted_2',
                                           'women_promoted_1',
                                           'men_promoted_1']).to_csv(filename)

    def deprecated_plot_overall_chart(self,
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
        self.run_multiple(number_of_runs)

        # set default plot parameters. The xaxis is generally duration,
        # though I have the option--depending on the plot, to put in a
        # different x-axis.

        xval = self.duration

        if plottype == 'probability proportion':
            self.run_probability_analysis_gender_proportion(number_of_runs,
                                                            target)

            yval = self.probability_matrix['Probability']
            fill_matrix = 0

        if plottype == 'gender proportion':
            yval = self.mean_matrix['gendprop']
            fill_matrix = self.std_matrix['gendprop']

        if plottype == 'unfilled vacancies':
            yval = self.mean_matrix['unfilled']
            fill_matrix = self.std_matrix['unfilled']

        if plottype == 'department size':
            yval = self.dept_size_matrix['mean']
            fill_matrix = self.dept_size_matrix['std']

        if plottype == 'male female numbers':
            yval = sum(list([self.mean_matrix['f1'],
                             self.mean_matrix['f2'],
                             self.mean_matrix['f3']]))

            fill_matrix = 0

            yval2 = sum(list([self.mean_matrix['m1'],
                              self.mean_matrix['m2'],
                              self.mean_matrix['m3']]))

            total_faculty = sum(list([self.mean_matrix['f1'],
                                      self.mean_matrix['f2'],
                                      self.mean_matrix['f3'],
                                      self.mean_matrix['m1'],
                                      self.mean_matrix['m2'],
                                      self.mean_matrix['m3']]))

            yval3 = np.round(target * total_faculty)

        plt.plot(range(xval),
                 yval,
                 linewidth=line_width,
                 marker=marker_shape,
                 color=linecolor,
                 label=model_legend_label)

        plt.fill_between(range(xval),
                         yval + 1.96 * fill_matrix,
                         yval - 1.96 * fill_matrix,
                         alpha=transparency,
                         facecolor=linecolor)

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

        if male_female_numbers_plot:
            plt.plot(range(xval),
                     yval2,
                     color=mf_male_color,
                     label=mf_male_label,
                     linestyle=mf_male_linestyle,
                     linewidth=mf_male_linewidth)

            plt.plot(range(xval),
                     yval3,
                     color=mf_target_color,
                     label=mf_target_label,
                     linestyle=mf_target_linestyle,
                     linewidth=mf_target_linewidth)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc=legend_location, shadow=True)
        plt.show()

    def deprecated_plot_level_chart(self,
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
                                    transparency=0.25,
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

        self.run_multiple(number_of_runs)

        # set default plot parameters. The xaxis is generally duration,
        # though I have the option--depending on the plot, to put in a
        # different x-axis.

        xval = self.duration

        if plottype == 'probability proportion':
            d = self.run_probability_analysis_gender_by_level(number_of_runs,
                                                              target)

            yval_f1 = d['pf1']
            yval_f2 = d['pf2']
            yval_f3 = d['pf3']
            yval_m1 = d['pm1']
            yval_m2 = d['pm2']
            yval_m3 = d['pm3']

            fill_f1 = 0
            fill_f2 = 0
            fill_f3 = 0
            fill_m1 = 0
            fill_m2 = 0
            fill_m3 = 0

        if plottype == 'gender proportion':
            pd_stats_matrix = pd.DataFrame(self.model_summary_stats)
            tmp = pd_stats_matrix.select_dtypes(include=[np.number])
            pd_stats_matrix.loc[:, tmp.columns] = np.round(tmp, 2)

            yval_f1 = self.pct_female_matrix['mpct_f1']
            yval_f2 = self.pct_female_matrix['mpct_f2']
            yval_f3 = self.pct_female_matrix['mpct_f3']
            yval_m1 = self.pct_female_matrix['mpct_m1']
            yval_m2 = self.pct_female_matrix['mpct_m2']
            yval_m3 = self.pct_female_matrix['mpct_m3']

            fill_f1 = self.pct_female_matrix['spct_f1']
            fill_f2 = self.pct_female_matrix['spct_f2']
            fill_f3 = self.pct_female_matrix['spct_f3']
            fill_m1 = self.pct_female_matrix['spct_m1']
            fill_m2 = self.pct_female_matrix['spct_m2']
            fill_m3 = self.pct_female_matrix['spct_m3']

        if plottype == 'gender number':
            if self.model_summary_stats == 0:
                self.run_multiple(number_of_runs)

            yval_f1 = self.mean_matrix['f1']
            yval_f2 = self.mean_matrix['f2']
            yval_f3 = self.mean_matrix['f3']
            yval_m1 = self.mean_matrix['m1']
            yval_m2 = self.mean_matrix['m2']
            yval_m3 = self.mean_matrix['m3']

            fill_f1 = self.std_matrix['f1']
            fill_f2 = self.std_matrix['f2']
            fill_f3 = self.std_matrix['f3']
            fill_m1 = self.std_matrix['m1']
            fill_m2 = self.std_matrix['m2']
            fill_m3 = self.std_matrix['m3']

        f, axarr = plt.subplots(nrows=2, ncols=3)
        f.suptitle(group_title)
        axarr[0, 0].plot(range(xval),
                         np.minimum(1,
                                    np.maximum(0,
                                               yval_f1)),
                         label=model_legend_label,
                         linewidth=line_width,
                         color=linecolor,
                         marker=marker_shape)
        axarr[0, 0].set_xlim([xmin_f1, xmax_f1])
        axarr[0, 0].set_ylim([ymin_f1, ymax_f1])
        axarr[0, 0].set_title(title_f1)
        axarr[0, 0].set_xlabel(xlabel_f1)
        axarr[0, 0].set_ylabel(ylabel_f1)
        axarr[0, 0].fill_between(range(xval),
                                 np.minimum(1,
                                            yval_f1 + 1.96 * fill_f1),
                                 np.maximum(0,
                                            yval_f1 - 1.96 * fill_f1),
                                 alpha=transparency,
                                 facecolor=linecolor)
        axarr[0, 0].legend(loc=legend_location, shadow=True)

        axarr[0, 1].plot(range(xval),
                         np.minimum(1,
                                    np.maximum(0,
                                               yval_f2)),
                         label=model_legend_label,
                         linewidth=line_width,
                         color=linecolor,
                         marker=marker_shape
                         )
        axarr[0, 1].set_xlim([xmin_f2, xmax_f2])
        axarr[0, 1].set_ylim([ymin_f2, ymax_f2])
        axarr[0, 1].set_title(title_f2)
        axarr[0, 1].set_xlabel(xlabel_f2)
        axarr[0, 1].set_ylabel(ylabel_f2)
        axarr[0, 1].fill_between(range(xval),
                                 np.minimum(1,
                                            yval_f2 + 1.96 * fill_f2),
                                 np.maximum(0,
                                            yval_f2 - 1.96 * fill_f2),
                                 alpha=transparency,
                                 facecolor=linecolor)

        axarr[0, 2].plot(range(xval),
                         np.minimum(1,
                                    np.maximum(0,
                                               yval_f3)),
                         label=model_legend_label,
                         linewidth=line_width,
                         color=linecolor,
                         marker=marker_shape
                         )
        axarr[0, 2].set_xlim([xmin_f3, xmax_f3])
        axarr[0, 2].set_ylim([ymin_f3, ymax_f3])
        axarr[0, 2].set_title(title_f3)
        axarr[0, 2].set_xlabel(xlabel_f3)
        axarr[0, 2].set_ylabel(ylabel_f3)
        axarr[0, 2].fill_between(range(xval),
                                 np.minimum(1,
                                            yval_f3 + 1.96 * fill_f3),
                                 np.maximum(0,
                                            yval_f3 - 1.96 * fill_f3),
                                 alpha=transparency,
                                 facecolor=linecolor)

        axarr[1, 0].plot(range(xval),
                         np.minimum(1,
                                    np.maximum(0,
                                               yval_m1)),
                         label=self.label,
                         linewidth=line_width,
                         color=linecolor,
                         marker=marker_shape
                         )
        axarr[1, 0].set_xlim([xmin_m1, xmax_m1])
        axarr[1, 0].set_ylim([ymin_m1, ymax_m1])
        axarr[1, 0].set_title(title_m1)
        axarr[1, 0].set_xlabel(xlabel_m1)
        axarr[1, 0].set_ylabel(ylabel_m1)
        axarr[1, 0].fill_between(range(xval),
                                 np.minimum(1,
                                            yval_m1 + 1.96 * fill_m1),
                                 np.maximum(0,
                                            yval_m1 - 1.96 * fill_m1),
                                 alpha=transparency,
                                 facecolor=linecolor)

        axarr[1, 1].plot(range(xval),
                         np.minimum(1,
                                    np.maximum(0,
                                               yval_m2)),
                         label=self.label,
                         linewidth=line_width,
                         color=linecolor,
                         marker=marker_shape
                         )
        axarr[1, 1].set_xlim([xmin_m2, xmax_m2])
        axarr[1, 1].set_ylim([ymin_m2, ymax_m2])
        axarr[1, 1].set_title(title_m2)
        axarr[1, 1].set_xlabel(xlabel_m2)
        axarr[1, 1].set_ylabel(ylabel_m2)
        axarr[1, 1].fill_between(range(xval),
                                 np.minimum(1,
                                            yval_m2 + 1.96 * fill_m2),
                                 np.maximum(0,
                                            yval_m2 - 1.96 * fill_m2),
                                 alpha=transparency,
                                 facecolor=linecolor)

        axarr[1, 2].plot(range(xval),
                         np.minimum(1,
                                    np.maximum(0,
                                               yval_m3)),
                         label=self.label,
                         linewidth=line_width,
                         color=linecolor,
                         marker=marker_shape
                         )
        axarr[1, 2].set_xlim([xmin_m3, xmax_m3])
        axarr[1, 2].set_ylim([ymin_m3, ymax_m3])
        axarr[1, 2].set_title(title_m3)
        axarr[1, 2].set_xlabel(xlabel_m3)
        axarr[1, 2].set_ylabel(ylabel_m3)
        axarr[1, 2].fill_between(range(xval),
                                 np.minimum(1,
                                            yval_m3 + 1.96 * fill_m3),
                                 np.maximum(0,
                                            yval_m3 - 1.96 * fill_m3),
                                 alpha=transparency,
                                 facecolor=linecolor)

        if target_plot == True:
            axarr[0, 0].axhline(y=target,
                                color=target_color,
                                linestyle=target_plot_line_style,
                                linewidth=target_plot_linewidth,
                                label=target_plot_legend_label)
            axarr[0, 0].legend(loc=legend_location, shadow=True)

            axarr[0, 1].axhline(y=target,
                                color=target_color,
                                linestyle=target_plot_line_style,
                                linewidth=target_plot_linewidth,
                                label=target_plot_legend_label)
            axarr[0, 2].axhline(y=target,
                                color=target_color,
                                linestyle=target_plot_line_style,
                                linewidth=target_plot_linewidth,
                                label=target_plot_legend_label)
            axarr[1, 0].axhline(y=1 - target,
                                color=target_color,
                                linestyle=target_plot_line_style,
                                linewidth=target_plot_linewidth,
                                label=target_plot_legend_label)
            axarr[1, 1].axhline(y=1 - target,
                                color=target_color,
                                linestyle=target_plot_line_style,
                                linewidth=target_plot_linewidth,
                                label=target_plot_legend_label)
            axarr[1, 2].axhline(y=1 - target,
                                color=target_color,
                                linestyle=target_plot_line_style,
                                linewidth=target_plot_linewidth,
                                label=target_plot_legend_label)

        if percent_line_plot == True:
            axarr[0, 0].axhline(y=percent_line_value,
                                color=color_percent_line,
                                linestyle=percent_line_style,
                                linewidth=percent_linewidth,
                                label=percent_legend_label)
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

    def plot_overall_chart(self,
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
                           xmax=defaults.width,
                           ymax=defaults.height,
                           transparency=0.25,
                           marker_shape=None,
                           linecolor='green',
                           target_plot=False,
                           legend_location='upper right',
                           color_target='red',
                           percent_line_plot=False,
                           percent_line_value=0.5,
                           color_percent_line='red',
                           target_plot_line_style='--',
                           percent_line_style='-.',
                           target_plot_linewidth=2,
                           percent_linewidth=2,
                           model_legend_label='model',
                           target_plot_legend_label='target',
                           percent_legend_label='percent',
                           male_female_numbers_plot=False,
                           mf_male_color='black',
                           mf_target_color='red',
                           mf_male_label='Male',
                           mf_target_label='Target',
                           mf_male_linestyle=None,
                           mf_target_linestyle=None,
                           mf_male_linewidth=2,
                           mf_target_linewidth=2
                           ):

        # generate data for the plot.
        self.run_multiple(number_of_runs)

        # set default plot parameters. The xaxis is generally duration,
        # though I have the option--depending on the plot, to put in a
        # different x-axis.

        xval = self.duration

        p = figure(title=title,
                   x_axis_label=xlabel,
                   y_axis_label=ylabel,
                   width=xmax,
                   height=ymax)

        if plottype == 'probability proportion':
            self.run_probability_analysis_gender_proportion(number_of_runs,
                                                            target)

            yval = self.probability_matrix['Probability']
            fill_matrix = 0

        if plottype == 'gender proportion':
            yval = self.mean_matrix['gendprop']
            fill_matrix = self.std_matrix['gendprop']

        if plottype == 'unfilled vacancies':
            yval = self.mean_matrix['unfilled']
            fill_matrix = self.std_matrix['unfilled']

        if plottype == 'department size':
            yval = self.dept_size_matrix['mean']
            fill_matrix = self.dept_size_matrix['std']

        if plottype == 'male female numbers':
            yval = sum(list([self.mean_matrix['f1'],
                             self.mean_matrix['f2'],
                             self.mean_matrix['f3']]))

            fill_matrix = 0

            yval2 = sum(list([self.mean_matrix['m1'],
                              self.mean_matrix['m2'],
                              self.mean_matrix['m3']]))

            total_faculty = sum(list([self.mean_matrix['f1'],
                                      self.mean_matrix['f2'],
                                      self.mean_matrix['f3'],
                                      self.mean_matrix['m1'],
                                      self.mean_matrix['m2'],
                                      self.mean_matrix['m3']]))

            yval3 = np.round(target * total_faculty)

        p.line(range(xval), yval,
               line_width=line_width,
               line_color=linecolor)
        p.circle(range(xval), yval, size=3)

        x_data = np.arange(0, xval)
        band_x = np.append(x_data, x_data[::-1])
        upper_band = yval + 1.96 * fill_matrix
        lower_band = yval - 1.96 * fill_matrix
        band_y = np.append(lower_band, upper_band[::-1])

        p.patch(band_x,
                band_y,
                color=linecolor,
                fill_alpha=transparency)

        # plt.fill_between(range(xval),
        #                  yval + 1.96 * fill_matrix,
        #                  yval - 1.96 * fill_matrix,
        #                  alpha=transparency,
        #                  facecolor=linecolor)

        if target_plot:
            p.line(range(xval),
                   target,
                   line_color=color_target,
                   line_width=target_plot_linewidth,
                   line_dash=[6, 6])

        if percent_line_plot:
            p.line(range(xval),
                   percent_line_value,
                   line_color=color_percent_line,
                   line_width=percent_linewidth,
                   line_dash=[2, 2])

        if male_female_numbers_plot:
            p.line(range(xval),
                   yval2,
                   line_color=mf_male_color,
                   line_width=mf_male_linewidth)

            p.line(range(xval),
                   yval3,
                   line_color=mf_target_color,
                   line_width=mf_target_linewidth)

        return (p)

    def plot_level_chart(self,
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
                         xmax_f1=defaults.width // 2,
                         ymax_f1=defaults.height // 2,
                         xmin_f2=0,
                         ymin_f2=0,
                         xmax_f2=defaults.width,
                         ymax_f2=defaults.height,
                         xmin_f3=0,
                         ymin_f3=0,
                         xmax_f3=defaults.width,
                         ymax_f3=defaults.height,
                         xmin_m1=0,
                         ymin_m1=0,
                         xmax_m1=defaults.width,
                         ymax_m1=defaults.height,
                         xmin_m2=0,
                         ymin_m2=0,
                         xmax_m2=defaults.width,
                         ymax_m2=defaults.height,
                         xmin_m3=0,
                         ymin_m3=0,
                         xmax_m3=defaults.width,
                         ymax_m3=defaults.height,
                         legend_location='upper right',
                         model_legend_label='model',
                         transparency=0.25,
                         marker_shape=None,
                         linecolor='green',
                         target_plot=False,
                         target_color='red',
                         target_plot_line_style='--',
                         target_plot_linewidth=2,
                         target_plot_legend_label='target',
                         percent_line_plot=False,
                         percent_line_value=0.5,
                         color_percent_line='red',
                         percent_line_style='-.',
                         percent_linewidth=2,
                         percent_legend_label='percent'):

        # generate data for the plot.

        self.run_multiple(number_of_runs)

        # set default plot parameters. The xaxis is generally duration,
        # though I have the option--depending on the plot, to put in a
        # different x-axis.

        xval = self.duration

        if plottype == 'probability proportion':
            d = self.run_probability_analysis_gender_by_level(number_of_runs,
                                                              target)

            yval_f1 = d['pf1']
            yval_f2 = d['pf2']
            yval_f3 = d['pf3']
            yval_m1 = d['pm1']
            yval_m2 = d['pm2']
            yval_m3 = d['pm3']

            fill_f1 = 0
            fill_f2 = 0
            fill_f3 = 0
            fill_m1 = 0
            fill_m2 = 0
            fill_m3 = 0

        if plottype == 'gender proportion':
            pd_stats_matrix = pd.DataFrame(self.model_summary_stats)
            tmp = pd_stats_matrix.select_dtypes(include=[np.number])
            pd_stats_matrix.loc[:, tmp.columns] = np.round(tmp, 2)

            yval_f1 = self.pct_female_matrix['mpct_f1']
            yval_f2 = self.pct_female_matrix['mpct_f2']
            yval_f3 = self.pct_female_matrix['mpct_f3']
            yval_m1 = self.pct_female_matrix['mpct_m1']
            yval_m2 = self.pct_female_matrix['mpct_m2']
            yval_m3 = self.pct_female_matrix['mpct_m3']

            fill_f1 = self.pct_female_matrix['spct_f1']
            fill_f2 = self.pct_female_matrix['spct_f2']
            fill_f3 = self.pct_female_matrix['spct_f3']
            fill_m1 = self.pct_female_matrix['spct_m1']
            fill_m2 = self.pct_female_matrix['spct_m2']
            fill_m3 = self.pct_female_matrix['spct_m3']

        if plottype == 'gender number':
            if self.model_summary_stats == 0:
                self.run_multiple(number_of_runs)

            yval_f1 = self.mean_matrix['f1']
            yval_f2 = self.mean_matrix['f2']
            yval_f3 = self.mean_matrix['f3']
            yval_m1 = self.mean_matrix['m1']
            yval_m2 = self.mean_matrix['m2']
            yval_m3 = self.mean_matrix['m3']

            fill_f1 = self.std_matrix['f1']
            fill_f2 = self.std_matrix['f2']
            fill_f3 = self.std_matrix['f3']
            fill_m1 = self.std_matrix['m1']
            fill_m2 = self.std_matrix['m2']
            fill_m3 = self.std_matrix['m3']

        # print(fill_f2)
        # print(fill_m2)

        # TODO set levels to configurable parameter. This is just for test.
        levels = ['f1', 'f2', 'f3', 'm1', 'm2', 'm3']
        yvals = [yval_f1, yval_f2, yval_f3, yval_m1, yval_m2, yval_m3]
        fills = [fill_f1, fill_f2, fill_f3, fill_m1, fill_m2, fill_m3]

        plots = []

        x_data = np.arange(0, xval)
        band_x = np.append(x_data, x_data[::-1])

        for key, val in enumerate(levels):
            plots.append(figure(title=val,
                                x_axis_label='year',
                                y_axis_label='percentage female',
                                width=xmax_f1,
                                height=ymax_f1))

        for i, p in enumerate(plots):
            p.line(range(xval), np.minimum(1,
                                           np.maximum(0,
                                                      yvals[i])),
                   line_width=line_width,
                   line_color=linecolor)

            upper_band = np.minimum(1, yvals[i] + 1.96 * fills[i])
            lower_band = np.maximum(0, yvals[i] - 1.96 * fills[i])
            band_y = np.append(lower_band, upper_band[::-1])

            p.patch(band_x,
                    band_y,
                    color=linecolor,
                    fill_alpha=transparency)

        if target_plot == True:

            for i, p in enumerate(plots):
                p.line(range(xval),
                       target,
                       line_color=target_color,
                       line_width=target_plot_linewidth,
                       line_dash=[6, 6])

        if percent_line_plot == True:

            for i, p in enumerate(plots):
                p.line(range(xval),
                       percent_line_value,
                       line_color=color_percent_line,
                       line_width=percent_linewidth,
                       line_dash=[2, 2])

        grid = gridplot([[plots[0], plots[1], plots[2]],
                         [plots[3], plots[4], plots[5]]])
        return (grid)


## Supplementary/Helper functions

def calculate_empirical_probability_of_value(criterion, data_vector):
    emp_prob = 1 - sum(data_vector <= criterion) / len(data_vector)
    return (emp_prob)
