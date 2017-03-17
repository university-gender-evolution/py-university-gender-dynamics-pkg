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

LEVELS = list(['number_f1',
               'number_f2',
               'number_f3',
               'number_m1',
               'number_m2',
               'number_m3'])

MODEL_RUN_COLUMNS = list(['number_f1',
                          'number_f2',
                          'number_f3',
                          'number_m1',
                          'number_m2',
                          'number_m3',
                          'vacancies_3',
                          'vacancies_2',
                          'vacancies_1',
                          'prom1',
                          'prom2',
                          'gender_proportion_overall',
                          'unfilled_vacancies',
                          'department_size',
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

RESULTS_COLUMNS = list(['year', 'mean_f1', 'mean_f2',
                        'mean_f3', 'mean_m1',
                        'mean_m2', 'mean_m3',
                        'mean_vac_3', 'mean_vac_2',
                        'mean_vac_1', 'mean_prom1',
                        'mean_prom2', 'mean_gendprop',
                        'mean_unfilled', 'mean_dept_size',
                        'mean_f_hire_3', 'mean_m_hire_3',
                        'mean_f_hire_2', 'mean_m_hire_2',
                        'mean_f_hire_1', 'mean_m_hire_1',
                        'mean_f_prom_3', 'mean_m_prom_3',
                        'mean_f_prom_2', 'mean_m_prom_2',
                        'mean_f_prom_1', 'mean_m_prom_1',
                        'std_f1', 'std_f2',
                        'std_f3', 'std_m1',
                        'std_m2', 'std_m3',
                        'std_vac_3', 'std_vac_2',
                        'std_vac_1', 'std_prom1',
                        'std_prom2', 'std_gendprop',
                        'std_unfilled', 'std_dept_size',
                        'std_f_hire_3', 'std_m_hire_3',
                        'std_f_hire_2', 'std_m_hire_2',
                        'std_f_hire_1', 'std_m_hire_1',
                        'std_f_prom_3', 'std_m_prom_3',
                        'std_f_prom_2', 'std_m_prom_2',
                        'std_f_prom_1', 'std_m_prom_1',
                        'f1_025', 'f2_025',
                        'f3_025', 'm1_025',
                        'm2_025', 'm3_025', 'vac_3_025',
                        'vac_2_025', 'vac_1_025',
                        'prom1_025', 'prom2_025',
                        'gendprop_025', 'unfilled_025',
                        'dept_size_025', 'f_hire_3_025',
                        'm_hire_3_025', 'f_hire_2_025',
                        'm_hire_2_025', 'f_hire_1_025',
                        'm_hire_1_025', 'f_prom_3_025',
                        'm_prom_3_025', 'f_prom_2_025',
                        'm_prom_2_025', 'f_prom_1_025',
                        'm_prom_1_025', 'f1_975',
                        'f2_975', 'f3_975',
                        'm1_975', 'm2_975',
                        'm3_975', 'vac_3_975',
                        'vac_2_975', 'vac_1_975',
                        'prom_1_975', 'prom2_975',
                        'gendprop_975', 'unfilled_975',
                        'dept_size_975', 'f_hire_3_975',
                        'm_hire_3_975', 'f_hire_2_975',
                        'm_hire_2_975', 'f_hire_1_975',
                        'm_hire_1_975', 'f_prom_3_975',
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

EXPORT_COLUMNS_FOR_CSV = list(['hiring_rate_women_1',
                               'hiring_rate_women_2',
                               'hiring_rate_women_3',
                               'hiring_rate_men_1',
                               'hiring_rate_men_2',
                               'hiring_rate_men_3',
                               'attrition_rate_women_1',
                               'attrition_rate_women_2',
                               'attrition_rate_women_3',
                               'attrition_rate_men_1',
                               'attrition_rate_men_2',
                               'attrition_rate_men_3',
                               'probablity_of_outside_hire_1',
                               'probability_of_outside_hire_2',
                               'probability_of_outside_hire_3',
                               'female_promotion_rate_1',
                               'female_promotion_rate_2',
                               'male_promotion_rate_1',
                               'male_promotion_rate_2',
                               'dept_size_upperbound',
                               'dept_size_lowerbound',
                               'dept_size_exogenous_variation_range',
                               'duration'])

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
        self.pd_last_row_data = 0
        self.pct_female_matrix = 0
        self.probability_matrix = 0
        self.probability_by_level = 0

    def load_baseline_data_mgmt(self):
        '''
        This function will load the parameter values for the baseline
        scenario of the Business School into the model
        :return: This function does not return anything. It changes the
        current model in place.
        :rtype: void
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
        self.name = "Promote-Hire baseline"
        self.label = "Promote-Hire baseline"

    def load_optimistic_data_mgmt(self):
        '''
        This function will load the parameter values for the optimistic
        scenario of the Business School into the model
        :return: This function does not return anything. It changes the
        current model in place.
        :rtype: void
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
        This function will load the parameter values for the most optimistic
        scenario of the Business School into the model
        :return: This function does not return anything. It changes the
        current model in place.
        :rtype: void

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
        This function will load the parameter values for the pessimistic
        scenario of the Business School into the model
        :return: This function does not return anything. It changes the
        current model in place.
        :rtype: void

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
        This function will sets the parameter values for the model
        to the baseline for the science department.
        :return: This function does not return anything. It changes the
        current model in place.
        :rtype: void

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
        This function will load the parameter values for the optimistic
        scenario of the Science Department into the model
        :return: This function does not return anything. It changes the
        current model in place.
        :rtype: void
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
        This function will load the parameter values for the most optimistic
        scenario of the Science Department into the model
        :return: This function does not return anything. It changes the
        current model in place.
        :rtype: void

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
        This function will load the parameter values for the pessimistic
        scenario of the Science Department into the model
        :return: This function does not return anything. It changes the
        current model in place.
        :rtype: void

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
                                                        len(
                                                            FEMALE_MATRIX_COLUMNS)]))
        self.pct_female_matrix.columns = FEMALE_MATRIX_COLUMNS

        ## Collect mean and standard deviations for each column/row across all
        # iterations of the model.


        for idx in range(self.duration):

            # Set the year in the results matrix

            self.results_matrix.loc[idx, 0] = idx

            ## This section will iterate over all of the values in the results
            ## matrix for a year, and it will get the mean and average values
            ## for each statistic for that year. This info contains the raw
            ## numbers for each grouping and goes to the gender numbers plots.

            for k, f in enumerate(MODEL_RUN_COLUMNS):
                _t = np.array([r['run'][f][idx] for r in res_array])

                self.results_matrix.loc[idx,
                                        RESULTS_COLUMNS[k + 1]] = np.array(
                    np.mean(_t)) if \
                    np.isfinite(np.array(np.mean(_t))) else 0

                self.results_matrix.loc[idx,
                                        RESULTS_COLUMNS[k + 27]] = np.array(
                    np.std(_t)) if \
                    np.isfinite(np.array(np.std(_t))) else 0

                self.results_matrix.loc[idx,
                                        RESULTS_COLUMNS[
                                            k + 53]] = np.percentile(_t, 2.5) if \
                    np.isfinite(np.percentile(_t, 2.5))  else 0

                self.results_matrix.loc[idx,
                                        RESULTS_COLUMNS[
                                            k + 79]] = np.percentile(_t,
                                                                     97.5) if \
                    np.isfinite(np.percentile(_t, 97.5)) else 0

            # Calculate the mean and standard deviation/percentiles
            # for each grouping.

            for l, lev in enumerate(LEVELS):
                if l <= 2:
                    _u = np.array(
                        [r['run'][LEVELS[l]][idx] / (r['run'][LEVELS[l]][idx]
                                                     + r['run'][LEVELS[l + 3]][
                                                         idx]) for r in
                         res_array])

                else:
                    _u = np.array(
                        [r['run'][LEVELS[l]][idx] / (r['run'][LEVELS[l]][idx]
                                                     + r['run'][LEVELS[l - 3]][
                                                         idx]) for r in
                         res_array])

                self.pct_female_matrix.loc[idx, 'year'] = idx

                self.pct_female_matrix.loc[
                    idx, FEMALE_MATRIX_COLUMNS[2 * l + 1]] \
                    = np.nanmean(_u)

                self.pct_female_matrix.loc[
                    idx, FEMALE_MATRIX_COLUMNS[2 * l + 2]] \
                    = np.nanstd(_u)

                self.pct_female_matrix.loc[idx,
                                           FEMALE_MATRIX_COLUMNS[
                                               12 + 2 * l + 1]] = np.nanpercentile(
                    _u,
                    2.5) if \
                    np.isfinite(np.nanpercentile(_u, 2.5)) else 0

                self.pct_female_matrix.loc[idx,
                                           FEMALE_MATRIX_COLUMNS[
                                               12 + 2 * l + 2]] = np.nanpercentile(
                    _u,
                    97.5) if \
                    np.isfinite(np.nanpercentile(_u, 97.5)) else 0

        self.res_array = res_array

    def run_parameter_sweep(self, number_of_runs, param, llim,
                            ulim, num_of_steps):

        '''

        This function sweeps a single parameter and captures the effect of
        that variation on the overall model. Any valid parameter can be chosen.

        :param number_of_runs: The number of model iterations per parameter
        value
        :type number_of_runs: int
        :param param: The name of the parameter to sweep
        :type param: basestring
        :param llim: lower limit of the parameter value
        :type llim: float
        :param ulim: upper limit of the parameter value
        :type ulim: float
        :param num_of_steps: total number of increments in the range between
        the upper and lower limits.
        :type num_of_steps: int
        :return: a Dataframe containing individual model runs using the
        parameter increments
        :rtype: pandas Dataframe
        '''

        # First I will create a structured array to hold the results of the
        # simulation. The size of the array should be one for each step in the
        # parameter sweep. To calculate that,

        parameter_sweep_increments = np.linspace(llim, ulim, num_of_steps)

        # TODO have to adjust the length of the dataframe to include the
        # extra columns for the EXPORT_COLUMNS_FOR_CSV

        parameter_sweep_results = pd.DataFrame(np.zeros([len(
            parameter_sweep_increments),
            len(RESULTS_COLUMNS) + 1]))

        parameter_sweep_results.loc[:, 0] = parameter_sweep_increments

        # Run simulations with parameter increments and collect into a container.


        for i, val in enumerate(parameter_sweep_increments):
            setattr(self, param, val)
            self.run_multiple(number_of_runs)
            parameter_sweep_results.iloc[i, 1:] = self.results_matrix.tail(
                1).iloc[0, 1:]

        self.parameter_sweep_results = parameter_sweep_results

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

    def export_model_run(self, model_label, model_choice, number_of_runs):

        if not hasattr(self, 'res'):
            self.run_multiple(number_of_runs)

        # first I will allocate the memory by creating an empty dataframe.
        # then I will iterate over the res_array matrix and write to the
        # correct rows of the dataframe. This is more memory efficient compared
        # to appending to a dataframe.

        # print(pd.DataFrame(self.res_array['run'][3]))

        columnnames = ['run', 'year'] + MODEL_RUN_COLUMNS + \
                      EXPORT_COLUMNS_FOR_CSV + ['model_name']

        print_array = np.zeros([self.duration * number_of_runs,
                                len(columnnames)])

        for idx in range(number_of_runs):
            print_array[(idx * self.duration):(idx * self.duration +
                                               self.duration), 0] = idx

            print_array[(idx * self.duration):(idx * self.duration +
                                               self.duration),
            1:-1] = pd.DataFrame(self.res_array['run'][idx])

        # work with barbara to craft the filename
        # model_label + 160114_HH:MM(24hour) +

        filename = model_label + "_" + str(datetime.datetime.now()) + "_iter" \
                   + str(number_of_runs) + ".csv"

        df_print_array = pd.DataFrame(print_array, columns=columnnames).round(2)
        df_print_array.iloc[:, -1] = model_choice
        df_print_array.to_csv(filename)


## Supplementary/Helper functions

def calculate_empirical_probability_of_value(criterion, data_vector):
    emp_prob = 1 - sum(data_vector <= criterion) / len(data_vector)
    return (emp_prob)
