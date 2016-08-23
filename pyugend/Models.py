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


## Initialize Constants

PROFESSOR_LEVEL_NAMES = list(['f1n', 'f2n', 'f3n', 'm1n', 'm2n', 'm3n'])
PROBABILITY_ARRAY_COLUMN_NAMES = list(
    ['param', 'prof_group_mean', 'probability'])


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
        self.model_summary_stats = 0
        self.std_matrix = 0
        self.pd_last_row_data = 0
        self.pct_female_matrix = 0
        self.probability_matrix = 0

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
        df_.columns = ['f1',
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
                       'm_prom_1']
        # print(df_)
        recarray_results = df_.to_records(index=True)
        self.res = recarray_results
        return recarray_results

    def run_multiple(self, number_of_runs):

        res_array = np.recarray((number_of_runs,), dtype=[('run', object)])

        ## Then I need to run a loop to run multiple models and return their values to the record array
        ## TODO fix the function here to reflect additional parameter for model
        # object
        for idx in range(number_of_runs):
            res_array['run'][idx] = self.run_model()

        ##Create empty arrays to hold the results

        self.mean_matrix = np.empty_like(res_array['run'][0])
        self.std_matrix = np.empty_like(res_array['run'][0])
        self.dept_size_matrix = pd.DataFrame(np.zeros([self.duration, 3]))
        self.dept_size_matrix.columns = ['year', 'mean', 'std']
        self.pct_female_matrix = pd.DataFrame(np.zeros([self.duration, 13]))
        self.pct_female_matrix.columns = ['year', 'mpct_f1', 'mpct_f2',
                                          'mpct_f3',
                                          'spct_f1', 'spct_f2', 'spct_f3',
                                          'mpct_m1', 'spct_m1', 'mpct_m2',
                                          'spct_m2', 'mpct_m3', 'spct_m3']
        ## Collect mean and standard deviations for each column/row across all
        # iterations of the model.


        for idx in range(self.duration):
            for f in list(self.std_matrix.dtype.names)[1:]:
                _t = np.array([r['run'][f][idx] for r in res_array])
                self.mean_matrix[f][idx] = np.mean(_t)
                self.std_matrix[f][idx] = np.std(_t)

            _s = np.array([sum(list([r['run']['f1'][idx],
                                     r['run']['f2'][idx],
                                     r['run']['f3'][idx],
                                     r['run']['m1'][idx],
                                     r['run']['m2'][idx],
                                     r['run']['m3'][idx]])) for r in res_array])
            self.dept_size_matrix['year'][idx] = idx
            self.dept_size_matrix['mean'][idx] = _s.mean()
            self.dept_size_matrix['std'][idx] = _s.std()

            _u = np.array([r['run']['f1'][idx] / (r['run']['f1'][idx] + r[
                'run']['m1'][idx]) for r in res_array])
            self.pct_female_matrix.loc[idx, 'year'] = idx
            self.pct_female_matrix.loc[idx, 'mpct_f1'] = _u.mean()
            self.pct_female_matrix.loc[idx, 'spct_f1'] = _u.std()

            _u = np.array([r['run']['f2'][idx] / (r['run']['f2'][idx] + r[
                'run']['m2'][idx]) for r in res_array])
            self.pct_female_matrix.loc[idx, 'mpct_f2'] = _u.mean()
            self.pct_female_matrix.loc[idx, 'spct_f2'] = _u.std()

            _u = np.array([r['run']['f3'][idx] / (r['run']['f3'][idx] + r[
                'run']['m3'][idx]) for r in res_array])
            self.pct_female_matrix.loc[idx, 'mpct_f3'] = _u.mean()
            self.pct_female_matrix.loc[idx, 'spct_f3'] = _u.std()

            _u = np.array([r['run']['m1'][idx] / (r['run']['m1'][idx] + r[
                'run']['f1'][idx]) for r in res_array])
            self.pct_female_matrix.loc[idx, 'mpct_m1'] = _u.mean()
            self.pct_female_matrix.loc[idx, 'spct_m1'] = _u.std()

            _u = np.array([r['run']['m2'][idx] / (r['run']['m2'][idx] + r[
                'run']['f2'][idx]) for r in res_array])
            self.pct_female_matrix.loc[idx, 'mpct_m2'] = _u.mean()
            self.pct_female_matrix.loc[idx, 'spct_m2'] = _u.std()

            _u = np.array([r['run']['m3'][idx] / (r['run']['f3'][idx] + r[
                'run']['m3'][idx]) for r in res_array])
            self.pct_female_matrix.loc[idx, 'mpct_m3'] = _u.mean()
            self.pct_female_matrix.loc[idx, 'spct_m3'] = _u.std()

        # create matrix to hold data for the final iteration of the model. That data
        # holds the distributions of the data. The matrix that holds
        # this data will have columns for each male female groups. The rows will
        # have one entry per run of the model.

        last_row_data = np.zeros(number_of_runs, dtype=[('f1', 'float'),
                                                        ('f2', 'float'),
                                                        ('f3', 'float'),
                                                        ('m1', 'float'),
                                                        ('m2', 'float'),
                                                        ('m3', 'float'),
                                                        ('vac_3', 'float'),
                                                        ('vac_2', 'float'),
                                                        ('vac_1', 'float'),
                                                        ('prom1', 'float'),
                                                        ('prom2', 'float'),
                                                        ('gendprop', 'float'),
                                                        ('unfilled', 'float'),
                                                        ('dept_size', 'float'),
                                                        ('f_hire_3', 'float'),
                                                        ('m_hire_3', 'float'),
                                                        ('f_hire_2', 'float'),
                                                        ('m_hire_2', 'float'),
                                                        ('f_hire_1', 'float'),
                                                        ('m_hire_1', 'float'),
                                                        ('f_prom_3', 'float'),
                                                        ('m_prom_3', 'float'),
                                                        ('f_prom_2', 'float'),
                                                        ('m_prom_2', 'float'),
                                                        ('f_prom_1', 'float'),
                                                        ('m_prom_1', 'float')])

        # Then I will need a second matrix to hold the rows for summary and the
        # columns for the male and female groups.

        summary_stat_list = ['mean', 'standard dev', 'median', 'kurtosis',
                             'skewness']
        self.model_summary_stats = np.zeros(len(summary_stat_list),
                                            dtype=[('statistic',
                                                    'object'),
                                                   ('f1', 'float'),
                                                   ('f2', 'float'),
                                                   ('f3', 'float'),
                                                   ('m1', 'float'),
                                                   ('m2', 'float'),
                                                   ('m3', 'float'),
                                                   ('vac_3', 'float'),
                                                   ('vac_2', 'float'),
                                                   ('vac_1', 'float'),
                                                   ('prom1', 'float'),
                                                   ('prom2', 'float'),
                                                   ('gendprop', 'float'),
                                                   ('unfilled', 'float'),
                                                   ('dept_size', 'float'),
                                                   ('f_hire_3', 'float'),
                                                   ('m_hire_3', 'float'),
                                                   ('f_hire_2', 'float'),
                                                   ('m_hire_2', 'float'),
                                                   ('f_hire_1', 'float'),
                                                   ('m_hire_1', 'float'),
                                                   ('f_prom_3', 'float'),
                                                   ('m_prom_3', 'float'),
                                                   ('f_prom_2', 'float'),
                                                   ('m_prom_2', 'float'),
                                                   ('f_prom_1', 'float'),
                                                   ('m_prom_1', 'float')])

        for r in range(self.model_summary_stats.shape[0]):
            self.model_summary_stats['statistic'][r] = summary_stat_list[r]

        for f in list(self.std_matrix.dtype.names)[1:]:
            _t = np.array([r['run'][f][-1] for r in res_array])
            last_row_data[f] = np.array([r['run'][f][-1] for r in res_array])

        # Produce statistical summaries for the data and write to summary
        # statistics matrix. So take the data from the data matrix column and
        # calculate its summaries and then write them to the corresponding column of
        # the summaries matrix.
        self.pd_last_row_data = pd.DataFrame(last_row_data)
        for f in list(last_row_data.dtype.names):
            self.model_summary_stats[0][f] = self.pd_last_row_data[f].mean()
            self.model_summary_stats[1][f] = self.pd_last_row_data[f].std()
            self.model_summary_stats[2][f] = self.pd_last_row_data[f].median()
            self.model_summary_stats[3][f] = self.pd_last_row_data[f].kurtosis()
            self.model_summary_stats[4][f] = self.pd_last_row_data[f].skew()

        ## Save the original res_array data to the object for use by later analysis

        self.res_array = res_array

        # return(list((self.mean_matrix,self.std_matrix, self.model_summary_stats, self.pd_last_row_data)))

    def run_parameter_sweep(self, number_of_runs, param, llim,
                            ulim, num_of_steps):

        # First I will create a structured array to hold the results of the
        # simulation. The size of the array should be one for each step in the
        # parameter sweep. To calculate that,

        parameter_sweep_increments = np.linspace(llim, ulim, num_of_steps)

        parameter_sweep_container = np.zeros(len(parameter_sweep_increments),
                                             dtype=[(
                                                 'sweep', 'object')])

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
            self.run_multiple(number_of_runs)
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

            parameter_sweep_container['sweep'][i] = model_final_year_results


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

    def plot_multiple_runs_detail(self, num_runs, group_title, target):

        if self.model_summary_stats == 0:
            print("generating multiple runs data.")
            self.run_multiple(num_runs)
        pd_stats_matrix = pd.DataFrame(self.model_summary_stats)
        tmp = pd_stats_matrix.select_dtypes(include=[np.number])
        pd_stats_matrix.loc[:, tmp.columns] = np.round(tmp, 2)

        f, axarr = plt.subplots(nrows=2, ncols=3)
        f.suptitle(group_title)
        axarr[0, 0].plot(range(self.duration), self.mean_matrix['f1'], 2,
                         label=self.label)
        axarr[0, 0].set_title('Female level 1')
        axarr[0, 0].set_xlabel('Years')
        axarr[0, 0].set_ylabel('Number of Females')
        axarr[0, 0].fill_between(range(self.duration), self.mean_matrix['f1'] +
                                 1.96 * self.std_matrix[
                                     'f1'], self.mean_matrix['f1'] - 1.96 *
                                 self.std_matrix[
                                     'f1'], alpha=0.25)
        axarr[0, 0].axhline(y=target, color='r')

        axarr[0, 1].plot(range(self.duration), self.mean_matrix['f2'])
        axarr[0, 1].set_title('Female level 2')
        axarr[0, 1].set_xlabel('Years')
        axarr[0, 1].set_ylabel('Number of Females')
        axarr[0, 1].fill_between(range(self.duration), self.mean_matrix['f2'] +
                                 1.96 * self.std_matrix[
                                     'f2'], self.mean_matrix['f2'] - 1.96 *
                                 self.std_matrix[
                                     'f2'], alpha=0.25)
        axarr[0, 1].axhline(y=target, color='r')

        axarr[0, 2].plot(range(self.duration), self.mean_matrix['f3'])
        axarr[0, 2].set_title('Female level 3')

        axarr[0, 2].set_xlabel('Years')
        axarr[0, 2].set_ylabel('Number of Females')
        axarr[0, 2].fill_between(range(self.duration), self.mean_matrix['f3'] +
                                 1.96 * self.std_matrix[
                                     'f3'], self.mean_matrix['f3'] - 1.96 *
                                 self.std_matrix[
                                     'f3'], alpha=0.25)
        axarr[0, 2].axhline(y=target, color='r')

        axarr[1, 0].plot(range(self.duration), self.mean_matrix['m1'])
        axarr[1, 0].set_title('Male level 1')
        axarr[1, 0].set_xlabel('Years')
        axarr[1, 0].set_ylabel('Number of Males')
        axarr[1, 0].fill_between(range(self.duration), self.mean_matrix['m1'] +
                                 1.96 * self.std_matrix[
                                     'm1'], self.mean_matrix['m1'] - 1.96 *
                                 self.std_matrix[
                                     'm1'], alpha=0.25)
        axarr[1, 0].axhline(y=target, color='r')

        axarr[1, 1].plot(range(self.duration), self.mean_matrix['m2'])
        axarr[1, 1].set_title('Male level 2')
        axarr[1, 1].set_xlabel('Years')
        axarr[1, 1].set_ylabel('Number of Males')
        axarr[1, 1].fill_between(range(self.duration), self.mean_matrix['m2'] +
                                 1.96 * self.std_matrix[
                                     'm2'], self.mean_matrix['m2'] - 1.96 *
                                 self.std_matrix[
                                     'm2'], alpha=0.25)
        axarr[1, 1].axhline(y=target, color='r')

        axarr[1, 2].plot(range(self.duration), self.mean_matrix['m3'])
        axarr[1, 2].set_title('Male level 3')
        axarr[1, 2].set_xlabel('Years')
        axarr[1, 2].set_ylabel('Number of Males')
        axarr[1, 2].fill_between(range(self.duration), self.mean_matrix['m3'] +
                                 1.96 * self.std_matrix[
                                     'm3'], self.mean_matrix['m3'] - 1.96 *
                                 self.std_matrix[
                                     'm3'], alpha=0.25)
        axarr[1, 2].axhline(y=target, color='r')

        plt.show()

    def plot_multiple_runs_detail_percentage(self,
                                             target,
                                             number_of_runs=100,
                                             caption='',
                                             xlabelf1='',
                                             xlabelf2='',
                                             xlabelf3='',
                                             xlabelm1='',
                                             xlabelm2='',
                                             xlabelm3='',
                                             ylabelf1='',
                                             ylabelf2='',
                                             ylabelf3='',
                                             ylabelm1='',
                                             ylabelm2='',
                                             ylabelm3='',
                                             group_title='',
                                             titlef1='',
                                             titlef2='',
                                             titlef3='',
                                             titlem1='',
                                             titlem2='',
                                             titlem3=''):

        if self.model_summary_stats == 0:
            print("generating multiple runs data.")
            self.run_multiple(number_of_runs)
        pd_stats_matrix = pd.DataFrame(self.model_summary_stats)
        tmp = pd_stats_matrix.select_dtypes(include=[np.number])
        pd_stats_matrix.loc[:, tmp.columns] = np.round(tmp, 2)
        # self.pct_female_matrix.replace(np.inf, 1)
        # self.pct_female_matrix = self.pct_female_matrix.fillna(0)

        f, axarr = plt.subplots(nrows=2, ncols=3)
        f.suptitle(group_title)
        axarr[0, 0].plot(range(self.duration),
                         np.minimum(1, np.maximum(0, self.pct_female_matrix[
                             'mpct_f1'])), label=self.label)
        axarr[0, 0].set_title(titlef1)
        axarr[0, 0].set_xlabel(xlabelf1)
        axarr[0, 0].set_ylabel(ylabelf1)
        axarr[0, 0].fill_between(range(self.duration), np.minimum(1,
                                                                  self.pct_female_matrix[
                                                                      'mpct_f1'] +
                                                                  1.96 *
                                                                  self.pct_female_matrix[
                                                                      'spct_f1']),
                                 np.maximum(0, self.pct_female_matrix[
                                     'mpct_f1'] - 1.96 * self.pct_female_matrix[
                                                'spct_f1']), alpha=0.25)
        axarr[0, 0].axhline(y=target, color='r')

        axarr[0, 1].plot(range(self.duration), np.minimum(1, np.maximum(0,
                                                                        self.pct_female_matrix[
                                                                            'mpct_f2'])))
        axarr[0, 1].set_title(titlef2)
        axarr[0, 1].set_xlabel(xlabelf2)
        axarr[0, 1].set_ylabel(ylabelf2)
        axarr[0, 1].fill_between(range(self.duration), np.minimum(1,
                                                                  self.pct_female_matrix[
                                                                      'mpct_f2'] +
                                                                  1.96 *
                                                                  self.pct_female_matrix[
                                                                      'spct_f2']),
                                 np.maximum(0, self.pct_female_matrix[
                                     'mpct_f2'] - 1.96 *
                                            self.pct_female_matrix[
                                                'spct_f2']), alpha=0.25)
        axarr[0, 1].axhline(y=target, color='r')

        axarr[0, 2].plot(range(self.duration), np.minimum(1, np.maximum(0,
                                                                        self.pct_female_matrix[
                                                                            'mpct_f3'])))
        axarr[0, 2].set_title(titlef3)

        axarr[0, 2].set_xlabel(xlabelf3)
        axarr[0, 2].set_ylabel(ylabelf3)
        axarr[0, 2].fill_between(range(self.duration), np.minimum(1,
                                                                  self.pct_female_matrix[
                                                                      'mpct_f3'] +
                                                                  1.96 *
                                                                  self.pct_female_matrix[
                                                                      'spct_f3']),
                                 np.maximum(0, self.pct_female_matrix[
                                     'mpct_f3'] - 1.96 *
                                            self.pct_female_matrix[
                                                'spct_f3']), alpha=0.25)
        axarr[0, 2].axhline(y=target, color='r')

        axarr[1, 0].plot(range(self.duration), np.minimum(1, np.maximum(0,
                                                                        self.pct_female_matrix[
                                                                            'mpct_m1'])))
        axarr[1, 0].set_title(titlem1)
        axarr[1, 0].set_xlabel(xlabelm1)
        axarr[1, 0].set_ylabel(ylabelm1)
        axarr[1, 0].fill_between(range(self.duration), np.minimum(1,
                                                                  self.pct_female_matrix[
                                                                      'mpct_m1'] +
                                                                  1.96 *
                                                                  self.pct_female_matrix[
                                                                      'spct_m1']),
                                 np.maximum(0, self.pct_female_matrix[
                                     'mpct_m1'] - 1.96 *
                                            self.pct_female_matrix[
                                                'spct_m1']), alpha=0.25)
        axarr[1, 0].axhline(y=1 - target, color='r')

        axarr[1, 1].plot(range(self.duration), np.minimum(1, np.maximum(0,
                                                                        self.pct_female_matrix[
                                                                            'mpct_m2'])))
        axarr[1, 1].set_title(titlem2)
        axarr[1, 1].set_xlabel(xlabelm2)
        axarr[1, 1].set_ylabel(ylabelm2)
        axarr[1, 1].fill_between(range(self.duration), np.minimum(1,
                                                                  self.pct_female_matrix[
                                                                      'mpct_m2'] +
                                                                  1.96 *
                                                                  self.pct_female_matrix[
                                                                      'spct_m2']),
                                 np.maximum(0, self.pct_female_matrix[
                                     'mpct_m2'] - 1.96 *
                                            self.pct_female_matrix[
                                                'spct_m2']), alpha=0.25)
        axarr[1, 1].axhline(y=1 - target, color='r')

        axarr[1, 2].plot(range(self.duration), np.minimum(1, np.maximum(0,
                                                                        self.pct_female_matrix[
                                                                            'mpct_m3'])))
        axarr[1, 2].set_title(titlem3)
        axarr[1, 2].set_xlabel(xlabelm3)
        axarr[1, 2].set_ylabel(ylabelm3)
        axarr[1, 2].fill_between(range(self.duration), np.minimum(1,
                                                                  self.pct_female_matrix[
                                                                      'mpct_m3'] +
                                                                  1.96 *
                                                                  self.pct_female_matrix[
                                                                      'spct_m3']),
                                 np.maximum(0, self.pct_female_matrix[
                                     'mpct_m3'] - 1.96 *
                                            self.pct_female_matrix[
                                                'spct_m3']), alpha=0.25)
        axarr[1, 2].axhline(y=1 - target, color='r')
        f.text(0.2, 0.2, caption)

        plt.show()

    def plot_multiple_runs_gender_prop(self, title, xlabel,
                                       ylabel, target, txt, num_runs=100):

        if self.mean_matrix == 0:
            print("generating multiple runs data.")
            self.run_multiple(num_runs)

        plt.plot(range(self.duration), self.mean_matrix['gendprop'],
                 label=self.label)
        plt.fill_between(range(self.duration), self.mean_matrix[
            'gendprop'] +
                         1.96 * self.std_matrix[
                             'gendprop'],
                         self.mean_matrix['gendprop'] - 1.96 * self.std_matrix[
                             'gendprop'], alpha=0.25)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axhline(y=target, color='r')
        plt.legend(loc='upper right', shadow=True)
        plt.text(0.02, 0.02, txt)

        plt.show()

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

    def plot_empirical_probability_gender_proportion(self, num_runs, target,
                                                     title, xlabel, ylabel,
                                                     txt):

        self.run_probability_analysis_gender_proportion(num_runs, target)

        plt.plot(self.probability_matrix['Year'],
                 self.probability_matrix['Probability'])
        plt.axhline(y=0.5, color='r')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.text(0.02, 0.02, txt)
        plt.show()

    def plot_empirical_probability_analysis_by_level(self,
                                                     target,
                                                     number_of_runs,
                                                     caption='',
                                                     xlabelf1='',
                                                     xlabelf2='',
                                                     xlabelf3='',
                                                     xlabelm1='',
                                                     xlabelm2='',
                                                     xlabelm3='',
                                                     ylabelf1='',
                                                     ylabelf2='',
                                                     ylabelf3='',
                                                     ylabelm1='',
                                                     ylabelm2='',
                                                     ylabelm3='',
                                                     group_title='',
                                                     titlef1='',
                                                     titlef2='',
                                                     titlef3='',
                                                     titlem1='',
                                                     titlem2='',
                                                     titlem3=''):

        d = self.run_probability_analysis_gender_by_level(number_of_runs,
                                                          target)

        f, axarr = plt.subplots(nrows=2, ncols=3)
        f.suptitle(group_title)
        axarr[0, 0].plot(range(self.duration), d['pf1'], label=self.label)
        axarr[0, 0].set_title(titlef1)
        axarr[0, 0].set_xlabel(xlabelf1)
        axarr[0, 0].set_ylabel(ylabelf1)
        axarr[0, 0].axhline(y=0.5, color='r')

        axarr[0, 1].plot(range(self.duration), d['pf2'], label=self.label)
        axarr[0, 1].set_title(titlef2)
        axarr[0, 1].set_xlabel(xlabelf2)
        axarr[0, 1].set_ylabel(ylabelf2)
        axarr[0, 1].axhline(y=0.5, color='r')

        axarr[0, 2].plot(range(self.duration), d['pf3'], label=self.label)
        axarr[0, 2].set_title(titlef3)
        axarr[0, 2].set_xlabel(xlabelf3)
        axarr[0, 2].set_ylabel(ylabelf3)
        axarr[0, 2].axhline(y=0.5, color='r')

        axarr[1, 0].plot(range(self.duration), d['pm1'], label=self.label)
        axarr[1, 0].set_title(titlem1)
        axarr[1, 0].set_xlabel(xlabelm1)
        axarr[1, 0].set_ylabel(ylabelm1)
        axarr[1, 0].axhline(y=0.5, color='r')

        axarr[1, 1].plot(range(self.duration), d['pm2'], label=self.label)
        axarr[1, 1].set_title(titlem2)
        axarr[1, 1].set_xlabel(xlabelm2)
        axarr[1, 1].set_ylabel(ylabelm2)
        axarr[1, 1].axhline(y=0.5, color='r')

        axarr[1, 2].plot(range(self.duration), d['pm3'], label=self.label)
        axarr[1, 2].set_title(titlem3)
        axarr[1, 2].set_xlabel(xlabelm3)
        axarr[1, 2].set_ylabel(ylabelm3)
        axarr[1, 2].axhline(y=0.5, color='r')

        plt.show()

    def plot_empirical_probability_group_detail(self, number_of_runs, param,
                                                prof_group, llim,
                                                ulim, num_of_steps, target,
                                                title, xlabel, ylabel):

        d = self.run_probability_analysis_parameter_sweep_gender_detail(
            number_of_runs, param, prof_group, llim,
            ulim, num_of_steps, target)

        plt.plot(d['param'], d['probability'])
        plt.plot(range(self.duration), 0.5, color='red')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_department_size_over_time_multiple_runs(self, num_runs,
                                                     title, xlable,
                                                     ylabel):

        '''
        This function will calculate the average number of individuals in the department in each year and provide a chart forthe results.
        Returns:

        '''

        self.run_multiple(num_runs)

        plt.plot(self.dept_size_matrix['year'], self.dept_size_matrix['mean'])
        plt.fill_between(range(self.duration), self.dept_size_matrix[
            'mean'] +
                         1.96 * self.dept_size_matrix[
                             'std'], self.dept_size_matrix['mean'] - 1.96 *
                         self.dept_size_matrix[
                             'std'], alpha=0.25)
        plt.xlabel(xlable)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plot_unfilled_vacancies_over_time_multiple_runs(self, num_runs,
                                                        title, xlable,
                                                        ylabel):
        '''
        This function will calculate the average number of individuals in the department in each year and provide
        chart forthe results.
        Returns:
        '''
        self.run_multiple(num_runs)

        plt.plot(range(self.duration), self.mean_matrix['unfilled'])
        plt.fill_between(range(self.duration), self.mean_matrix[
            'unfilled'] +
                         1.96 * self.std_matrix[
                             'unfilled'], self.mean_matrix['unfilled'] - 1.96 *
                         self.std_matrix[
                             'unfilled'], alpha=0.25)
        plt.xlabel(xlable)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plot_male_female_total_numbers(self, xlabel, ylabel, title, txt, target,
                                       num_runs):

        if self.mean_matrix == 0:
            print("generating multiple runs data.")
            self.run_multiple(num_runs)

        plt.plot(range(self.duration), sum(list([self.mean_matrix['f1'],
                                                 self.mean_matrix['f2'],
                                                 self.mean_matrix['f3']])),
                 color='b', label='Women')
        plt.plot(range(self.duration), sum(list([self.mean_matrix['m1'],
                                                 self.mean_matrix['m2'],
                                                 self.mean_matrix['m3']])),
                 color='k', label='Men')

        total_faculty = self.mean_matrix['f1'] \
                        + self.mean_matrix['f2'] \
                        + self.mean_matrix['f3'] \
                        + self.mean_matrix['m1'] \
                        + self.mean_matrix['m2'] \
                        + self.mean_matrix['m3']

        plt.plot(range(self.duration), np.round(target * total_faculty),
                 color='r', label='Target')

        plt.title(title)
        plt.text(0.02, 0.02, txt)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='upper right', shadow=True)
        plt.show()

    def export_model_run(self, number_of_runs):

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
        filename = self.label + "_" + str(datetime.datetime.now()) + "_iter" \
                   + str(number_of_runs) + ".xls"
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
                                           'men_promoted_1']).to_excel(filename)


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
                           xmax,
                           ymax,
                           alpha_val,
                           marker_val,
                           color_val,
                           target_plot,
                           legend_location,
                           color_target,
                           fifty_percent_plot,
                           color_fifty_percent,
                           target_plot_line_style,
                           fifty_percent_line_style
                           ):



        # generate data for the plot.
        self.run_multiple(number_of_runs)

        # set default plot parameters. The xaxis is generally duration,
        # though I have the option--depending on the plot, to put in a
        # different x-axis.

        xval = self.duration

        if plottype == 'probability proportion':

            self.run_probability_analysis_gender_proportion(number_of_runs, target)

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


        plt.plot(range(xval), yval, linewidth=line_width,
                 marker = marker_val, color = color_val, label = self.label)

        plt.fill_between(range(xval),
                         yval + 1.96*fill_matrix,
                         yval - 1.96*fill_matrix,
                         alpha=alpha_val)

        if target_plot == True:
            plt.axhline(target, color = color_target, linestyle = target_plot_line_style, label = 'Target percentage')

        if fifty_percent_plot == True:
            plt.axhline(y = 0.5, color = color_fifty_percent, linestyle = fifty_percent_line_style, label = '50% line')

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc=legend_location, shadow=True)
        plt.show()


    def plot_level_chart(self,
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
                           alpha_val,
                           marker_val,
                           color_val,
                           color_target,
                           target_plot,
                           legend_location,
                           legend_label
                           ):



        # generate data for the plot.

        self.run_multiple(number_of_runs)

        # set default plot parameters. The xaxis is generally duration,
        # though I have the option--depending on the plot, to put in a
        # different x-axis.

        xval = self.duration

        if plottype == 'probability proportion':

            self.run_probability_analysis_gender_proportion(number_of_runs, target)

            yval = self.probability_matrix['unfilled']
            fill_matrix = 0

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
            pass



        f, axarr = plt.subplots(nrows=2, ncols=3)
        f.suptitle(group_title)
        axarr[0, 0].plot(range(xval),
                         np.minimum(1,
                         np.maximum(0,
                         yval_f1)),
                         label=legend_label,
                         linewidth = line_width,
                         color = color_val,
                         marker = marker_val)
        axarr[0, 0].set_xlim([xlim_min_f1, xlim_max_f1])
        axarr[0, 0].set_ylim([ylim_min_f1, ylim_max_f1])
        axarr[0, 0].set_title(titlef1)
        axarr[0, 0].set_xlabel(xlabelf1)
        axarr[0, 0].set_ylabel(ylabelf1)
        axarr[0, 0].fill_between(range(xval),
                                 np.minimum(1,
                                 yval_f1 + 1.96 *fill_f1),
                                 np.maximum(0,
                                 yval_f1 - 1.96 *fill_f1),
                                 alpha=alpha_val,
                                 facecolor = color_val)
        axarr[0, 0].legend(loc=legend_location, shadow=True)

        axarr[0, 1].plot(range(xval),
                         np.minimum(1,
                         np.maximum(0,
                         yval_f1)),
                         label=self.label,
                         linewidth=line_width,
                         color=color_val,
                         marker=marker_val
                         )
        axarr[0, 0].set_xlim([xlim_min_f2, xlim_max_f2])
        axarr[0, 0].set_ylim([ylim_min_f2, ylim_max_f2])
        axarr[0, 1].set_title(titlef2)
        axarr[0, 1].set_xlabel(xlabelf2)
        axarr[0, 1].set_ylabel(ylabelf2)
        axarr[0, 1].fill_between(range(xval),
                                 np.minimum(1,
                                 yval_f2 + 1.96 *fill_f2),
                                 np.maximum(0,
                                 yval_f2 - 1.96 * fill_f2),
                                 alpha=alpha_val,
                                 facecolor=color_val)




        axarr[0, 2].plot(range(xval),
                         np.minimum(1,
                         np.maximum(0,
                         yval_f3)),
                         label=self.label,
                         linewidth=line_width,
                         color=color_val,
                         marker=marker_val
                         )
        axarr[0, 0].set_xlim([xlim_min_f3, xlim_max_f3])
        axarr[0, 0].set_ylim([ylim_min_f3, ylim_max_f3])
        axarr[0, 2].set_title(titlef3)
        axarr[0, 2].set_xlabel(xlabelf3)
        axarr[0, 2].set_ylabel(ylabelf3)
        axarr[0, 2].fill_between(range(xval),
                                 np.minimum(1,
                                 yval_f3 + 1.96 *fill_f3),
                                 np.maximum(0,
                                 yval_f3 - 1.96 * fill_f3),
                                 alpha=alpha_val,
                                 facecolor=color_val)


        axarr[1, 0].plot(range(xval),
                         np.minimum(1,
                         np.maximum(0,
                         yval_m1)),
                         label=self.label,
                         linewidth=line_width,
                         color=color_val,
                         marker=marker_val
                         )
        axarr[0, 0].set_xlim([xlim_min_m1, xlim_max_m1])
        axarr[0, 0].set_ylim([ylim_min_m1, ylim_max_m1])
        axarr[1, 0].set_title(titlem1)
        axarr[1, 0].set_xlabel(xlabelm1)
        axarr[1, 0].set_ylabel(ylabelm1)
        axarr[1, 0].fill_between(range(xval),
                                 np.minimum(1,
                                 yval_m1 + 1.96 *fill_m1),
                                 np.maximum(0,
                                 yval_m1 - 1.96 * fill_m1),
                                 alpha=alpha_val,
                                 facecolor=color_val)


        axarr[1, 1].plot(range(xval),
                         np.minimum(1,
                         np.maximum(0,
                         yval_m2)),
                         label=self.label,
                         linewidth=line_width,
                         color=color_val,
                         marker=marker_val
                         )
        axarr[0, 0].set_xlim([xlim_min_m2, xlim_max_m2])
        axarr[0, 0].set_ylim([ylim_min_m2, ylim_max_m2])
        axarr[1, 1].set_title(titlem2)
        axarr[1, 1].set_xlabel(xlabelm2)
        axarr[1, 1].set_ylabel(ylabelm2)
        axarr[1, 1].fill_between(range(xval),
                                 np.minimum(1,
                                 yval_m2 + 1.96 *fill_m2),
                                 np.maximum(0,
                                 yval_m2 - 1.96 * fill_m2),
                                 alpha=alpha_val,
                                 facecolor=color_val)


        axarr[1, 2].plot(range(xval),
                         np.minimum(1,
                         np.maximum(0,
                         yval_m3)),
                         label=self.label,
                         linewidth=line_width,
                         color=color_val,
                         marker=marker_val
                         )
        axarr[0, 0].set_xlim([xlim_min_m3, xlim_max_m3])
        axarr[0, 0].set_ylim([ylim_min_m3, ylim_max_m3])
        axarr[1, 2].set_title(titlem3)
        axarr[1, 2].set_xlabel(xlabelf1)
        axarr[1, 2].set_ylabel(ylabelf1)
        axarr[1, 2].fill_between(range(xval),
                                 np.minimum(1,
                                 yval_f1 + 1.96 *fill_m3),
                                 np.maximum(0,
                                 yval_f1 - 1.96 * fill_m3),
                                 alpha=alpha_val,
                                 facecolor=color_val)

        if target_plot == True:

            axarr[0, 0].axhline(y=target, color= color_target)
            axarr[0, 1].axhline(y=target, color=color_target)
            axarr[0, 2].axhline(y=target, color=color_target)
            axarr[1, 0].axhline(y=1 - target, color=color_target)
            axarr[1, 1].axhline(y=1 - target, color=color_target)
            axarr[1, 2].axhline(y=1 - target, color=color_target)








        plt.show()






## Supplementary/Helper functions

def calculate_empirical_probability_of_value(criterion, data_vector):
    emp_prob = 1 - sum(data_vector <= criterion) / len(data_vector)
    return (emp_prob)
