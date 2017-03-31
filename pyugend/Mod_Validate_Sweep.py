"""
Stochastic Model FBPH
---------------------

This model follows allows for the department to hire and then promote
faculty.




Notes:

5/7/2016 - This is an alternative model for the range models. It is based
upon allowing the size of the department to vary but with greater probability
for male hires. So I will see how that goes.
"""

__author__ = 'krishnab'
__version__ = '0.1.0'

from operator import neg, truediv
import numpy as np
import pandas as pd
from numpy.random import binomial
from pyugend.pyugend.Models import Base_model
from random import random, uniform

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


class Mod_Validate_Sweep(Base_model):
    def __init__(self, **kwds):
        Base_model.__init__(self, **kwds)
        self.name = "validate sweep"
        self.label = "validate sweep"

    def run_model(self):

        ## initialize data structure

        self.res = np.zeros([self.duration,
                             len(MODEL_RUN_COLUMNS) +
                             len(EXPORT_COLUMNS_FOR_CSV)], dtype=np.float32)

        self.res[0, :] = 0

        for i in range(1, self.duration):
            # run base static simulation.

            self.res[i, 0] = self.bf1
            self.res[i, 1] = self.bf1
            self.res[i, 2] =self.bf1
            self.res[i, 3] =self.bf1
            self.res[i, 4] =self.bf1
            self.res[i, 5] =self.bf1
            self.res[i, 6] = self.phire2
            self.res[i, 7] =self.phire2
            self.res[i, 8] =self.phire2
            self.res[i, 9] =self.phire2
            self.res[i, 10] =self.phire2

            # Gender proportion overall col 11

            self.res[i, 11] = self.bf1 + uniform(0,0.1)
            self.res[i, 12] =self.female_promotion_probability_1
            self.res[i, 13] =self.female_promotion_probability_1
            self.res[i, 14] =self.female_promotion_probability_1
            self.res[i, 15] =self.female_promotion_probability_1
            self.res[i, 16] =self.female_promotion_probability_1
            self.res[i, 17] =self.df1
            self.res[i, 18] =self.df1
            self.res[i, 19] =self.df1
            self.res[i, 20] =self.df1
            self.res[i, 21] =self.df1
            self.res[i, 22] =self.df1
            self.res[i, 23] =self.df1
            self.res[i, 24] =self.df1
            self.res[i, 25] =self.df1
            self.res[i, 26] = self.bf1
            self.res[i, 27] = self.bf2
            self.res[i, 28] = self.bf3
            self.res[i, 29] = 1 - self.bf1
            self.res[i, 30] = 1 - self.bf2
            self.res[i, 31] = 1 - self.bf3
            self.res[i, 32] = self.df1
            self.res[i, 33] = self.df2
            self.res[i, 34] = self.df3
            self.res[i, 35] = self.dm1
            self.res[i, 36] = self.dm2
            self.res[i, 37] = self.dm3
            self.res[i, 38] = 1
            self.res[i, 39] = self.phire2
            self.res[i, 40] = self.phire3
            self.res[i, 41] = self.female_promotion_probability_1
            self.res[i, 42] = self.female_promotion_probability_2
            self.res[i, 43] = 1 - self.female_promotion_probability_1
            self.res[i, 44] = 1 - self.female_promotion_probability_2
            self.res[i, 45] = self.upperbound
            self.res[i, 46] = self.lowerbound
            self.res[i, 47] = self.variation_range
            self.res[i, 48] = self.duration

            # this produces an array of values. Then I need to assign the
            # values to levels. So if I have say a range of variation of 5. I
            #  will get something like [-1,0,1,-1,0] or something. I need to
            # turn this into something like [2,-1,0]. That means randomly
            # assigning the values in the array to levels.


        df_ = pd.DataFrame(self.res)
        df_.columns = MODEL_RUN_COLUMNS + EXPORT_COLUMNS_FOR_CSV

        # print(df_)
        recarray_results = df_.to_records(index=True)
        self.run = recarray_results
        return recarray_results
