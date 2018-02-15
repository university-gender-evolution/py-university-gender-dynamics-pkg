"""
Stochastic Model Gender Diversity
---------------------

This is the second generation of the gender diversity model.





Notes:
09/14/2017 - This is the first version of the second generate model.
"""
__author__ = 'krishnab'
__version__ = '0.1.0'
import numpy as np
import pandas as pd
from numpy.random import binomial, multinomial
from .Models import Base_model

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


class Model3GenderDiversity(Base_model):
    def __init__(self, **kwds):
        Base_model.__init__(self, **kwds)
        self.name = "model-3-baseline"
        self.label = "model-3-baseline"


    def init_hiring_rates(self,
                          _hiring_rate_f1,
                          _hiring_rate_f2,
                          _hiring_rate_f3,
                          _hiring_rate_m1,
                          _hiring_rate_m2,
                          _hiring_rate_m3):

        self.hiring_rate_f1 = _hiring_rate_f1
        self.hiring_rate_f2 = _hiring_rate_f2
        self.hiring_rate_f3 = _hiring_rate_f3
        self.hiring_rate_m1 = _hiring_rate_m1
        self.hiring_rate_m2 = _hiring_rate_m2
        self.hiring_rate_m3 = _hiring_rate_m3

    def init_default_hiring_rate(self):

        self.hiring_rate_f1 = 5/40
        self.hiring_rate_f2 = 2/40
        self.hiring_rate_f3 = 1/40
        self.hiring_rate_m1 = 24/40
        self.hiring_rate_m2 = 3/40
        self.hiring_rate_m3 = 5/40


    def run_model(self):

        # initialize data structure

        self.res = np.zeros([self.duration,
                            len(MODEL_RUN_COLUMNS) +
                            len(EXPORT_COLUMNS_FOR_CSV)],
                            dtype=np.float32)
        self.res[0, 0] = self.nf1
        self.res[0, 1] = self.nf2
        self.res[0, 2] = self.nf3
        self.res[0, 3] = self.nm1
        self.res[0, 4] = self.nm2
        self.res[0, 5] = self.nm3
        self.res[0, 6] = self.vac3
        self.res[0, 7] = self.vac2
        self.res[0, 8] = self.vac1
        self.res[0, 9] = self.female_promotion_probability_1
        self.res[0, 10] = self.female_promotion_probability_2
        self.res[0, 11] = self.res[0, 0:3].sum()/self.res[0, 0:6].sum()
        self.res[0, 12] = 0
        self.res[0, 13] = self.res[0, 0:6].sum()
        self.res[0, 14:] = 0

        # I assign the state variables to temporary variables. That way I
        # don't have to worry about overwriting the original state variables.

        attrition_rate_female_level_1 = self.df1
        attrition_rate_female_level_2 = self.df2
        attrition_rate_female_level_3 = self.df3
        attrition_rate_male_level_1 = self.dm1
        attrition_rate_male_level_2 = self.dm2
        attrition_rate_male_level_3 = self.dm3
        female_promotion_probability_1_2 = self.female_promotion_probability_1
        female_promotion_probability_2_3 = self.female_promotion_probability_2
        male_promotion_probability_1_2 = self.male_promotion_probability_1
        male_promotion_probability_2_3 = self.male_promotion_probability_2
        department_size_upper_bound = self.upperbound
        department_size_lower_bound = self.lowerbound
        variation_range = self.variation_range
        unfilled_vacanies = 0
        extra_vacancies=0

        for i in range(1, self.duration):
            # initialize variables for this iteration

            prev_number_of_females_level_1 = self.res[i - 1, 0]
            prev_number_of_females_level_2 = self.res[i - 1, 1]
            prev_number_of_females_level_3 = self.res[i - 1, 2]
            prev_number_of_males_level_1 = self.res[i - 1, 3]
            prev_number_of_males_level_2 = self.res[i - 1, 4]
            prev_number_of_males_level_3 = self.res[i - 1, 5]
            department_size = self.res[i - 1, 0:6].sum()
            # Process Model
            # attrition process
            female_attrition_level_1 = binomial(prev_number_of_females_level_1,
                                                attrition_rate_female_level_1)
            female_attrition_level_2 = binomial(prev_number_of_females_level_2,
                                                attrition_rate_female_level_2)
            female_attrition_level_3 = binomial(prev_number_of_females_level_3,
                                                attrition_rate_female_level_3)
            male_attrition_level_1 = binomial(prev_number_of_males_level_1,
                                              attrition_rate_male_level_1)
            male_attrition_level_2 = binomial(prev_number_of_males_level_2,
                                              attrition_rate_male_level_2)
            male_attrition_level_3 = binomial(prev_number_of_males_level_3,
                                              attrition_rate_male_level_3)
            # update model numbers
            self.res[i, 0] = self.res[i-1, 0] - female_attrition_level_1
            self.res[i, 1] = self.res[i-1, 1] - female_attrition_level_2
            self.res[i, 2] = self.res[i-1, 2] - female_attrition_level_3
            self.res[i, 3] = self.res[i-1, 3] - male_attrition_level_1
            self.res[i, 4] = self.res[i-1, 4] - male_attrition_level_2
            self.res[i, 5] = self.res[i-1, 5] - male_attrition_level_3

            # get total number of vacancies based on attrition
            subtotal_vacancies_1 = female_attrition_level_1 \
                + male_attrition_level_1
            subtotal_vacancies_2 = female_attrition_level_2 \
                + male_attrition_level_2
            subtotal_vacancies_3 = female_attrition_level_3 \
                + male_attrition_level_3
            total_vacancies = subtotal_vacancies_3 \
                + subtotal_vacancies_2 + subtotal_vacancies_1

            total_vacancies = max(total_vacancies+extra_vacancies, 0)

            # process promotions
            promotions_of_females_level_2_3 = binomial(self.res[i, 1],
                                        female_promotion_probability_2_3)
            promotions_of_males_level_2_3 = binomial(self.res[i, 4],
                                        male_promotion_probability_2_3)
            promotions_of_females_level_1_2 = binomial(self.res[i, 0],
                                        female_promotion_probability_1_2)
            promotions_of_males_level_1_2 = binomial(self.res[i, 3],
                                        male_promotion_probability_1_2)

            # update model numbers
            # add promotions to levels
            self.res[i, 1] += promotions_of_females_level_1_2
            self.res[i, 2] += promotions_of_females_level_2_3
            self.res[i, 4] += promotions_of_males_level_1_2
            self.res[i, 5] += promotions_of_males_level_2_3

            # remove the promoted folks from previous level
            self.res[i, 0] -= promotions_of_females_level_1_2
            self.res[i, 1] -= promotions_of_females_level_2_3
            self.res[i, 3] -= promotions_of_males_level_1_2
            self.res[i, 4] -= promotions_of_males_level_2_3

            # hiring of new faculty
            hires = multinomial(total_vacancies,
                              [self.hiring_rate_f1,
                               self.hiring_rate_f2,
                               self.hiring_rate_f3,
                               self.hiring_rate_m1,
                               self.hiring_rate_m2,
                               self.hiring_rate_m3])

            self.res[i, 0] += hires[0]
            self.res[i, 1] += hires[1]
            self.res[i, 2] += hires[2]
            self.res[i, 3] += hires[3]
            self.res[i, 4] += hires[4]
            self.res[i, 5] += hires[5]

            # fill in summary info for model run

            # capture attrition level 3
            self.res[i, 6] = sum(list([
                male_attrition_level_3,
                female_attrition_level_3]))

            # capture attrition level 2
            self.res[i, 7] = sum(list([
                male_attrition_level_2,
                female_attrition_level_2]))

            # capture attrition level 1
            self.res[i, 8] = sum(list([
                male_attrition_level_1,
                female_attrition_level_1]))

            # capture female promotion probabilities
            self.res[i, 9] = 0
            self.res[i, 10] = 0

            # capture gender proportion for department
            self.res[i, 11] = self.res[i, 0:3].sum()/self.res[i,0:6].sum()

            # capture number of unfilled vacancies as the department size in
            # the last time-step versus the current department size (summing
            # all professor groups). If there is a difference then some
            # vacancies were not filled. This is not a good metric to monitor
            # when using a growth model because the department size is supposed
            # to change from time-step to timestep.
            unfilled_vacanies = abs(department_size - self.res[i, 0:6].sum())
            self.res[i, 12] = unfilled_vacanies

            # capture the current department size.
            department_size = self.res[i, 0:6].sum()
            self.res[i, 13] = department_size

            # capture the number of hires for each group.
            self.res[i, 14] = hires[2]
            self.res[i, 15] = hires[5]
            self.res[i, 16] = hires[1]
            self.res[i, 17] = hires[4]
            self.res[i, 18] = hires[0]
            self.res[i, 19] = hires[3]

            # capture promotions for each group. Since we cannot
            # have promotions from level 3 (full professor), these are set to
            # zero by default.
            self.res[i, 20] = 0
            self.res[i, 21] = 0
            self.res[i, 22] = promotions_of_females_level_2_3
            self.res[i, 23] = promotions_of_males_level_2_3
            self.res[i, 24] = promotions_of_females_level_1_2
            self.res[i, 25] = promotions_of_males_level_1_2

            # capture the hiring rate parameters for each group
            self.res[i, 26] = self.hiring_rate_f1
            self.res[i, 27] = self.hiring_rate_f2
            self.res[i, 28] = self.hiring_rate_f3
            self.res[i, 29] = self.hiring_rate_m1
            self.res[i, 30] = self.hiring_rate_m2
            self.res[i, 31] = self.hiring_rate_m3

            # capture the attrition rate parameters for each group
            self.res[i, 32] = attrition_rate_female_level_1
            self.res[i, 33] = attrition_rate_female_level_2
            self.res[i, 34] = attrition_rate_female_level_3
            self.res[i, 35] = attrition_rate_male_level_1
            self.res[i, 36] = attrition_rate_male_level_2
            self.res[i, 37] = attrition_rate_male_level_3

            # capture the probability of outside hire
            # this parameter is not used in the model any more,
            # however this was something used in the original
            # simulation. This data is preserved for historical
            # significance, but otherwise serves no purpose. 
            self.res[i, 38] = 1
            self.res[i, 39] = 1
            self.res[i, 40] = 1

            # capture the promotion probabilities for each group
            self.res[i, 41] = female_promotion_probability_1_2
            self.res[i, 42] = female_promotion_probability_2_3
            self.res[i, 43] = male_promotion_probability_1_2
            self.res[i, 44] = male_promotion_probability_2_3

            # capture the department size bounds and variation ranges. 
            self.res[i, 45] = department_size_upper_bound
            self.res[i, 46] = department_size_lower_bound
            self.res[i, 47] = variation_range

            # capture the model duration, or the number of time-steps
            self.res[i, 48] = self.duration


            # this produces an array of values. Then I need to assign the
            # values to levels. So if I have say a range of variation of 5. I
            #  will get something like [-1,0,1,-1,0] or something. I need to
            # turn this into something like [2,-1,0]. That means randomly
            # assigning the values in the array to levels.

            flag = False
            while flag == False:

                changes = np.random.choice([-1, 0, 1], variation_range)

                # [-1, -1, 0] gains/losses -- where to apply these?
                # levels {1,2,3}, pick randomly from this set
                # [1,1,2]
                # matching wise [(-1, 1), (-1, 1), (0, 2)]

                if (department_size + changes.sum() <=
                        department_size_upper_bound and department_size +
                    changes.sum() >= department_size_lower_bound):
                    extra_vacancies = changes.sum()
                    flag = True

                if (department_size > department_size_upper_bound):
                    extra_vacancies = 0
                    flag = True

                if department_size < department_size_lower_bound:
                    extra_vacancies = variation_range
                    flag = True

        df_ = pd.DataFrame(self.res)
        df_.columns = MODEL_RUN_COLUMNS + EXPORT_COLUMNS_FOR_CSV
        recarray_results = df_.to_records(index=True)
        self.run = recarray_results
        return recarray_results


