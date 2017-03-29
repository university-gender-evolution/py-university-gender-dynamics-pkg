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
from pyugend.Models import Base_model

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


class Mod_Stoch_FBPH(Base_model):
    def __init__(self, **kwds):
        Base_model.__init__(self, **kwds)
        self.name = "Promote-Hire baseline"
        self.label = "Promote-Hire baseline"

    def run_model(self):

        ## initialize data structure

        self.res = np.zeros([self.duration,
                             len(MODEL_RUN_COLUMNS) +
                             len(
                                 EXPORT_COLUMNS_FOR_CSV)],
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
        self.res[0, 11] = np.float32(
            sum(list([self.nf1, self.nf2, self.nf3])) / sum(list([self.nf1,
                                                                  self.nf2,
                                                                  self.nf3,
                                                                  self.nm1,
                                                                  self.nm2,
                                                                  self.nm3])))
        self.res[0, 12] = 0
        self.res[0, 13] = self.res[0, 0:6].sum()
        self.res[0, 14:] = 0

        # I assign the state variables to temporary variables. That way I
        # don't have to worry about overwriting the original state variables.

        hiring_rate_female_level_1 = self.bf1
        hiring_rate_female_level_2 = self.bf2
        hiring_rate_female_level_3 = self.bf3
        attrition_rate_female_level_1 = self.df1
        attrition_rate_female_level_2 = self.df2
        attrition_rate_female_level_3 = self.df3
        attrition_rate_male_level_1 = self.dm1
        attrition_rate_male_level_2 = self.dm2
        attrition_rate_male_level_3 = self.dm3
        probability_of_outside_hire_level_3 = self.phire3
        probability_of_outside_hire_level_2 = self.phire2
        prob_no_prom_2_3 = 0.0
        prob_no_prom_1_2 = 0.0
        female_promotion_probability_1_2 = self.female_promotion_probability_1
        female_promotion_probability_2_3 = self.female_promotion_probability_2
        department_size_upper_bound = self.upperbound
        department_size_lower_bound = self.lowerbound
        variation_range = self.variation_range
        unfilled_vacanies = 0
        change_to_level_1 = 0
        change_to_level_2 = 0
        change_to_level_3 = 0

        for i in range(1, self.duration):
            # initialize variables for this iteration

            prev_number_of_females_level_1 = self.res[i - 1, 0]
            prev_number_of_females_level_2 = self.res[i - 1, 1]
            prev_number_of_females_level_3 = self.res[i - 1, 2]
            prev_number_of_males_level_1 = self.res[i - 1, 3]
            prev_number_of_males_level_2 = self.res[i - 1, 4]
            prev_number_of_males_level_3 = self.res[i - 1, 5]
            prev_number_of_vacancies_level_3 = self.res[i - 1, 6]
            prev_number_of_vacancies_level_2 = self.res[i - 1, 7]
            prev_number_of_vacancies_level_1 = self.res[i - 1, 8]
            department_size = self.res[i - 1, 0:6].sum()

            # Process Model

            # Determine department size variation for this timestep


            # first both female and males leave the department according to binomial probability.

            female_attrition_level_3 = binomial(prev_number_of_females_level_3,
                                                attrition_rate_female_level_3)

            male_attrition_level_3 = binomial(prev_number_of_males_level_3,
                                              attrition_rate_male_level_3)

            # the departures create a set of vacancies. These vacancies are
            # the basis for promotions from level 2. The possible number of
            # promotions from level 2 depend on the number of women and men
            # at level 2 versus the total number of vacancies generated by
            # attrition at level 3.

            total_vacancies_3 = female_attrition_level_3 + \
                                male_attrition_level_3 + change_to_level_3

            promotions_of_females_level_2_3 = binomial(max(0, min(
                total_vacancies_3, prev_number_of_females_level_2)),
                                                       female_promotion_probability_2_3)

            promotions_of_males_level_2_3 = binomial(max(0,
                                                         min(total_vacancies_3 - \
                                                             promotions_of_females_level_2_3,
                                                             prev_number_of_males_level_2)),
                                                     1 - female_promotion_probability_2_3)

            # After promotions, then the remaining vacancies are settled by
            # hiring.

            vacancies_remaining_after_promotion_3 = max(0, total_vacancies_3 - \
                                                        promotions_of_females_level_2_3 - \
                                                        promotions_of_males_level_2_3)

            hiring_female_3 = binomial(vacancies_remaining_after_promotion_3,
                                       probability_of_outside_hire_level_3 * hiring_rate_female_level_3)

            hiring_male_3 = binomial(
                max(0, vacancies_remaining_after_promotion_3 - \
                    hiring_female_3),
                probability_of_outside_hire_level_3 * (
                    1 - hiring_rate_female_level_3))

            # attrition at level 2 - either people leave from attrition or
            # promotion. I have to make sure that when I account of
            # vacancies, I have to add the promotions to level 3 as well as
            # attrition. I also have to factor in promotions, because the
            # people who are promoted are not candidates for attrition.

            female_attrition_level_2 = binomial(max(0,
                                                    prev_number_of_females_level_2 -
                                                    promotions_of_females_level_2_3),
                                                attrition_rate_female_level_2)

            male_attrition_level_2 = binomial(max(0,
                                                  prev_number_of_males_level_2 -
                                                  promotions_of_males_level_2_3),
                                              attrition_rate_male_level_2)

            # The total number of promotions plus attrition for both men and
            # women create the number of vacancies at level 2.

            total_vacancies_2 = sum(list([female_attrition_level_2,
                                          male_attrition_level_2,
                                          promotions_of_females_level_2_3,
                                          promotions_of_males_level_2_3,
                                          change_to_level_2]))

            promotions_of_females_level_1_2 = binomial(max(0, min(
                total_vacancies_2,
                prev_number_of_females_level_1)),
                                                       female_promotion_probability_1_2)

            promotions_of_males_level_1_2 = binomial(max(0,
                                                         min(total_vacancies_2 - \
                                                             promotions_of_females_level_1_2,
                                                             prev_number_of_males_level_1)),
                                                     1 - female_promotion_probability_1_2)

            vacancies_remaining_after_promotion_2 = max(0, total_vacancies_2 - \
                                                        promotions_of_females_level_1_2 - \
                                                        promotions_of_males_level_1_2)

            hiring_female_2 = binomial(vacancies_remaining_after_promotion_2,
                                       probability_of_outside_hire_level_2 * hiring_rate_female_level_2)
            hiring_male_2 = binomial(max(0,
                                         vacancies_remaining_after_promotion_2 - \
                                         hiring_female_2),
                                     probability_of_outside_hire_level_2 * (
                                         1 - hiring_rate_female_level_2))

            ## Level 1

            female_attrition_level_1 = binomial(max(0,
                                                    prev_number_of_females_level_1 - \
                                                    promotions_of_females_level_1_2),
                                                attrition_rate_female_level_1)

            male_attrition_level_1 = binomial(max(0,
                                                  prev_number_of_males_level_1 - \
                                                  promotions_of_males_level_1_2),
                                              attrition_rate_male_level_1)

            total_vacancies_1 = sum(list([female_attrition_level_1,
                                          male_attrition_level_1,
                                          promotions_of_females_level_1_2,
                                          promotions_of_males_level_1_2,
                                          change_to_level_1]))

            # Recall that there is no promotion at level 1. So we only have
            # hiring.

            hiring_female_1 = binomial(max(0, total_vacancies_1),
                                       hiring_rate_female_level_1)

            hiring_male_1 = binomial(max(0, total_vacancies_1 -
                                         hiring_female_1),
                                     1 - hiring_rate_female_level_1)

            # Write state variables to array and move to next iteration

            self.res[i, 0] = number_of_females_level_1 = sum(
                list([prev_number_of_females_level_1,
                      neg(female_attrition_level_1),
                      neg(promotions_of_females_level_1_2),
                      hiring_female_1]))

            self.res[i, 1] = number_of_females_level_2 = max(0, sum(
                list([prev_number_of_females_level_2,
                      neg(female_attrition_level_2),
                      neg(promotions_of_females_level_2_3),
                      promotions_of_females_level_1_2,
                      hiring_female_2])))

            self.res[i, 2] = number_of_females_level_3 = sum(list([
                prev_number_of_females_level_3,
                neg(female_attrition_level_3),
                promotions_of_females_level_2_3,
                hiring_female_3]))

            self.res[i, 3] = number_of_males_level_1 = sum(list([
                prev_number_of_males_level_1,
                neg(male_attrition_level_1),
                neg(promotions_of_males_level_1_2),
                hiring_male_1]))

            self.res[i, 4] = number_of_males_level_2 = sum(
                list([prev_number_of_males_level_2,
                      neg(male_attrition_level_2),
                      neg(promotions_of_males_level_2_3),
                      promotions_of_males_level_1_2,
                      hiring_male_2]))

            self.res[i, 5] = number_of_males_level_3 = sum(
                list([prev_number_of_males_level_3,
                      neg(male_attrition_level_3),
                      promotions_of_males_level_2_3,
                      hiring_male_3]))

            self.res[i, 6] = sum(list([
                male_attrition_level_3,
                female_attrition_level_3]))

            self.res[i, 7] = sum(list([
                male_attrition_level_2,
                female_attrition_level_2,
                promotions_of_females_level_2_3,
                promotions_of_males_level_2_3]))

            self.res[i, 8] = sum(list([
                male_attrition_level_1,
                female_attrition_level_1,
                promotions_of_males_level_1_2,
                promotions_of_females_level_1_2]))

            self.res[i, 9] = self.female_promotion_probability_1
            self.res[i, 10] = self.female_promotion_probability_2

            self.res[i, 11] = np.float32(
                truediv(sum(list([number_of_females_level_1,
                                  number_of_females_level_2,
                                  number_of_females_level_3])), sum(list([
                    number_of_females_level_1,
                    number_of_females_level_2,
                    number_of_females_level_3,
                    number_of_males_level_1,
                    number_of_males_level_2,
                    number_of_males_level_3]))))

            unfilled_vacanies = abs(department_size - self.res[i, 0:6].sum())
            self.res[i, 12] = unfilled_vacanies
            department_size = self.res[i, 0:6].sum()
            self.res[i, 13] = department_size
            self.res[i, 14] = hiring_female_3
            self.res[i, 15] = hiring_male_3
            self.res[i, 16] = hiring_female_2
            self.res[i, 17] = hiring_male_2
            self.res[i, 18] = hiring_female_1
            self.res[i, 19] = hiring_male_1
            self.res[i, 20] = 0
            self.res[i, 21] = 0
            self.res[i, 22] = promotions_of_females_level_2_3
            self.res[i, 23] = promotions_of_males_level_2_3
            self.res[i, 24] = promotions_of_females_level_1_2
            self.res[i, 25] = promotions_of_males_level_1_2
            self.res[i, 26] = hiring_rate_female_level_1
            self.res[i, 27] = hiring_rate_female_level_2
            self.res[i, 28] = hiring_rate_female_level_3
            self.res[i, 29] = 1 - hiring_rate_female_level_1
            self.res[i, 30] = 1 - hiring_rate_female_level_2
            self.res[i, 31] = 1 - hiring_rate_female_level_3
            self.res[i, 32] = attrition_rate_female_level_1
            self.res[i, 33] = attrition_rate_female_level_2
            self.res[i, 34] = attrition_rate_female_level_3
            self.res[i, 35] = attrition_rate_male_level_1
            self.res[i, 36] = attrition_rate_male_level_2
            self.res[i, 37] = attrition_rate_male_level_3
            self.res[i, 38] = 1
            self.res[i, 39] = probability_of_outside_hire_level_2
            self.res[i, 40] = probability_of_outside_hire_level_3
            self.res[i, 41] = female_promotion_probability_1_2
            self.res[i, 42] = female_promotion_probability_2_3
            self.res[i, 43] = 1 - female_promotion_probability_1_2
            self.res[i, 44] = 1 - female_promotion_probability_2_3
            self.res[i, 45] = department_size_upper_bound
            self.res[i, 46] = department_size_lower_bound
            self.res[i, 47] = variation_range
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


                levels = np.random.choice([1, 2, 3], variation_range)  #
                # random level
                # choice

                # need to test whether the candidate changes keep the
                # department size within bounds.
                # print(["old dept size:", department_size,
                #        "new dept size:", self.res[i, 0:6].sum(),
                #        "candidate:", department_size +
                #        changes.sum(),
                #        " added postions: ", changes.sum(),
                #        "unfilled ", unfilled_vacanies])
                if (department_size + changes.sum() <=
                        department_size_upper_bound and department_size +
                    changes.sum() >= department_size_lower_bound):
                    change_to_level_3 = np.int(changes[np.where(levels ==
                                                                3)[0]].sum())
                    change_to_level_2 = np.int(changes[np.where(levels ==
                                                                2)[0]].sum())
                    change_to_level_1 = np.int(changes[np.where(levels ==
                                                                1)[0]].sum())
                    flag = True

                if (department_size > department_size_upper_bound):
                    change_to_level_3 = 0
                    change_to_level_2 = 0
                    change_to_level_1 = 0

                    flag = True

                if department_size < department_size_lower_bound:
                    changes = np.ones(variation_range)
                    change_to_level_3 = np.int(changes[np.where(levels ==
                                                                3)[
                        0]].sum())
                    change_to_level_2 = np.int(changes[np.where(levels ==
                                                                2)[
                        0]].sum())
                    change_to_level_1 = np.int(changes[np.where(levels ==
                                                                1)[
                        0]].sum())
                    flag = True


        df_ = pd.DataFrame(self.res)
        df_.columns = MODEL_RUN_COLUMNS + EXPORT_COLUMNS_FOR_CSV

        # print(df_)
        recarray_results = df_.to_records(index=True)
        self.run = recarray_results
        return recarray_results
