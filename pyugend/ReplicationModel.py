"""
Replication Module
------------------
This module simulates the original mathematical model from the paper [FILL
IN]. This model is deterministic in that it does not assign individuals to
promotion and hiring based upon a stochastic process.


"""



__author__ = 'krishnab'

from operator import neg, truediv
import numpy as np
import pandas as pd
from numpy.random import binomial
from .Models import Base_model


class Replication_model(Base_model):

    def __init__(self, **kwds):
        Base_model.__init__(self, **kwds)
        self.name = "Replication Model(mathmod_orig)"
        self.label = "Replication m"

    def run_model(self):

        self.res = np.zeros([self.duration,12], dtype=np.float32)


        self.res[0,0] = self.nf1
        self.res[0,1] = self.nf2
        self.res[0,2] = self.nf3
        self.res[0,3] = self.nm1
        self.res[0,4] = self.nm2
        self.res[0,5] = self.nm3
        self.res[0,6] = self.vac3
        self.res[0,7] = self.vac2
        self.res[0,8] = self.vac1
        self.res[0,9] = np.float32(self.nf1/(self.nf1 + self.nm1))
        self.res[0,10] = np.float32(self.nf2/(self.nf2 + self.nm2))
        self.res[0, 11] = np.float32(
                sum(list([self.nf1, self.nf2, self.nf3])) / sum(list([self.nf1,
                                                                      self.nf2,
                                                                      self.nf3,
                                                                      self.nm1,
                                                                      self.nm2,
                                                                      self.nm3])))

        for i in range(1,self.duration):

            #set promotion probabilities


            # level 3
            female_leave_3 = self.res[i-1,2]*self.df3
            male_leave_3 = self.res[i-1, 5]*self.dm3
            self.res[i,6] = total_vacancies = female_leave_3 + male_leave_3

            hiring_female3 = total_vacancies*self.phire3*self.bf3
            hiring_male3 = total_vacancies*self.phire3*(1-self.bf3)
            hiring_total3 = total_vacancies*self.phire3
            left_to_prom3 = total_vacancies - hiring_total3
            self.res[i,2] = self.res[i-1,2] - female_leave_3 + hiring_female3 + \
                       left_to_prom3*np.float32(self.res[i-1,10])
            self.res[i,5] = self.res[i-1,5] - male_leave_3 + hiring_male3 + left_to_prom3*(
                1-np.float32(self.res[i-1,10]))

            #level 2

            f_go_2 = self.res[i-1,1]*self.df2 + left_to_prom3*np.float32(self.res[i-1, 10])
            m_go_2 = self.res[i-1,4]*self.dm2 + left_to_prom3*np.float32((1 - self.res[i-1,
                                                                            10]))

            self.res[i,7] = f_go_2 + m_go_2

            hiring_female2 = self.res[i,7]*self.phire2*self.bf2
            hiring_male2 = self.res[i,7]*self.phire2*(1-self.bf2)
            hiring_total2 = self.res[i,7]*self.phire2
            left_to_prom2 = self.res[i,7] - hiring_total2

            # Update the male/female level 2 numbers for the current iteration.
            self.res[i,1] = self.res[i-1,1] - f_go_2 + hiring_female2 + \
                       left_to_prom2*np.float32(self.res[
                i-1, 9])
            self.res[i,4] = self.res[i-1,4] - m_go_2 + hiring_male2 + \
                       left_to_prom2*np.float32((1-self.res[
                i-1,9]))

            ## update the promotion probability for level 2 -> 3 with new data
            self.res[i,10] = np.float32(self.res[i,1]/(self.res[i,1] + self.res[i,4]))

            ## Level 1

            f_go_1 = self.res[i-1, 0]*self.df1 + left_to_prom2*np.float32(self.res[i-1,9])
            m_go_1 = self.res[i-1, 3]*self.dm1 + left_to_prom2*np.float32((1 - self.res[i-1,
                                                                             9]))
            self.res[i,8] = f_go_1 + m_go_1
            self.res[i,0] = self.res[i-1,0] - f_go_1 + self.res[i,8]*self.bf1
            self.res[i,3] = self.res[i-1,3] - m_go_1 + self.res[i,8]*(1-self.bf1)

            self.res[i,9] = np.float32(self.res[i,0]/(self.res[i,0] + self.res[i,3]))

            # Update overall department gender balance.
            self.res[i, 11] = np.sum(self.res[i, 0:3]) / np.sum(self.res[i, 0:6])
    ## Print Data matrix

        df_ = pd.DataFrame(self.res)
        df_.columns = ['f1',
                       'f2',
                       'f3',
                       'm1',
                       'm2',
                       'm3',
                       't3',
                       't2',
                       't1',
                       'prom1',
                       'prom2',
                       'gendprop']
        # print(df_)
        recarray_results = df_.to_records(index=True)
        self.run = recarray_results
        return recarray_results


