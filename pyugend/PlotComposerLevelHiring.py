# !/usr/bin/python

"""

Builder Class for Level Hiring plot

"""

## MIT License
##
## Copyright (c) 2017, krishna bhogaonker
## Permission is hereby granted, free of charge, to any person obtaining a ## copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


__author__ = 'krishna bhogaonker'
__copyright__ = 'copyright 2018'
__credits__ = ['krishna bhogaonker']
__license__ = "MIT"
__version__ = '0.1.0'
__maintainer__ = 'krishna bhogaonker'
__email__ = 'cyclotomiq@gmail.com'
__status__ = 'pre-alpha'

import pytest
from bokeh.plotting import show
from .abcComparisonPlot import abcComparisonPlot
from .PlotDirector import PlotDirector
from .PlotSettingsOverallMF import PlotSettingsOverallMF


# Set constants
# These are the fields involves in this plot.
FIELDS = ['mean_f_hire_1',
         'mean_f_hire_2',
         'mean_f_hire_3',
         'mean_m_hire_1',
         'mean_m_hire_2',
         'mean_m_hire_3']

EMPFIELDS_UPPER = ['f_hire_1_975',
                 'f_hire_2_975',
                 'f_hire_3_975',
                 'm_hire_1_975',
                 'm_hire_2_975',
                 'm_hire_3_975']


EMPFIELDS_LOWER = ['f_hire_1_025',
                 'f_hire_2_025',
                 'f_hire_3_025',
                 'm_hire_1_025',
                 'm_hire_2_025',
                 'm_hire_3_025']

SDFIELDS = ['std_f_hire_1',
           'std_f_hire_2',
           'std_f_hire_3',
           'std_m_hire_1',
           'std_m_hire_2',
           'std_m_hire_3',]

GROUNDTRUTH =  ['hiring_f1',
              'hiring_f2',
              'hiring_f3',
              'hiring_m1',
              'hiring_m2',
              'hiring_m3']

NUMBEROFLEVELS = 3

height = 800
width = 800


class PlotComposerLevelHiring(abcComparisonPlot):


    def helper_overall_data(self):
        pass

    def helper_level_data(self):
        for lev in range(NUMBEROFLEVELS):
            self.coordinates['yval_f' + str(lev)] = [m.results_matrix['mean_f_hire_' + str(lev)] for m in self.comparison]
            self.coordinates['yval_m' + str(lev)] = [m.results_matrix['mean_m_hire_' + str(lev)] for m in self.comparison]

    def helper_overall_empirical_upper_bound(self):
        for lev in range(NUMBEROFLEVELS):
            self.coordinates['emp_ub_f_' + str(lev)] = [m.results_matrix['f_hire_' + str(lev) + '_975'] for m in self.comparison]
            self.coordinates['emp_ub_m_' + str(lev)] = [m.results_matrix['m_hire_' + str(lev) + '_975'] for m in self.comparison]

    def helper_overall_empirical_lower_bound(self):
        for lev in range(NUMBEROFLEVELS):
            self.coordinates['emp_lb_f_' + str(lev)] = [m.results_matrix['f_hire_' + str(lev) + '_025'] for m in
                                                        self.comparison]
            self.coordinates['emp_lb_m_' + str(lev)] = [m.results_matrix['m_hire_' + str(lev) + '_025'] for m in
                                                        self.comparison]

    def helper_ground_truth_mgmt(self):
        for lev in range(NUMBEROFLEVELS):
            self.coordinates['truth_f_' + str(lev)] = self.helper_original_data_mgmt('hiring_f' + str(lev))
            self.coordinates['truth_m_' + str(lev)] = self.helper_original_data_mgmt('hiring_m' + str(lev))

    def helper_build_overall_plot_coordinates(self):
        self.helper_indicate_number_of_models()
        self.helper_overall_data()
        self.helper_year_duration()
        self.helper_overall_empirical_upper_bound()
        self.helper_overall_empirical_lower_bound()
        self.helper_ground_truth_mgmt()
        self.helper_build_settings()

    def helper_build_settings(self):
        self.settings = {**PlotSettingsOverallMF.get_settings(), **self.settings}


    def execute_plot(self):
        pass

