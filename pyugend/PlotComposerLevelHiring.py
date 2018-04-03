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
from .PlotSettingsLevelHiring import PlotSettingsLevelHiring
from .BuilderGenericLevelPlot import BuilderGenericLevelPlot

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

height = 600
width = 600


class PlotComposerLevelHiring(abcComparisonPlot):


    def helper_overall_data(self):
        pass

    def helper_level_data(self):
        for lev in range(1,NUMBEROFLEVELS + 1):
            self.coordinates['yval_f' + str(lev)] = [m.results_matrix['mean_f_hire_' + str(lev)] for m in self.comparison]
            self.coordinates['yval_m' + str(lev)] = [m.results_matrix['mean_m_hire_' + str(lev)] for m in self.comparison]

    def helper_overall_empirical_upper_bound(self):
        for lev in range(1, NUMBEROFLEVELS + 1):
            self.coordinates['empirical_upperbound_f' + str(lev)] = [m.results_matrix['f_hire_' + str(lev) + '_975'] for m in self.comparison]
            self.coordinates['empirical_upperbound_m' + str(lev)] = [m.results_matrix['m_hire_' + str(lev) + '_975'] for m in self.comparison]

    def helper_overall_empirical_lower_bound(self):
        for lev in range(1, NUMBEROFLEVELS + 1):
            self.coordinates['empirical_lowerbound_f' + str(lev)] = [m.results_matrix['f_hire_' + str(lev) + '_025'] for m in
                                                        self.comparison]
            self.coordinates['empirical_lowerbound_m' + str(lev)] = [m.results_matrix['m_hire_' + str(lev) + '_025'] for m in
                                                        self.comparison]

    def helper_ground_truth_mgmt(self):
        for lev in range(1, NUMBEROFLEVELS + 1):
            self.coordinates['truth_f' + str(lev)] = self.helper_original_data_mgmt('hiring_f' + str(lev))
            self.coordinates['truth_m' + str(lev)] = self.helper_original_data_mgmt('hiring_m' + str(lev))

    def helper_build_overall_plot_coordinates(self):
        self.helper_indicate_number_of_models()
        self.helper_level_data()
        self.helper_year_duration()
        self.helper_overall_empirical_upper_bound()
        self.helper_overall_empirical_lower_bound()
        self.helper_ground_truth_mgmt()
        self.helper_build_settings()

    def helper_build_settings(self):
        self.settings = {**PlotSettingsLevelHiring.get_settings(), **self.settings}

    def execute_plot(self):
        self.helper_build_overall_plot_coordinates()
        builder = BuilderGenericLevelPlot(self.coordinates,
                                      self.settings)
        director = PlotDirector()
        return director.construct(builder)


# test cases

@pytest.mark.usefixtures('mgmt_data', 'mock_data', 'one_model', 'multi_model')
class TestClass(object):

    def test_plot_level_hiring(self, one_model):
        plot_settings = { 'intervals': 'empirical',
                         'number_of_runs': 100,
                         'target': 0.25,
                         'model_legend_label': ['model 1',
                                              'model 2',
                                              'model 3'],
                         'height_': height,
                         'width_': width,
                         'year_offset': 0
                         }
        show(one_model.plot_hiring_bylevel(plot_settings))

    def test_plot_level_hiring_multiple_models(self, multi_model):
        pass
