# !/usr/bin/python

"""

Builder Class for Hiring plot

"""

## MIT License
##
## Copyright (c) 2017, krishna bhogaonker
## Permission is hereby granted, free of charge, to any person obtaining a ## copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


__author__ = 'krishna bhogaonker'
__copyright__ = 'copyright '
__credits__ = ['krishna bhogaonker']
__license__ = "MIT"
__version__ = ''
__maintainer__ = 'krishna bhogaonker'
__email__ = 'cyclotomiq@gmail.com'
__status__ = ''

import pytest
from bokeh.plotting import show
from .abcComparisonPlot import abcComparisonPlot
from .BuilderGenericOverallPlot import BuilderGenericOverallPlot
from .PlotDirector import PlotDirector
from .BuilderGenericOverallMFPlot import BuilderGenericOverallMFPlot

# Set constants
# These are the fields involves in this plot.
MFIELDS = ['mean_m_hire_3',
           'mean_m_hire_2',
           'mean_m_hire_1']

FFIELDS = ['mean_f_hire_3',
           'mean_f_hire_2',
           'mean_f_hire_1']

M_EMPFIELDS_UPPER = ['m_hire_3_975',
                     'm_hire_2_975',
                     'm_hire_1_975']

F_EMPFIELDS_UPPER = ['f_hire_3_975',
                     'f_hire_2_975',
                     'f_hire_1_975']

M_EMPFIELDS_LOWER = ['m_hire_3_025',
                     'm_hire_2_025',
                     'm_hire_1_025']

F_EMPFIELDS_LOWER = ['f_hire_3_025',
                     'f_hire_2_025',
                     'f_hire_1_025']

M_SDFIELDS = ['std_m_hire_3',
              'std_m_hire_2',
              'std_m_hire_1']

F_SDFIELDS = ['std_m_hire_3',
              'std_m_hire_2',
              'std_m_hire_1']

M_GROUNDTRUTH = 'total_male_hiring'

F_GROUNDTRUTH = 'total_female_hiring'
height = 800
width = 800


class ComparisonPlotOverallMFHiring(abcComparisonPlot):

    def helper_overall_data(self):
        male_yval = [m.results_matrix[MFIELDS].sum(1) for m in self.comparison]
        self.coordinates['male_yval'] = male_yval

        female_yval = [m.results_matrix[FFIELDS].sum(1) for m in self.comparison]
        self.coordinates['female_yval'] = female_yval

    def helper_overall_empirical_upper_bound(self):
        male_empirical_upper_bound = [m.results_matrix[M_EMPFIELDS_UPPER].sum(axis=1)
                                 for m in self.comparison]
        self.coordinates['male_empirical_upper_bound'] = male_empirical_upper_bound

        female_empirical_upper_bound = [m.results_matrix[F_EMPFIELDS_UPPER].sum(axis=1)
                                 for m in self.comparison]
        self.coordinates['female_empirical_upper_bound'] = female_empirical_upper_bound


    def helper_overall_empirical_lower_bound(self):

        male_empirical_lower_bound = [m.results_matrix[M_EMPFIELDS_LOWER].sum(axis=1)
                                 for m in self.comparison]
        self.coordinates['male_empirical_lower_bound'] = male_empirical_lower_bound

        female_empirical_lower_bound = [m.results_matrix[F_EMPFIELDS_LOWER].sum(axis=1)
                                 for m in self.comparison]
        self.coordinates['female_empirical_lower_bound'] = female_empirical_lower_bound

    def helper_ground_truth_mgmt(self):
        self.coordinates['male_ground_truth'] = self.helper_original_data_mgmt(M_GROUNDTRUTH)
        self.coordinates['female_ground_truth'] = self.helper_original_data_mgmt(F_GROUNDTRUTH)

    def helper_indicate_number_of_models(self):
        self.coordinates['number_of_models'] = len(self.comparison)

    def helper_year_duration(self):
        self.coordinates['xval'] = self.helper_duration()

    def helper_build_overall_plot_coordinates(self):
        # This will build the data dictionary for the plot
        self.helper_indicate_number_of_models()
        self.helper_overall_data()
        self.helper_year_duration()
        self.helper_overall_empirical_upper_bound()
        self.helper_overall_empirical_lower_bound()
        self.helper_ground_truth_mgmt()

    def helper_level_data(self):
        pass

    def execute_plot(self):
        self.helper_build_overall_plot_coordinates()
        builder = BuilderGenericOverallMFPlot(self.coordinates,
                                            self.settings)
        director = PlotDirector()
        return director.construct(builder)


# test cases


# content of test_class.py
@pytest.mark.usefixtures('mgmt_data', 'mock_data', 'one_model', 'multi_model')
class TestClass(object):

    def test_plot_mfhiring(self, one_model):
        plot_settings = {'plottype': 'hiring',
                         'intervals': 'empirical',
                         'number_of_runs': 100,
                         # number simulations to average over
                         'target': 0.25,
                         # target percentage of women in the department
                         # Main plot settings
                         'xlabel': 'Years',
                         'ylabel': 'Number of Attritions',
                         'title': 'MF Hiring Plot',
                         'model_legend_label': ['Model 3 No Growth',
                                                'Model 3 Linear Growth',
                                                'Model 3 Moving Average'],
                         'height_': height,
                         'width_': width,
                         'year_offset': 0
                         }
        show(one_model.plot_hiring_mf_overall(plot_settings))
