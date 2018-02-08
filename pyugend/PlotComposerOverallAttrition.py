

#!/usr/bin/python

"""

Builder Class for attrition plot

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
from .PlotSettingsOverall import PlotSettingsOverall


# Set constants
# These are the fields involves in this plot.
FIELDS = ['mean_vac_3',
         'mean_vac_2',
         'mean_vac_1']

EMPFIELDS_UPPER = ['vac_3_975',
                'vac_2_975',
                'vac_1_975']

EMPFIELDS_LOWER = ['vac_3_025',
                'vac_2_025',
                'vac_1_025']

SDFIELDS = ['std_vac_3',
          'std_vac_2',
          'std_vac_1']

GROUNDTRUTH = 'total_attrition'
height = 800
width = 800


class PlotComposerOverallAttrition(abcComparisonPlot):

    def helper_overall_data(self):
        yval = [m.results_matrix[FIELDS].sum(1) for m in self.comparison]
        self.coordinates['yval'] = yval

    def helper_overall_empirical_upper_bound(self):
        empirical_upper_bound = [m.results_matrix[EMPFIELDS_UPPER].sum(axis=1)
                          for m in self.comparison]
        self.coordinates['empirical_upper_bound'] = empirical_upper_bound

    def helper_overall_empirical_lower_bound(self):
        empirical_lower_bound = [m.results_matrix[EMPFIELDS_LOWER].sum(axis=1)
                          for m in self.comparison]
        self.coordinates['empirical_lower_bound'] = empirical_lower_bound

    def helper_ground_truth_mgmt(self):
        self.coordinates['ground_truth'] = self.helper_original_data_mgmt(GROUNDTRUTH)

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
        self.helper_build_settings()

    def helper_level_data(self):
        pass

    def helper_build_settings(self):
        self.settings = {**PlotSettingsOverall.get_settings(), **self.settings}

    def execute_plot(self):

        self.helper_build_overall_plot_coordinates()
        builder = BuilderGenericOverallPlot(self.coordinates,
                                            self.settings)
        director = PlotDirector()
        return director.construct(builder)


# test cases


# content of test_class.py
@pytest.mark.usefixtures('mgmt_data', 'mock_data', 'one_model', 'multi_model')
class TestClass(object):

    def test_plot_attrition(self, one_model):
        plot_settings = {'plottype': 'attrition',
                         'intervals': 'empirical',
                         'number_of_runs': 100,
                         # number simulations to average over
                         'target': 0.25,
                         # target percentage of women in the department
                         # Main plot settings
                         'xlabel': 'Years',
                         'ylabel': 'Number of Attritions',
                         'title': 'Attrition Plot',
                         'line_width': 2,
                         'transparency': 0.25,
                         'model_legend_label': ['Model 3 No Growth',
                                              'Model 3 Linear Growth',
                                              'Model 3 Moving Average'],
                         'legend_location': 'top_right',
                         'height_': height,
                         'width_': width,
                         'year_offset': 0
                         }
        show(one_model.plot_attrition_overall(plot_settings))
