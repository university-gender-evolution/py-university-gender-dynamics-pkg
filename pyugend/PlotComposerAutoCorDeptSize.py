

#!/usr/bin/python

"""

Plot Composer class for Auto-Correlation plot

"""

## MIT License
##
## Copyright (c) 2017, krishna bhogaonker
## Permission is hereby granted, free of charge, to any person obtaining a ## copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


__author__ = 'krishna bhogaonker'
__copyright__ = 'copyright 2017'
__credits__ = ['krishna bhogaonker']
__license__ = "MIT"
__version__ = '0.1.0'
__maintainer__ = 'krishna bhogaonker'
__email__ = 'cyclotomiq@gmail.com'
__status__ = 'pre-alpha'

import pytest
from bokeh.plotting import show
from .abcComparisonPlot import abcComparisonPlot
from .BuilderGenericAutoCorPlot import BuilderGenericAutoCorrelationPlot
from .PlotDirector import PlotDirector
from .PlotSettingsAutoCorrelation import PlotSettingsAutoCorrelation


# Set constants
# These are the fields involves in this plot.
GROUNDTRUTH = 'change_dept_size_numbers'

height = 800
width = 800


class PlotComposerAutoCorDeptSize(abcComparisonPlot):

    def helper_overall_data(self):
        pass

    def helper_overall_empirical_upper_bound(self):
        pass

    def helper_overall_empirical_lower_bound(self):
        pass

    def helper_ground_truth_mgmt(self):
        self.coordinates['ground_truth'] = self.helper_original_data_mgmt(GROUNDTRUTH)

    def helper_indicate_number_of_models(self):
        pass
    def helper_year_duration(self):
        pass
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
        self.settings = {**PlotSettingsAutoCorrelation.get_settings(), **self.settings}

    def execute_plot(self):

        self.helper_build_overall_plot_coordinates()
        builder = BuilderGenericAutoCorrelationPlot(self.coordinates,
                                            self.settings)
        director = PlotDirector()
        return director.construct(builder)


# test cases


# content of test_class.py
@pytest.mark.usefixtures('mgmt_data', 'mock_data', 'one_model', 'multi_model')
class TestClass(object):

    def test_plot_autocorrelation(self, one_model):
        plot_settings = {'plottype': 'autocorrelation',

                         # target percentage of women in the department
                         # Main plot settings
                         'xlabel': 'Change in Department Size (t)',
                         'ylabel': 'Change in Department Size (t+1)',
                         'title': 'Autocorrelation plot',
                         'model_legend_label': ['Model 3 No Growth',
                                              'Model 3 Linear Growth',
                                              'Model 3 Moving Average'],
                         'legend_location': 'top_right',
                         'height_': height,
                         'width_': width,
                         'year_offset': 0
                         }
        show(one_model.plot_autocor_dept_size(plot_settings))
