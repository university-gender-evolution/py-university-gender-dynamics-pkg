

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
from bokeh.plotting import figure, output_file, show
from .abcComparisonPlot import abcComparisonPlot
from pyugend.ModelGenderDiversityGrowthForecast import ModelGenderDiversityGrowthForecast
from pyugend.ModelGenderDiversityLinearGrowth import ModelGenderDiversityLinearGrowth
from pyugend.ModelGenderDiversityGrowthForecastIncrementalChange import ModelGenderDiversityGrowthForecastIncremental
from pyugend.Comparison import Comparison
from .BuilderOverallAttritionPlot import BuilderOverallAttritionPlot
from .PlotDirector import PlotDirector


# Set constants
# These are the fields involves in this plot.
FIELDS = ['vacancies_3',
         'vacancies_2',
         'vacancies_1']

EMPFIELDS_UPPER: ['vac_3_975',
                'vac_2_975',
                'vac_1_975',]

EMPFIELDS_LOWER: ['vac_3_025',
                'vac_2_025',
                'vac_1_025',]

SDFIELDS: ['std_vac_3',
          'std_vac_2',
          'std_vac_1',]

GROUNDTRUTH = 'total_attrition'
height = 800
width = 800


class ComparisonPlotAttrition(abcComparisonPlot):

    def helper_overall_data(self):

        fields = FIELDS # so that no mutation happens
        yval = [m.results_matrix[[fields]].sum(axis=1)
                          for m in self.comparison.mlist]
        self.coordinates['yval'] = yval

    def helper_overall_empirical_upper_bound(self):
        fields = EMPFIELDS_UPPER # so that no mutation happens
        empirical_upper_bound = [m.results_matrix[[fields]].sum(axis=1)
                          for m in self.comparison.mlist]
        self.coordinates['empirical_upper_bound'] = empirical_upper_bound

    def helper_overall_empirical_lower_bound(self):
        fields = EMPFIELDS_LOWER # so that no mutation happens
        empirical_lower_bound = [m.results_matrix[[fields]].sum(axis=1)
                          for m in self.comparison.mlist]
        self.coordinates['empirical_lower_bound'] = empirical_lower_bound

    def helper_ground_truth_mgmt(self):
        self.coordinates['ground_truth'] = self.helper_original_data(GROUNDTRUTH)

    def help_build_overall_plot_coordinates(self):

        # This will build the data dictionary for the plot
        self.helper_overall_data()
        self.helper_overall_empirical_upper_bound()
        self.helper_overall_empirical_lower_bound()
        self.helper_ground_truth_mgmt()

    def helper_level_data(self):
        pass


    def execute_plot(self):
        builder = BuilderAttritionPlotOverall(self,
                                          self.coordinates,
                                          self.settings)
        director = PlotDirector()
        return director(builder)


# test cases


# content of test_class.py
@pytest.mark.usefixtures('mgmt_data', 'mock_data')
class TestClass(object):
    def test_bokeh_comparison_plot_overall_one_model(self, mgmt_data):
        modlist = list([ModelGenderDiversityGrowthForecastIncremental(**mgmt_data)])
        # modlist = list([Model2GenderDiversity(**mgmt_data),
        #                 Mod_Stoch_FBPH(**mgmt_data)])
        modlist[0].init_default_hiring_rate()
        modlist[0].init_growth_rate([0.02, 0.01, 0.10, 0.05])
        c = Comparison(modlist)

        # print(modlist[0].calculate_yearly_dept_size_targets())

        plot_settings = {'plottype': 'gender proportion',
                         'intervals': 'empirical',
                         'number_of_runs': 100,
                         # number simulations to average over
                         'target': 0.25,
                         # target percentage of women in the department
                         # Main plot settings
                         'xlabel': 'Years',
                         'ylabel': 'Proportion Women',
                         'title': 'Change in Proportion Women',
                         'line_width': 2,
                         'transparency': 0.25,
                         'model_legend_label': ['New Model',
                                                'Mode 2, Promote-Hire'],
                         'legend_location': 'top_right',
                         'height_': height,
                         'width_': width,
                         'year_offset': 0
                         }
        show(c.plot_comparison_overall_chart(**plot_settings))
