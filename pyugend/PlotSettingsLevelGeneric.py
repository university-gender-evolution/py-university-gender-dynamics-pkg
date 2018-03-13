

#!/usr/bin/python

"""

Plot Settings class

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

from .abcPlotSettings import abcPlotSettings

class PlotSettingsLevelGeneric(abcPlotSettings):

    @staticmethod
    def get_settings():

        defaults_overall = {'line_width': 2,
        'width_' : 800,
        'height_' : 800,
        'transparency' : 0.25,
        'target_plot ': False,
        'legend_location ': 'top_right',
        'color_target ': '#ca0020',
        'percent_line_plot': False,
        'percent_line_value' : 0.5,
        'color_percent_line' : '#ca0020',
        'target_plot_linewidth' : 2,
        'percent_linewidth' : 2,
        'model_legend_label' : ['Model 3'],
        'target_plot_legend_label' : 'target',
        'percent_legend_label' : 'percent',
        'male_female_numbers_plot' : False,
        'mf_male_color' : ['blue', 'purple'],
        'mf_female_color': ['green', 'brown'],
        'mf_target_color' : ['red', 'orange', 'deeppink', 'darkred'],
        'mf_male_label' : ['Male Model 3 Spec1', 'Male Model 3 Spec2', 'Male Model 3 Spec3' ],
        'mf_female_label': ['Female Model 3 Spec1', 'Female Model 3 Spec2', 'Female Model 3 Spec 3'],
        'mf_target_label' : ['Target 1', 'Target 2'],
        'mf_male_linewidth' : 2,
        'mf_target_linewidth' : 2,
        'mf_data_color' : ['blue'],
        'mf_female_data_label': ['Female Data Spec1', 'Female Data Spec2'],
        'mf_male_data_label': ['Male Data Spec1', 'Male Data Spec2'],
        'data_plot' : True,
        'data_line_legend_label' : 'Management Data',
        'year_offset' : 0,
        'data_line_style': 'dashdot'}

        return defaults_overall

    plot_settings = {'plottype': 'gender number',
                     'intervals': 'empirical',
                     'number_of_runs': 100,
                     'target': 0.25,
                     'line_width': 2,
                     'model_legend_label': ['Model 3 No Growth',
                                            'Model 3 Lin Growth',
                                            'Model 3 Forecast'],
                     'transparency': 0.25,
                     'legend_location': 'top right',
                     'height_': 400,
                     'width_': 400,
                     # main plot axis labels
                     'xlabels': ['Years', 'Years', 'Years', 'Years',
                                 'Years', 'Years'],
                     'ylabels': ['Number of Women', 'Number of Women',
                                 'Number of Women', 'Number of Men',
                                 'Number of Men', 'Number of Men'],
                     'titles': ['f1', 'f2', 'f3', 'm1', 'm2', 'm3'],

                     # target plot settings
                     'target_plot': True,
                     'target_color': 'red',
                     'target_plot_linewidth': 2,
                     'target_number_labels': ['Target Model 3 NG',
                                              'Target Model 3 LG',
                                              'Target Model 3 FG'],

                     # percent plot settings
                     'percent_line_plot': False,
                     'percent_line_value': 0.5,
                     'color_percent_line': 'red',
                     'percent_linewidth': 2,
                     'percent_legend_label': 'Reference Line'
                     }
