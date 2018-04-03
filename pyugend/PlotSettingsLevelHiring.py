

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

class PlotSettingsLevelHiring(abcPlotSettings):

    @staticmethod
    def get_settings():

        defaults_overall = {'line_width': 2,
        'width_' : 400,
        'height_' : 400,
        'transparency' : 0.25,
        'target_plot ': False,
        'legend_location ': 'top_right',
        'linecolor': {'model1':'blue',
                     'model2':'green',
                     'model3':'purple',
                     'model4':'brown'},
        'xlabel': 'Years',
        'ylabels': {'ylabel_f1':'Number of Women',
                    'ylabel_f2':'Number of Women',
                    'ylabel_f3':'Number of Women',
                    'ylabel_m1':'Number of Men',
                    'ylabel_m2':'Number of Men',
                    'ylabel_m3':'Number of Men'},
        'titles': {'title_f1':'Hiring Women Asst. Professors',
                   'title_f2':'Hiring Women Assoc. Professors',
                   'title_f3':'Hiring Women Full Professors',
                   'title_m1':'Hiring Male Asst. Professors',
                   'title_m2':'Hiring Male Assoc. Professors',
                   'title_m3':'Hiring Male Full Professors'},
        'target_number_labels': ['Fill value',
                              'Fill value',
                              'Fill value'],
        'color_target ': '#ca0020',
        'percent_legend_label': 'Reference Line',
        'percent_line_plot': False,
        'percent_line_value' : 0.5,
        'color_percent_line' : '#ca0020',
        'target_plot_linewidth' : 2,
        'percent_linewidth' : 2,
        'model_legend_labels' : {'model1':'Model 3',
                              'model2':'Model 3',
                              'model3':'Model 3',
                              'model4':'Model 3'},
        'target_plot_legend_label' : 'target',
        'percent_legend_label' : 'percent',
        'male_female_numbers_plot' : False,
        'data_plot' : True,
        'data_line_legend_label' : 'Management Data',
        'year_offset' : 0,
        'data_line_color': {'mgmt':'blue',
                          'science': 'purple'},
        'data_line_style': 'dashdot'}

        return defaults_overall

