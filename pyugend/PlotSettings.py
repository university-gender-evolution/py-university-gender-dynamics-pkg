

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



class PlotSettings():

    @staticmethod
    def get_overall_plot_settings():

        defaults_overall = {'line_width' : 2,
        'width_' : 800,
        'height_' : 800,
        'transparency' : 0.25,
        'linecolor' : ['blue', 'green', 'purple', 'brown'],
        'target_plot ': False,
        'legend_location ': 'top_right',
        'color_target ': '#ca0020',
        'percent_line_plot': False,
        'percent_line_value' : 0.5,
        'color_percent_line' : '#ca0020',
        'target_plot_linewidth' : 2,
        'percent_linewidth' : 2,
        'model_legend_label' : ['model'],
        'target_plot_legend_label' : 'target',
        'percent_legend_label' : 'percent',
        'male_female_numbers_plot' : False,
        'mf_male_color' : ['#0004ff', '#2c7bb6'],
        'mf_target_color' : ['red', 'orange', 'deeppink', 'darkred'],
        'mf_male_label' : ['Male model 1', 'Male model 2'],
        'mf_target_label' : ['Target 1', 'Target 2'],
        'mf_male_linewidth' : 2,
        'mf_target_linewidth' : 2,
        'mf_data_color' : ['blue'],
        'mf_data_label' : ['Female Faculty Data', 'Male Faculty Data'],
        'mf_data_linewidth' : [2],
        'parameter_sweep_param' : None,
        'parameter_ubound' : 0,
        'parameter_lbound' : 0,
        'number_of_steps' : 0,
        'vertical_line_label' : 'Original Value',
        'vertical_line_width' : 2,
        'vertical_line_color' : ['black'],
        'data_plot' : True,
        'data_line_color' : ['blue'],
        'data_line_legend_label' : 'Management Data',
        'year_offset' : 0,
        'data_line_style': 'dashdot'}

        return defaults_overall
