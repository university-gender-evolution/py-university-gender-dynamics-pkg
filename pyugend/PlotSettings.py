

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

from bokeh.models.annotations import Title


class PlotSettings():

    def __init__(self, settings):

        self.title = None
        self.plottype = None
        self.intervals = None
        self.number_of_runs = None
        self.target = None
        self.xlabel = None
        self.ylabel = None
        self.title = None
        self.line_width = None
        self.width_= None
        self.height_ = None
        self.transparency = None
        self.linecolor  = None
        self.target_plot= None
        self.legend_location=None,
        self.color_target = None
        self.percent_line_plot = None
        self.percent_line_value = None
        self.color_percent_line = None
        self.target_plot_linewidth = None
        self.percent_linewidth= None
        self.model_legend_label= None
        self.target_plot_legend_label = None
        self.percent_legend_label= None
        self.male_female_numbers_plot = None
        self.mf_male_color= None
        self.mf_target_color = None
        self.mf_male_label = None
        self.mf_target_label = None
        self.mf_male_linewidth = None
        self.mf_target_linewidth = None
        self.mf_data_color = None
        self.mf_data_label = None
        self.mf_data_linewidth = None
        self.parameter_sweep_param = None
        self.parameter_ubound = None
        self.parameter_lbound = None
        self.number_of_steps = None
        self.vertical_line_label = None
        self.vertical_line_width = None
        self.vertical_line_color = None
        self.data_plot = None
        self.data_line_color = None
        self.data_line_legend_label = None
