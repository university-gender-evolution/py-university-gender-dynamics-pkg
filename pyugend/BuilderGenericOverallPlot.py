

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


from .abcOverallPlotBuilder import abcOverallPlotBuilder
from .abcSettingsBuilder import abcPlotSettingsBuilder
import numpy as np

class BuilderGenericOverallPlot(abcOverallPlotBuilder):



    def draw_lines(self):
        for k in range(self.coordinates['number_of_models']):
            self.plot.line(self.coordinates['xval'],
                           self.coordinates['yval'][k],
                           line_width = self.settings['line_width'],
                           legend = self.settings['model_legend_label'][k],
                           line_color = self.settings['linecolor'][k])
            self.plot.circle(self.coordinates['xval'],
                           self.coordinates['yval'][k],
                           size = 3)

    def draw_error_intervals(self):
        for k in range(self.coordinates['number_of_models']):
            x_data = np.asarray(self.coordinates['xval'])
            band_x = np.append(x_data, x_data[::-1])
            band_y = np.append(self.coordinates['empirical_lower_bound'][k],
                             self.coordinates['empirical_upper_bound'][k][::-1])

            self.plot.patch(band_x,
                    band_y,
                    color= self.settings['linecolor'][k],
                    fill_alpha= self.settings['transparency'])

    def draw_data_lines(self):
        self.plot.line(self.coordinates['xval'],
                     self.coordinates['ground_truth'],
                     line_color=self.settings['data_line_color'][0],
                     legend=self.settings['data_line_legend_label'],
                     line_width=self.settings['line_width'],
                     line_dash=self.settings['data_line_style'])

    def draw_target(self):
        pass



    def draw_misc(self):
        pass



class BuilderSettingsAttritionPlot(abcPlotSettingsBuilder):

    def __init__(self):
        pass



if __name__ == "__main__":
    print('This is an abstract base class for building plots')
