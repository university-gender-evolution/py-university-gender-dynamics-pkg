

#!/usr/bin/python

"""

Builder Class for generic level plot

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


from .abcLevelPlotBuilder import abcLevelPlotBuilder
import numpy as np
NUMBEROFLEVELS = 3

class BuilderGenericLevelPlot(abcLevelPlotBuilder):



    def draw_lines(self):

        for lev in range(1, NUMBEROFLEVELS + 1):
            for k in range(1,self.coordinates['number_of_models']+1):
                self.plots['plotf_' + str(lev)].line(self.coordinates['xval'],
                               self.coordinates['yval_f' + str(lev)],
                               line_width = self.settings['line_width'],
                               legend = self.settings['model_legend_labels']['model'+str(k)],
                               line_color = self.settings['linecolor']['model'+str(k)])
                self.plots['plotm_' + str(lev)].line(self.coordinates['xval'],
                               self.coordinates['yval_m' + str(lev)],
                               line_width = self.settings['line_width'],
                               legend = self.settings['model_legend_labels']['model'+str(k)],
                               line_color = self.settings['linecolor']['model'+str(k)])
                self.plots['plotf_' + str(lev)].circle(self.coordinates['xval'],
                               self.coordinates['yval_f' + str(lev)],
                               size = 3)
                self.plots['plotm_' + str(lev)].circle(self.coordinates['xval'],
                               self.coordinates['yval_m' + str(lev)],
                               size = 3)


    def draw_error_intervals(self):

        for lev in range(1, NUMBEROFLEVELS + 1):
            for k in range(self.coordinates['number_of_models']):
                x_data = np.asarray(self.coordinates['xval'])
                band_x = np.append(x_data, x_data[::-1])
                f_band_y = np.append(self.coordinates['empirical_lowerbound_f' + str(lev)][k],
                                 self.coordinates['empirical_upperbound_f' + str(lev)][k][::-1])
                m_band_y = np.append(self.coordinates['empirical_lowerbound_m' + str(lev)][k],
                                 self.coordinates['empirical_upperbound_m' + str(lev)][k][::-1])

                self.plots['plotf_' + str(lev)].patch(band_x,
                        f_band_y,
                        color= self.settings['linecolor']['model'+str(k+1)],
                        fill_alpha= self.settings['transparency'])
                self.plots['plotm_' + str(lev)].patch(band_x,
                        m_band_y,
                        color= self.settings['linecolor']['model'+str(k+1)],
                        fill_alpha= self.settings['transparency'])

    def draw_data_lines(self):
        for lev in range(1, NUMBEROFLEVELS + 1):
            self.plots['plotf_' + str(lev)].line(self.coordinates['xval'],
                         self.coordinates['truth_f' + str(lev)],
                         line_color=self.settings['data_line_color']['mgmt'],
                         legend=self.settings['data_line_legend_label'],
                         line_width=self.settings['line_width'],
                         line_dash=self.settings['data_line_style'])

            self.plots['plotm_' + str(lev)].line(self.coordinates['xval'],
                         self.coordinates['truth_m' + str(lev)],
                         line_color=self.settings['data_line_color']['mgmt'],
                         legend=self.settings['data_line_legend_label'],
                         line_width=self.settings['line_width'],
                         line_dash=self.settings['data_line_style'])

    def draw_target(self):
        pass



    def draw_misc(self):
        pass


if __name__ == "__main__":
    print('This is an abstract base class for building plots')
