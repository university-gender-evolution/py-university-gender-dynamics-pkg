
#!/usr/bin/python

"""

Abstract Base Class for building level plots

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
__version__ = '0.1.0'
__maintainer__ = 'krishna bhogaonker'
__email__ = 'cyclotomiq@gmail.com'
__status__ = 'pre-alpha'

import abc
from bokeh.plotting import figure
from bokeh.layouts import gridplot

NUMBEROFPLOTS = 3

class abcLevelPlotBuilder(metaclass=abc.ABCMeta):


    def __init__(self, coordinates, settings):
        self.coordinates = coordinates
        self.settings = settings
        self.plots = {}
    def create_plot(self):

        for p in range(1, NUMBEROFPLOTS+1):
            self.plots['plotf_' + str(p)] = figure(title=self.settings['titles']['title_f' + str(p)],
                                  x_axis_label = self.settings['xlabel'],
                                  y_axis_label = self.settings['ylabels']['ylabel_f' + str(p)],
                                  width = self.settings['width_'],
                                  height = self.settings['height_'])

            self.plots['plotm_' + str(p)] = figure(title=self.settings['titles']['title_m' + str(p)],
                                 x_axis_label=self.settings['xlabel'],
                                 y_axis_label=self.settings['ylabels']['ylabel_m' + str(p)],
                                 width=self.settings['width_'],
                                 height=self.settings['height_'])

            self.plots['plotf_' + str(p)].legend.click_policy = 'hide'
            self.plots['plotm_' + str(p)].legend.click_policy = 'hide'

    @abc.abstractmethod
    def draw_lines(self):
        pass

    @abc.abstractmethod
    def draw_data_lines(self):
        pass

    @abc.abstractmethod
    def draw_error_intervals(self):
        pass


    @abc.abstractmethod
    def draw_target(self):
        pass



    @abc.abstractmethod
    def draw_misc(self):
        pass


    def return_plot(self):
        grid = gridplot([[self.plots['plotf_1'],
                        self.plots['plotf_2'],
                        self.plots['plotf_3']],
                         [self.plots['plotm_1'],
                          self.plots['plotm_2'],
                          self.plots['plotm_3']]])
        return grid

if __name__ == "__main__":
    print('This is an abstract base class for building plots')
