#!/usr/bin/python

"""

Abstract Base Class for building plots

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



import abc
from bokeh.plotting import figure, output_file, show
import pytest

height = 800
width = 800





class abcComparisonPlot(metaclass=abc.ABCMeta):


    def __init__(self, comparison, settings=None):
        self.plot = None
        self.settings = settings
        self.comparison = comparison
        self.coordinates = {}
    @abc.abstractmethod
    def helper_overall_data(self):
        pass

    @abc.abstractmethod
    def helper_level_data(self):
        pass

    @abc.abstractmethod
    def execute_plot(self):
        pass

    def helper_duration(self):
        xval = list(range(min([m.duration for m in self.comparison.mlist])))[self.settings['year_offset']:]
        return xval

    def helper_original_data_mgmt(self, field):
        dval = self.comparison.mlist[0].mgmt_data.get_field(field)
        return dval
