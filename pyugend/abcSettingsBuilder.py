
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

import abc
from .PlotSettingsOverall import PlotSettingsOverall

class abcPlotSettingsBuilder(metaclass=abc.ABCMeta):

    def __init__(self, settings=None):
        self.plot = PlotSettingsOverall()
        self.settings = settings

    @abc.abstractmethod
    def set_global_settings(self):
        pass

    @abc.abstractmethod
    def set_line_settings(self):
        pass

    @abc.abstractmethod
    def set_interval_settings(self):
        pass

    @abc.abstractmethod
    def set_target_settings(self):
        pass

    @abc.abstractmethod
    def set_percent_settings(self):
        pass

    @abc.abstractmethod
    def set_mf_plot(self):
        pass

    @abc.abstractmethod
    def set_parameter_sweep_settings(self):
        pass


