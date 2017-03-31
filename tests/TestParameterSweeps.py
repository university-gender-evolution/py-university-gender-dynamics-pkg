import pytest
from pyugend.pyugend.Mod_Stoch_FBHP import Mod_Stoch_FBHP
from pyugend.pyugend.Mod_Stoch_FBPH import Mod_Stoch_FBPH
from pyugend.pyugend.Mod_Validate_Sweep import Mod_Validate_Sweep
from pyugend.pyugend.Comparison import Comparison
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.charts import defaults
import warnings


defaults.height = 800
defaults.width = 800

@pytest.mark.usefixtures('mgmt_data')
class Test_param_sweeps:

    def test_comparison_model_param_sweep_increment_min(self, mgmt_data):
        """
        This testcase executes a parameter sweep on the hiring rate of women
        level 1, and then ensures that the minimum value for the sweep range
        matches the smallest increment for the parameter values in the
        collected data.
        :param mgmt_data: Use management department data
        :type mgmt_data: dictionary
        :return: Pass/Fail
        :rtype: Boolean
        """
        t = Mod_Stoch_FBPH(**mgmt_data)
        t.run_parameter_sweep(3, 'bf1', 0.1, 0.5, 3)

        assert(t.parameter_sweep_results['increment'].min() == 0.1)

    def test_comparison_model_param_sweep_increment_max(self, mgmt_data):
        """
        This testcase executes a parameter sweep on the hiring rate of women
        level 1, and then ensures that the maximum value for the sweep range
        matches the maximum increment for the parameter values in the
        collected data.
        :param mgmt_data: Use management department data
        :type mgmt_data: dictionary
        :return: Pass/Fail
        :rtype: Boolean
        """

        t = Mod_Stoch_FBPH(**mgmt_data)
        t.run_parameter_sweep(3, 'bf1', 0.1, 0.5, 3)

        assert(t.parameter_sweep_results['increment'].max() == 0.5)


    def test_parameter_sweep_function_overall_catch_warnings(self, mgmt_data):
        """
        This testcase executes a parameter sweep and then checks for any
        warnings generated while computing the plot figure. In the past there
        were issues with column sizes not matching the in the plot functions.

        :param mgmt_data: Use management department data
        :type mgmt_data: dictionary
        :return: Pass/Fail
        :rtype: Boolean
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            modlist = list([Mod_Validate_Sweep(**mgmt_data)])

            c = Comparison(modlist)

            plot_settings = {'plottype': 'parameter sweep percentage',
                             'intervals': 'empirical',
                             'number_of_runs': 3,
                             # number simulations to average over
                             'target': 0.25,
                             'xlabel': 'Hiring Rate for Women',
                             'ylabel': 'Proportion Women',
                             'title': 'Parameter Sweep Validation, uniform noise(0,'
                                      '0.1)',
                             'model_legend_label': ['Model 1, Hire-Promote',
                                                    'Model '
                                                    '2, '
                                                    'Promote-Hire'],
                             'parameter_sweep_param': 'bf1',
                             'parameter_ubound': 0.6,
                             'parameter_lbound': 0.05,
                             'number_of_steps': 3
                             }
            c.plot_comparison_overall_chart(**plot_settings)
        assert(len(w) == 0)


    def test_parameter_sweep_plot_level_warnings(self, mgmt_data):
        """
        This testcase executes a parameter sweep and captures any warnings
        generated during the construction of the bokeh figure object.
        :param mgmt_data: Use management department data
        :type mgmt_data: dictionary
        :return: Pass/Fail
        :rtype: Boolean
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            modlist = list([Mod_Validate_Sweep(**mgmt_data)])

            c = Comparison(modlist)

            plot_settings = {'plottype': 'parameter sweep gender percentage',
                             'intervals': 'empirical',
                             'number_of_runs': 10,
                             # number simulations to average over
                             'target': 0.25,
                             'xlabels': ['Hiring Rate for Women',
                                         'Hiring Rate for Women',
                                         'Hiring Rate for Women',
                                         'Hiring Rate for Women',
                                         'Hiring Rate for Women',
                                         'Hiring Rate for Women'],
                             'ylabels': ['Proportion Women',
                                         'Proportion Women',
                                         'Proportion Women',
                                         'Proportion Women',
                                         'Proportion Women',
                                         'Proportion Women'],
                             'model_legend_label': ['Model 1, Hire-Promote',
                                                    'Model '
                                                    '2, '
                                                    'Promote-Hire'],
                             'parameter_sweep_param': 'bf1',
                             'parameter_ubound': 0.6,
                             'parameter_lbound': 0.05,
                             'number_of_steps': 5
                             }
            c.plot_comparison_level_chart(**plot_settings)
        assert(len(w) == 0)


    def test_parameter_sweep_function_validation_level_val(self, mgmt_data):
        """
        This testcase executes a parameter sweep and captures any warnings
        generated during the construction of the bokeh figure object.
        :param mgmt_data: Use management department data
        :type mgmt_data: dictionary
        :return: Pass/Fail
        :rtype: Boolean
        """

        modlist = list([Mod_Stoch_FBHP(**mgmt_data)])

        c = Comparison(modlist)

        plot_settings = {'plottype': 'parameter sweep gender percentage',
                         'intervals': 'empirical',
                         'number_of_runs': 10,
                         # number simulations to average over
                         'target': 0.25,
                         'xlabels': ['Hiring Rate for Women',
                                     'Hiring Rate for Women',
                                     'Hiring Rate for Women',
                                     'Hiring Rate for Women',
                                     'Hiring Rate for Women',
                                     'Hiring Rate for Women'],
                         'ylabels': ['Proportion Women',
                                     'Proportion Women',
                                     'Proportion Women',
                                     'Proportion Women',
                                     'Proportion Women',
                                     'Proportion Women'],
                         'model_legend_label': ['Model 1, Hire-Promote',
                                                'Model '
                                                '2, '
                                                'Promote-Hire'],
                         'parameter_sweep_param': 'bf1',
                         'parameter_ubound': 0.6,
                         'parameter_lbound': 0.05,
                         'number_of_steps': 5
                         }
        show(c.plot_comparison_level_chart(**plot_settings))



    @pytest.mark.skip(reason="This needs to be developed before testings")
    def test_parameter_sweep_probability_overall(self, mgmt_data):

        t = Mod_Stoch_FBPH(**mgmt_data)
        t.run_probability_parameter_sweep_overall(10,
                                                  'bf1',
                                                  0.05,
                                                  0.5,
                                                  4,
                                                  0.50)
        assert (hasattr(t, 'probability_matrix'))
