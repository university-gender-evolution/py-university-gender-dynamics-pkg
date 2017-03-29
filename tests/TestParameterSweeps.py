import pytest
from pyugend.pyugend.Mod_Stoch_FBHP import Mod_Stoch_FBHP
from pyugend.pyugend.Mod_Stoch_FBPH import Mod_Stoch_FBPH
from pyugend.pyugend.Mod_Validate_Sweep import Mod_Validate_Sweep
from pyugend.pyugend.Comparison import Comparison
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.charts import defaults

defaults.height = 800
defaults.width = 800

@pytest.mark.usefixtures('mgmt_data')
class Test_param_sweeps:

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_comparison_model_param_sweep_detail(self, mgmt_data):
        t = Mod_Stoch_FBPH(**mgmt_data)
        t.run_parameter_sweep(10, 'female_promotion_probability_2', 0.1, 0.5, 8)
        assert (hasattr(t, 'parameter_sweep_results'))

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_parameter_sweep_probability_overall(self, mgmt_data):
        t = Mod_Stoch_FBPH(**mgmt_data)
        t.run_probability_parameter_sweep_overall(10,
                                                  'hiring_rate_women_1',
                                                  0.05,
                                                  0.5,
                                                  4,
                                                  0.50)
        assert (hasattr(t, 'probability_matrix'))

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_parameter_sweep_function_validation_overall_val(self, mgmt_data):
        modlist = list([Mod_Validate_Sweep(**mgmt_data)])

        c = Comparison(modlist)

        plot_settings = {'plottype': 'parameter sweep percentage',
                         'intervals': 'empirical',
                         'number_of_runs': 10,
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
                         'number_of_steps': 5
                         }
        show(c.plot_comparison_overall_chart(**plot_settings))

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_parameter_sweep_function_validation_overall_hp(self, mgmt_data):
        modlist = list([Mod_Stoch_FBHP(**mgmt_data)])

        c = Comparison(modlist)

        plot_settings = {'plottype': 'parameter sweep percentage',
                         'intervals': 'empirical',
                         'number_of_runs': 500,
                         # number simulations to average over
                         'target': 0.25,
                         'xlabel': 'Promotion Rate for Women',
                         'ylabel': 'Proportion Women',
                         'title': 'Sweep, Hire-Promote, Female Promotion Rate 1->2',
                         'model_legend_label': ['Model 1, Hire-Promote',
                                                'Model '
                                                '2, '
                                                'Promote-Hire'],
                         'parameter_sweep_param': 'female_promotion_probability_1',
                         'parameter_ubound': 0.5,
                         'parameter_lbound': 0.05,
                         'number_of_steps': 15
                         }
        show(c.plot_comparison_overall_chart(**plot_settings))

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_parameter_sweep_function_validation_overall_ph(self, mgmt_data):
        modlist = list([Mod_Stoch_FBPH(**mgmt_data)])

        c = Comparison(modlist)

        plot_settings = {'plottype': 'parameter sweep percentage',
                         'intervals': 'empirical',
                         'number_of_runs': 10,
                         # number simulations to average over
                         'target': 0.25,
                         'xlabel': 'Hiring Rate for Women',
                         'ylabel': 'Proportion Women',
                         'title': 'Parameter Sweep Validation, Promote-Hire',
                         'model_legend_label': ['Model 1, Hire-Promote',
                                                'Model '
                                                '2, '
                                                'Promote-Hire'],
                         'parameter_sweep_param': 'bf1',
                         'parameter_ubound': 0.6,
                         'parameter_lbound': 0.05,
                         'number_of_steps': 5
                         }
        show(c.plot_comparison_overall_chart(**plot_settings))

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_parameter_sweep_function_validation_level_val(self, mgmt_data):
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
        show(c.plot_comparison_level_chart(**plot_settings))

    #@pytest.mark.skip(reason="no way of currently testing this")
    def test_parameter_sweep_function_validation_level_hp(self, mgmt_data):
        modlist = list([Mod_Stoch_FBPH(**mgmt_data)])

        c = Comparison(modlist)

        plot_settings = {'plottype': 'parameter sweep gender percentage',
                         'intervals': 'empirical',
                         'number_of_runs': 500,
                         # number simulations to average over
                         'target': 0.25,
                         'xlabels': ['Promotion Rate for Women',
                                     'Promotion Rate for Women',
                                     'Promotion Rate for Women',
                                     'Promotion Rate for Women',
                                     'Promotion Rate for Women',
                                     'Promotion Rate for Women'],
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
                         'parameter_sweep_param': 'female_promotion_probability_1',
                         'parameter_ubound': 0.5,
                         'parameter_lbound': 0.05,
                         'number_of_steps': 15
                         }
        show(c.plot_comparison_level_chart(**plot_settings))

    #@pytest.mark.skip(reason="no way of currently testing this")
    def test_parameter_sweep_function_validation_level_hp(self, mgmt_data):
        modlist = list([Mod_Stoch_FBPH(**mgmt_data)])

        c = Comparison(modlist)

        plot_settings = {'plottype': 'parameter sweep gender percentage',
                         'intervals': 'empirical',
                         'number_of_runs': 500,
                         # number simulations to average over
                         'target': 0.25,
                         'xlabels': ['Attrition Rate for Men_3',
                                     'Attrition Rate for Men_3',
                                     'Attrition Rate for Men_3',
                                     'Attrition Rate for Men_3',
                                     'Attrition Rate for Men_3',
                                     'Attrition Rate for Men_3'],
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
                         'parameter_sweep_param': 'dm3',
                         'parameter_ubound': 0.5,
                         'parameter_lbound': 0.05,
                         'number_of_steps': 15
                         }
        show(c.plot_comparison_level_chart(**plot_settings))

    #@pytest.mark.skip(reason="no way of currently testing this")
    def test_parameter_sweep_function_validation_level_hp(self, mgmt_data):
        modlist = list([Mod_Stoch_FBPH(**mgmt_data)])

        c = Comparison(modlist)

        plot_settings = {'plottype': 'parameter sweep gender percentage',
                         'intervals': 'empirical',
                         'number_of_runs': 500,
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
                         'parameter_ubound': 0.5,
                         'parameter_lbound': 0.05,
                         'number_of_steps': 15
                         }
        show(c.plot_comparison_level_chart(**plot_settings))

    #@pytest.mark.skip(reason="no way of currently testing this")
    def test_parameter_sweep_function_validation_overall_hp(self, mgmt_data):
        modlist = list([Mod_Stoch_FBPH(**mgmt_data)])

        c = Comparison(modlist)

        plot_settings = {'plottype': 'parameter sweep percentage',
                         'intervals': 'empirical',
                         'number_of_runs': 500,
                         # number simulations to average over
                         'target': 0.25,
                         'xlabel': 'Hiring Rate for Women',
                         'ylabel': 'Proportion Women',
                         'title': 'Sweep, Hire-Promote, Female Promotion Rate 1->2',
                         'model_legend_label': ['Model 1, Hire-Promote',
                                                'Model '
                                                '2, '
                                                'Promote-Hire'],
                         'parameter_sweep_param': 'female_promotion_probability_1',
                         'parameter_ubound': 0.5,
                         'parameter_lbound': 0.05,
                         'number_of_steps': 15
                         }
        show(c.plot_comparison_overall_chart(**plot_settings))

    #@pytest.mark.skip(reason="no way of currently testing this")
    def test_parameter_sweep_function_validation_overall_hp(self, mgmt_data):
        modlist = list([Mod_Stoch_FBPH(**mgmt_data)])

        c = Comparison(modlist)

        plot_settings = {'plottype': 'parameter sweep percentage',
                         'intervals': 'empirical',
                         'number_of_runs': 500,
                         # number simulations to average over
                         'target': 0.25,
                         'xlabel': 'Hiring Rate for Women',
                         'ylabel': 'Attrition Rate Men_3',
                         'title': 'Sweep, Hire-Promote, Attrition Rate Male_3',
                         'model_legend_label': ['Model 1, Hire-Promote',
                                                'Model '
                                                '2, '
                                                'Promote-Hire'],
                         'parameter_sweep_param': 'dm3',
                         'parameter_ubound': 0.5,
                         'parameter_lbound': 0.05,
                         'number_of_steps': 15
                         }
        show(c.plot_comparison_overall_chart(**plot_settings))

    #@pytest.mark.skip(reason="no way of currently testing this")
    def test_parameter_sweep_function_validation_overall_hp(self, mgmt_data):
        modlist = list([Mod_Stoch_FBPH(**mgmt_data)])

        c = Comparison(modlist)

        plot_settings = {'plottype': 'parameter sweep percentage',
                         'intervals': 'empirical',
                         'number_of_runs': 500,
                         # number simulations to average over
                         'target': 0.25,
                         'xlabel': 'Hiring Rate for Women',
                         'ylabel': 'Proportion Women',
                         'title': 'Sweep, Hire-Promote, Female Promotion Rate 1->2',
                         'model_legend_label': ['Model 1, Hire-Promote',
                                                'Model '
                                                '2, '
                                                'Promote-Hire'],
                         'parameter_sweep_param': 'bf1',
                         'parameter_ubound': 0.5,
                         'parameter_lbound': 0.05,
                         'number_of_steps': 15
                         }
        show(c.plot_comparison_overall_chart(**plot_settings))



    @pytest.mark.skip(reason="no way of currently testing this")
    def test_parameter_sweep_function_validation_level_ph(self, mgmt_data):
        modlist = list([Mod_Stoch_FBPH(**mgmt_data)])

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
                         'title': 'Parameter Sweep Validation, Promote-Hire',
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
