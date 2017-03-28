import pytest
from pyugend.pyugend.Models import Base_model
from pyugend.pyugend.Mod_Stoch_FBHP import Mod_Stoch_FBHP
from pyugend.pyugend.Mod_Stoch_FBPH import Mod_Stoch_FBPH
from pyugend.pyugend.Mod_Validate_Sweep import Mod_Validate_Sweep
from pyugend.pyugend.ReplicationModel import Replication_model
from pyugend.pyugend.Comparison import Comparison
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.charts import defaults

defaults.height = 800
defaults.width = 800

## Tests for base model class initialization



# Tests for CSV export of model results



# Tests for Model Simulation and Analysis functions

def test_comparison_model_param_sweep_detail(mgmt_data):
    t = Mod_Stoch_FBPH(**mgmt_data)
    t.run_parameter_sweep(10, 'female_promotion_probability_2', 0.1, 0.5, 8)
    assert(hasattr(t, 'parameter_sweep_results'))

def test_base_model_probability_calc_detail_array(mgmt_data):
    t = Mod_Stoch_FBPH(**mgmt_data)
    res = t.run_probability_analysis_parameter_sweep_gender_detail(10,
                                                                   'female_promotion_probability_2',
                                                                   'm2', 0.1,
                                                                   0.8, 8, 150)
    assert (isinstance(res, pd.DataFrame))

def test_parameter_sweep_probability_overall(mgmt_data):
    t = Mod_Stoch_FBPH(**mgmt_data)
    t.run_probability_parameter_sweep_overall(10,
                                              'hiring_rate_women_1',
                                              0.05,
                                              0.5,
                                              4,
                                              0.50)
    assert (hasattr(t, 'probability_matrix'))


def test_parameter_sweep_function_validation_overall(mgmt_data):
    modlist = list([Mod_Validate_Sweep(**mgmt_data)])
    c = Comparison(modlist)

    #assert (isinstance(c, Comparison))
    plot_settings = {'plottype': 'parameter sweep percentage',
                     'intervals': 'empirical',
                     'number_of_runs': 10,  # number simulations to average over
                     'target': 0.25,
                     'xlabel': 'Hiring Rate for Women',
                     'ylabel': 'Proportion Women',
                     'title': 'Parameter Sweep Validation, uniform noise(0,'
                              '0.1)',
                     'model_legend_label': ['Model 1, Hire-Promote', 'Model '
                                                                     '2, '
                                                                     'Promote-Hire'],
                     'parameter_sweep_param': 'bf1',
                     'parameter_ubound': 0.6,
                     'parameter_lbound': 0.05,
                     'number_of_steps': 5
                     }
    show(c.plot_comparison_overall_chart(**plot_settings))

def test_parameter_sweep_function_validation_bylevel(mgmt_data):
    modlist = list([Mod_Validate_Sweep(**mgmt_data)])
    c = Comparison(modlist)

    #assert (isinstance(c, Comparison))
    plot_settings = {'plottype': 'parameter sweep percentage',
                     'intervals': 'empirical',
                     'number_of_runs': 10,  # number simulations to average over
                     'target': 0.25,
                     'xlabel': 'Hiring Rate for Women',
                     'ylabel': 'Proportion Women',
                     'title': 'Parameter Sweep Validation, uniform noise(0,'
                              '0.1)',
                     'model_legend_label': ['Model 1, Hire-Promote', 'Model '
                                                                     '2, '
                                                                     'Promote-Hire'],
                     'parameter_sweep_param': 'bf1',
                     'parameter_ubound': 0.6,
                     'parameter_lbound': 0.05,
                     'number_of_steps': 5
                     }
    show(c.plot_comparison_overall_chart(**plot_settings))

# Test for plots of simulation results.

def test_bokeh_comparison_plot_overall_one_model(mgmt_data):
    modlist = list([Mod_Stoch_FBHP(**mgmt_data)])
    # modlist = list([Mod_Stoch_FBHP(**mgmt_data),
    #                 Mod_Stoch_FBPH(**mgmt_data)])
    c = Comparison(modlist)

    plot_settings = {'plottype': 'gender proportion',
                     'intervals': 'empirical',
            'number_of_runs': 10,  # number simulations to average over
            'target': 0.25,  # target percentage of women in the department
            # Main plot settings
            'xlabel':'Years',
            'ylabel': 'Proportion Women' ,
            'title': 'Figure 4.1.3a: Change in Proportion Women, Model 1',
            'line_width': 2,
            'transparency': 0.25,
            'linecolor': ['green'],
            'model_legend_label': ['Model 1, Hire-Promote'],
            'legend_location': 'top_right',
            'height_': 800,
            'width_': 800,

            }
    show(c.plot_comparison_overall_chart(**plot_settings))

def test_bokeh_comparison_plot_overall_multiple_models(mgmt_data):
    modlist = list([Mod_Stoch_FBHP(**mgmt_data),
                    Mod_Stoch_FBPH(**mgmt_data)])
    c = Comparison(modlist)

    plot_settings = {'plottype': 'gender proportion',
                     'intervals': 'empirical',
                     'number_of_runs': 10,  # number simulations to average over
                     'target': 0.25,
                     # target percentage of women in the department
                     # Main plot settings
                     'xlabel': 'Years',
                     'ylabel': 'Proportion Women',
                     'title': 'Figure 4.1.3a: Change in Proportion Women, Compare Models 1 and 2',
                     'transparency': 0.25,
                     'model_legend_label': ['Model 1, Hire-Promote',
                                            'Model 2, Promote-Hire']

                     }
    show(c.plot_comparison_overall_chart(**plot_settings))

def test_bokeh_comparison_plot_bylevel(mgmt_data):
    modlist = list([Mod_Stoch_FBHP(**mgmt_data),
                    Mod_Stoch_FBPH(**mgmt_data)])
    c = Comparison(modlist)

    plot_settings = {'plottype': 'gender number',
                     'intervals': 'empirical',
                 'number_of_runs': 10,
                 'target': 0.25,
                 'line_width': 2,
                 'model_legend_label': ['model 1', 'model2'],
                 'transparency': 0.25,
                 'linecolor': ['green', 'blue'],
                 'legend_location': 'top right',
                 'height_': 400,
                 'width_': 400,
                 # main plot axis labels
                 'xlabels' : ['Years', 'Years','Years','Years','Years','Years'],
                 'ylabels': ['Proportion Women','Proportion Women',
                             'Proportion Women','Proportion Women',
                             'Proportion Women','Proportion Women'],
                 'titles': ['f1', 'f2', 'f3', 'm1', 'm2', 'm3'],


                 # target plot settings
                 'target_plot': True,
                 'target_color': 'red',
                 'target_plot_linewidth': 2,
                 'target_plot_legend_label': 'target',

                 # percent plot settings
                 'percent_line_plot': True,
                 'percent_line_value': 0.5,
                 'color_percent_line': 'red',
                 'percent_linewidth': 2,
                 'percent_legend_label': 'Reference Line'
                     }

    show(c.plot_comparison_level_chart(**plot_settings))

def test_bokeh_sweep_plot_overall(mgmt_data):
    #modlist = list([Mod_Stoch_FBHP(**mgmt_data)])
    modlist = list([Mod_Stoch_FBHP(**mgmt_data),
                     Mod_Stoch_FBPH(**mgmt_data)])
    c = Comparison(modlist)

    plot_settings = {'plottype': 'parameter sweep percentage',
                     'intervals': 'empirical',
                     'number_of_runs': 10,  # number simulations to average over
                     'target': 0.25,
                     'xlabel': 'Years',
                     'ylabel': 'Proportion Women',
                     'title': 'Figure 4.1.3a: Change in Proportion Women, Model 1',
                     'model_legend_label': ['Model 1, Hire-Promote', 'Model '
                                                                     '2, '
                                                                     'Promote-Hire'],
                     'parameter_sweep_param': 'hiring_rate_women_1',
                     'parameter_ubound': 0.6,
                     'parameter_lbound': 0.05,
                     'number_of_steps': 5
                     }
    show(c.plot_comparison_overall_chart(**plot_settings))

