import pytest
from pyugend.pyugend.Mod_Stoch_FBHP import Mod_Stoch_FBHP
from pyugend.pyugend.Mod_Stoch_FBPH import Mod_Stoch_FBPH
from pyugend.pyugend.Mod_Validate_Sweep import Mod_Validate_Sweep
from pyugend.pyugend.Comparison import Comparison
from bokeh.plotting import figure, output_file, show
from bokeh.charts import defaults

defaults.height = 800
defaults.width = 800

class TestPlots:
    def test_bokeh_comparison_plot_overall_one_model(self, mgmt_data):
        modlist = list([Mod_Stoch_FBHP(**mgmt_data)])
        # modlist = list([Mod_Stoch_FBHP(**mgmt_data),
        #                 Mod_Stoch_FBPH(**mgmt_data)])
        c = Comparison(modlist)

        plot_settings = {'plottype': 'gender proportion',
                         'intervals': 'empirical',
                         'number_of_runs': 10,
                         # number simulations to average over
                         'target': 0.25,
                         # target percentage of women in the department
                         # Main plot settings
                         'xlabel': 'Years',
                         'ylabel': 'Proportion Women',
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

    def test_bokeh_comparison_plot_overall_multiple_models(self, mgmt_data):
        modlist = list([Mod_Stoch_FBHP(**mgmt_data),
                        Mod_Stoch_FBPH(**mgmt_data)])
        c = Comparison(modlist)

        plot_settings = {'plottype': 'gender proportion',
                         'intervals': 'empirical',
                         'number_of_runs': 10,
                         # number simulations to average over
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

    def test_bokeh_comparison_plot_bylevel(self, mgmt_data):
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
                         'xlabels': ['Years', 'Years', 'Years', 'Years',
                                     'Years', 'Years'],
                         'ylabels': ['Proportion Women', 'Proportion Women',
                                     'Proportion Women', 'Proportion Women',
                                     'Proportion Women', 'Proportion Women'],
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

    def test_bokeh_sweep_plot_overall(self, mgmt_data):
        # modlist = list([Mod_Stoch_FBHP(**mgmt_data)])
        modlist = list([Mod_Stoch_FBHP(**mgmt_data),
                        Mod_Stoch_FBPH(**mgmt_data)])
        c = Comparison(modlist)

        plot_settings = {'plottype': 'parameter sweep percentage',
                         'intervals': 'empirical',
                         'number_of_runs': 10,
                         # number simulations to average over
                         'target': 0.25,
                         'xlabel': 'Years',
                         'ylabel': 'Proportion Women',
                         'title': 'Figure 4.1.3a: Change in Proportion Women, Model 1',
                         'model_legend_label': ['Model 1, Hire-Promote',
                                                'Model '
                                                '2, '
                                                'Promote-Hire'],
                         'parameter_sweep_param': 'hiring_rate_women_1',
                         'parameter_ubound': 0.6,
                         'parameter_lbound': 0.05,
                         'number_of_steps': 5
                         }
        show(c.plot_comparison_overall_chart(**plot_settings))

