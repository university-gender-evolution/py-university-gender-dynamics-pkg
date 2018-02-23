import pytest
from pyugend.Mod_Stoch_FBHP import Mod_Stoch_FBHP
from pyugend.Mod_Stoch_FBPH import Mod_Stoch_FBPH
from pyugend.ModelGenderDiversity import Model3GenderDiversity
from pyugend.Mod_Validate_Sweep import Mod_Validate_Sweep
from pyugend.Comparison import Comparison
from bokeh.plotting import figure, output_file, show
from pyugend.ModelGenderDiversityGrowthForecast import ModelGenderDiversityGrowthForecast
from pyugend.ModelGenderDiversityLinearGrowth import ModelGenderDiversityLinearGrowth
from pyugend.ModelGenderDiversityGrowthForecastIncrementalChange import ModelGenderDiversityGrowthForecastIncremental

height = 800
width = 800


@pytest.mark.usefixtures('mgmt_data')
@pytest.mark.usefixtures('mock_data')
def test_bokeh_comparison_plot_overall_one_model(mgmt_data):
    modlist = list([ModelGenderDiversityGrowthForecastIncremental(**mgmt_data)])
    # modlist = list([Model2GenderDiversity(**mgmt_data),
    #                 Mod_Stoch_FBPH(**mgmt_data)])
    modlist[0].init_default_hiring_rate()
    modlist[0].init_growth_rate([0.02, 0.01, 0.10, 0.05])
    # modlist[0].init_growth_rate([0.015])

    c = Comparison(modlist)

    # print(modlist[0].calculate_yearly_dept_size_targets())

    plot_settings = {'plottype': 'gender proportion',
                     'intervals': 'empirical',
                     'number_of_runs': 100,
                     # number simulations to average over
                     'target': 0.25,
                     # target percentage of women in the department
                     # Main plot settings
                     'xlabel': 'Years',
                     'ylabel': 'Proportion Women',
                     'title': 'Change in Proportion Women',
                     'line_width': 2,
                     'transparency': 0.25,
                     'model_legend_label': ['New Model',
                                            'Mode 2, Promote-Hire'],
                     'legend_location': 'top_right',
                     'height_': height,
                     'width_': width,
                     'year_offset': 0
                     }
    show(c.plot_comparison_overall_chart(**plot_settings))


def test_bokeh_comparison_plot_dept_size_overall(mgmt_data):
    modlist = list([Model3GenderDiversity(**mgmt_data),
                    ModelGenderDiversityLinearGrowth(**mgmt_data),
                    ModelGenderDiversityGrowthForecast(**mgmt_data)])
    modlist[0].init_default_hiring_rate()
    modlist[1].init_default_hiring_rate()
    modlist[1].init_growth_rate(0.01)
    modlist[2].init_default_hiring_rate()
    modlist[2].init_growth_rate([73, 78, 83, 88])

    c = Comparison(modlist)
    plot_settings = {'plottype': 'department size',
                     'intervals': 'empirical',
                     'number_of_runs': 100,
                     # number simulations to average over
                     'target': 0.25,
                     # target percentage of women in the department
                     # Main plot settings
                     'xlabel': 'Years',
                     'ylabel': 'Department Size',
                     'title': 'Department Size',
                     'line_width': 2,
                     'transparency': 0.25,
                     'model_legend_label': ['Model 3 No Growth',
                                            'Model 3 Lin Growth(1%/year)',
                                            'Model 3 Forecast(+5/5 year)'],
                     'legend_location': 'top_right',
                     'height_': height,
                     'width_': width,

                     }
    show(c.plot_comparison_overall_chart(**plot_settings))


def test_bokeh_comparison_plot_overall_multiple_models(mgmt_data):
    modlist = list([Model3GenderDiversity(**mgmt_data),
                    ModelGenderDiversityLinearGrowth(**mgmt_data),
                    ModelGenderDiversityGrowthForecast(**mgmt_data)])
    modlist[0].init_default_hiring_rate()
    modlist[1].init_default_hiring_rate()
    modlist[1].init_growth_rate(0.05)
    modlist[2].init_default_hiring_rate()
    modlist[2].init_growth_rate([73, 78, 83, 88])

    c = Comparison(modlist)

    plot_settings = {'plottype': 'gender proportion',
                     'intervals': 'empirical',
                     'number_of_runs': 100,
                     # number simulations to average over
                     'target': 0.25,
                     # target percentage of women in the department
                     # Main plot settings
                     'xlabel': 'Years',
                     'ylabel': 'Proportion Women',
                     'title': 'Change in Proportion Women Overall',
                     'transparency': 0.25,
                     'model_legend_label': ['Model 3 No Growth',
                                            'Model 3 Lin Growth',
                                            'Model 3 Forecast']

                     }
    show(c.plot_comparison_overall_chart(**plot_settings))


def test_bokeh_comparison_plot_bylevel(mgmt_data):
    modlist = list([Model3GenderDiversity(**mgmt_data),
                    ModelGenderDiversityLinearGrowth(**mgmt_data),
                    ModelGenderDiversityGrowthForecast(**mgmt_data)])
    modlist[0].init_default_hiring_rate()
    modlist[1].init_default_hiring_rate()
    modlist[1].init_growth_rate(0.05)
    modlist[2].init_default_hiring_rate()
    modlist[2].init_growth_rate([73, 78, 83, 88])

    c = Comparison(modlist)
    plot_settings = {'plottype': 'gender number',
                     'intervals': 'empirical',
                     'number_of_runs': 100,
                     'target': 0.25,
                     'line_width': 2,
                     'model_legend_label': ['Model 3 No Growth',
                                            'Model 3 Lin Growth',
                                            'Model 3 Forecast'],
                     'transparency': 0.25,
                     'legend_location': 'top right',
                     'height_': 400,
                     'width_': 400,
                     # main plot axis labels
                     'xlabels': ['Years', 'Years', 'Years', 'Years',
                                 'Years', 'Years'],
                     'ylabels': ['Number of Women', 'Number of Women',
                                 'Number of Women', 'Number of Men',
                                 'Number of Men', 'Number of Men'],
                     'titles': ['f1', 'f2', 'f3', 'm1', 'm2', 'm3'],

                     # target plot settings
                     'target_plot': True,
                     'target_color': 'red',
                     'target_plot_linewidth': 2,
                     'target_number_labels': ['Target Model 3 NG',
                                              'Target Model 3 LG',
                                              'Target Model 3 FG'],

                     # percent plot settings
                     'percent_line_plot': False,
                     'percent_line_value': 0.5,
                     'color_percent_line': 'red',
                     'percent_linewidth': 2,
                     'percent_legend_label': 'Reference Line'
                     }

    show(c.plot_comparison_level_chart(**plot_settings))


def test_bokeh_sweep_plot_overall(mgmt_data):
    modlist = list([Model3GenderDiversity(**mgmt_data)])
    # modlist = list([Model2GenderDiversity(**mgmt_data),
    #                 Mod_Stoch_FBPH(**mgmt_data)])
    modlist[0].init_default_hiring_rate()

    c = Comparison(modlist)

    plot_settings = {'plottype': 'parameter sweep percentage',
                     'intervals': 'empirical',
                     'number_of_runs': 100,
                     # number simulations to average over
                     'target': 0.25,
                     'xlabel': 'Years',
                     'ylabel': 'Proportion Women',
                     'title': 'Parameter Sweep Gender Percentage',
                     'model_legend_label': ['New Model',
                                            'Model '
                                            '2, '
                                            'Promote-Hire'],
                     'parameter_sweep_param': 'bf1',
                     'parameter_ubound': 0.6,
                     'parameter_lbound': 0.05,
                     'number_of_steps': 5
                     }
    show(c.plot_comparison_overall_chart(**plot_settings))


def test_bokeh_comparison_plot_probability_bylevel(mgmt_data):
    modlist = list([Model3GenderDiversity(**mgmt_data)])
    # modlist = list([Model2GenderDiversity(**mgmt_data),
    #                 Mod_Stoch_FBPH(**mgmt_data)])
    modlist[0].init_default_hiring_rate()

    c = Comparison(modlist)

    plot_settings = {'plottype': 'probability proportion',
                     'intervals': 'empirical',
                     'number_of_runs': 100,
                     'target': 0.25,
                     'line_width': 2,
                     'model_legend_label': ['new model',
                                            'model2, Promote-Hire'],
                     'legend_location': 'top right',
                     'height_': 300,
                     'width_': 300,
                     # main plot axis labels
                     'xlabels': ['Years', 'Years', 'Years', 'Years',
                                 'Years', 'Years'],
                     'ylabels': ['Probability of Target',
                                 'Probability of Target',
                                 'Probability of Target',
                                 'Probability of Target',
                                 'Probability of Target',
                                 'Probability of Target'],
                     'titles': ['f1', 'f2', 'f3', 'm1', 'm2', 'm3'],
                     }

    show(c.plot_comparison_level_chart(**plot_settings))


def test_bokeh_comparison_plot_sweep_bylevel(mgmt_data):
    modlist = list([Model3GenderDiversity(**mgmt_data)])

    # modlist = list([Model2GenderDiversity(**mgmt_data),
    #                 Mod_Stoch_FBPH(**mgmt_data)])
    modlist[0].init_default_hiring_rate()
    c = Comparison(modlist)

    plot_settings = {'plottype': 'parameter sweep percentage',
                     'intervals': 'empirical',
                     'number_of_runs': 100,
                     'target': 0.25,
                     'line_width': 2,
                     'model_legend_label': ['new model',
                                            'model2, Promote-Hire'],
                     'legend_location': 'top right',
                     'height_': 300,
                     'width_': 300,
                     # main plot axis labels
                     'xlabels': ['Years', 'Years', 'Years', 'Years',
                                 'Years', 'Years'],
                     'ylabels': ['Percentage of Women',
                                 'Percentage of Women',
                                 'Percentage of Women', 'Percentage of Men',
                                 'Percentage of Men', 'Percentage of Men'],
                     'titles': ['f1', 'f2', 'f3', 'm1', 'm2', 'm3'],
                     'parameter_sweep_param': 'bf1',
                     'parameter_ubound': 0.6,
                     'parameter_lbound': 0.05,
                     'number_of_steps': 5
                     }

    show(c.plot_comparison_level_chart(**plot_settings))


def test_plot_overall_unfilled_vacancies(mgmt_data):
    modlist = list([Model3GenderDiversity(**mgmt_data)])
    # modlist = list([Model2GenderDiversity(**mgmt_data),
    #                 Mod_Stoch_FBPH(**mgmt_data)])
    modlist[0].init_default_hiring_rate()
    c = Comparison(modlist)

    plot_settings = {'plottype': 'unfilled vacancies',
                     'intervals': 'empirical',
                     'number_of_runs': 100,
                     # number simulations to average over
                     'target': 0.25,
                     # target percentage of women in the department
                     # Main plot settings
                     'xlabel': 'Years',
                     'ylabel': 'Unfilled Vacancies',
                     'title': 'Unfilled Vacancies',
                     'line_width': 2,
                     'transparency': 0.25,
                     'model_legend_label': ['New Model',
                                            'Mode 2, Promote-Hire'],
                     'legend_location': 'top_right',
                     'height_': height,
                     'width_': width,

                     }
    show(c.plot_comparison_overall_chart(**plot_settings))


def test_plot_overall_mf_numbers(mgmt_data):
    modlist = list([Model3GenderDiversity(**mgmt_data)])
    # modlist = list([Model2GenderDiversity(**mgmt_data),
    #                 Mod_Stoch_FBPH(**mgmt_data)])
    modlist[0].init_default_hiring_rate()
    c = Comparison(modlist)

    plot_settings = {'plottype': 'male female numbers',
                     'intervals': 'empirical',
                     'number_of_runs': 100,
                     # number simulations to average over
                     'target': 0.25,
                     # target percentage of women in the department
                     # Main plot settings
                     'xlabel': 'Years',
                     'ylabel': 'Number of Professors',
                     'title': 'Male Female numbers plot',
                     'line_width': 2,
                     'transparency': 0.25,
                     'model_legend_label': ['Model 3',
                                            'Mode 2, Promote-Hire'],
                     'legend_location': 'top_right',
                     'height_': height,
                     'width_': width,
                     'male_female_numbers_plot': True,
                     'mf_male_label': ['Male Model 3', 'Female Model 3'],
                     'mf_target_label': ['Target 3']
                     }
    show(c.plot_comparison_overall_chart(**plot_settings))
