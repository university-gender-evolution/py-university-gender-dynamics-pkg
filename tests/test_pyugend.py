import pytest
from pyugend.Models import Base_model
from pyugend.Mod_Stoch_FBHP import Mod_Stoch_FBHP
from pyugend.Mod_Stoch_FBPH import Mod_Stoch_FBPH
from pyugend.ReplicationModel import Replication_model
from pyugend.Comparison import Comparison
import numpy as np
import pandas as pd


@pytest.fixture
def mock_data():
    return ({'number_of_females_1': 14,
             'number_of_females_2': 3,
             'number_of_females_3': 19,
             'number_of_males_1': 37,
             'number_of_males_2': 28,
             'number_of_males_3': 239,
             'number_of_initial_vacancies_1': 5.303,
             'number_of_initial_vacancies_2': 5.9,
             'number_of_initial_vacancies_3': 8.31,
             'hiring_rate_women_1': 0.310,
             'hiring_rate_women_2': 0.222,
             'hiring_rate_women_3': 0,
             'attrition_rate_women_1': 0,
             'attrition_rate_women_2': 0,
             'attrition_rate_women_3': 0.017,
             'attrition_rate_men_1': 0.009,
             'attrition_rate_men_2': 0.017,
             'attrition_rate_men_3': 0.033,
             'probablity_of_outside_hire_1': 1,
             'probability_of_outside_hire_2': 0.158,
             'probability_of_outside_hire_3': 0.339,
             'female_promotion_probability_1': 0.122,
             'female_promotion_probability_2': 0.188,
             'male_promotion_probability_1': 0.19,
             'male_promotion_probability_2': 0.19,
             'upperbound': 350,
             'lowerbound': 330,
             'variation_range': 3,
             'duration': 40})


@pytest.fixture
def mgmt_data():
    return ({'number_of_females_1': 3,
             'number_of_females_2': 3,
             'number_of_females_3': 2,
             'number_of_males_1': 11,
             'number_of_males_2': 12,
             'number_of_males_3': 43,
             'number_of_initial_vacancies_1': 5.303,
             'number_of_initial_vacancies_2': 5.9,
             'number_of_initial_vacancies_3': 8.31,
             'hiring_rate_women_1': 0.172,
             'hiring_rate_women_2': 0.4,
             'hiring_rate_women_3': 0.167,
             'attrition_rate_women_1': 0.056,
             'attrition_rate_women_2': 0.00,
             'attrition_rate_women_3': 0.074,
             'attrition_rate_men_1': 0.069,
             'attrition_rate_men_2': 0.057,
             'attrition_rate_men_3': 0.040,
             'probablity_of_outside_hire_1': 1,
             'probability_of_outside_hire_2': 0.125,
             'probability_of_outside_hire_3': 0.150,
             'female_promotion_probability_1': 0.0555,
             'female_promotion_probability_2': 0.1905,
             'male_promotion_probability_1': 0.0635,
             'male_promotion_probability_2': 0.1149,
             'upperbound': 84,
             'lowerbound': 64,
             'variation_range': 3,
             'duration': 40})


def test_Base_model(mock_data):
    assert isinstance(Base_model(**mock_data), Base_model)


def test_base_model_run(mock_data):
    t = Base_model(**mock_data)
    t.run_model()
    assert (isinstance(t.res, np.ndarray))


def test_base_model_persistence(mock_data):
    t = Base_model(**mock_data)
    assert (t.nf1 == 14)


def test_base_model_multiple_runs(mock_data):
    t = Mod_Stoch_FBHP(**mock_data)
    assert (isinstance(t.run_multiple(10), int))


def test_base_model_multiple_runs_persistent_state(mock_data):
    t = Mod_Stoch_FBHP(**mock_data)
    t.run_multiple(10)
    assert (isinstance(t.mean_matrix, np.ndarray))


def test_base_model_parameter_sweep(mock_data):
    t = Mod_Stoch_FBHP(**mock_data)
    v = t.run_parameter_sweep(12, 'female_pp_1', 0.1, 0.3, 2)
    assert (isinstance(v, int))

def test_base_model_multiple_runs_gender_prop(mock_data):
     t = Mod_Stoch_FBHP(**mock_data)
     t.run_multiple(100)
     print(t.pct_female_matrix)


def test_basic_stochastic_model(mock_data):
    assert (isinstance(Mod_Stoch_FBHP(**mock_data),
                       Mod_Stoch_FBHP))


def test_basic_stochastic_model_run(mock_data):
    t = Mod_Stoch_FBHP(**mock_data).run_model()
    assert isinstance(t, np.recarray)


def test_basic_stochastic_model_run_with_saved_data(mock_data):
    t = Mod_Stoch_FBHP(**mock_data)
    t.run_model()
    assert isinstance(t.run, np.recarray)


def test_basic_stochastic_model_promotion_probability_recovery(mock_data):
    t = Mod_Stoch_FBHP(**mock_data)
    assert (t.female_promotion_probability_2 == 0.188)


def test_replication_model(mock_data):
    t = Replication_model(**mock_data)
    t.run_model()
    assert (isinstance(t.run, np.ndarray))

def test_excel_export(mock_data):
    t = Mod_Stoch_FBPH(**mock_data)
    t.export_model_run(10)

def test_comparison_model_plot_detail(mock_data):
    modlist = list([Mod_Stoch_FBHP(**mock_data),
                    Mod_Stoch_FBPH(**mock_data)])
    c = Comparison(modlist)
    c.plot_comparison_detail(10)


def test_comparison_plot_percentage_women_by_level(mock_data):
    modlist = list([Mod_Stoch_FBHP(**mock_data),
                    Mod_Stoch_FBPH(**mock_data)])
    c = Comparison(modlist)
    c.plot_comparison_percentage_women_by_level(0.25,10)


def test_comparison_model_param_sweep(mock_data):
    modlist = list([Mod_Stoch_FBHP(**mock_data),
                    Mod_Stoch_FBPH(**mock_data)])
    c = Comparison(modlist)
    c.plot_parameter_sweep_gender_proportion(10, 'female_promotion_probability_2', 0.1, 0.5, 8)


def test_comparison_model_param_sweep_detail(mock_data):
    modlist = list([Mod_Stoch_FBHP(**mock_data),
                    Mod_Stoch_FBPH(**mock_data)])
    c = Comparison(modlist)
    c.plot_parameter_sweep_detail(10, 'female_promotion_probability_2', 0.1, 0.5, 8)


def test_base_model_probability_calc_detail_array(mock_data):
    t = Mod_Stoch_FBPH(**mock_data)
    res = t.run_probability_analysis_parameter_sweep_gender_detail(10,
                                                                   'female_promotion_probability_2',
                                                                   'm2', 0.1, 0.8, 8, 150)
    print(res)
    assert (isinstance(res, pd.DataFrame))


def test_multiple_runs_created_res_array(mock_data):
    t = Mod_Stoch_FSPH(**mock_data)
    t.run_multiple(10)
    assert hasattr(t, 'mean_matrix')



def test_plot_comparision_empirical_probability_gender_proportion(mock_data):
    modlist = list([Mod_Stoch_FBHP(**mock_data),
                    Mod_Stoch_FBPH(**mock_data)])
    c = Comparison(modlist)
    c.plot_comparison_empirical_probability_gender_proportion(100, 0.19)

def test_plot_comparision_unfilled_vacancies(mock_data):
    modlist = list([Mod_Stoch_FBPH(**mock_data),
                    Mod_Stoch_FBHP(**mock_data)])
    c = Comparison(modlist)
    c.plot_comparison_spec_parameter('dept_size', 'department size',
                                     'year', 'department size')

def test_plot_comparison_gender_proportion(mgmt_data):

    modlist = list([Mod_Stoch_FBPH(**mgmt_data)])

    c = Comparison(modlist)
    c.plot_comparison_gender_proportion('Years',
                                          '% Female',
                                          'Comparison of Faculty Percentage',
                                          '',
                                          0.20,
                                          100)


def test_FBHP_plot_comparison_male_female_numbers(mgmt_data):

    modlist = list([Mod_Stoch_FBPH(**mgmt_data),
                    Mod_Stoch_FBHP(**mgmt_data)])

    c = Comparison(modlist)
    c.plot_comparison_female_male_numbers('Years',
                                          'Faculty Number',
                                          'Comparison of Faculty Numbers for '
                                          'each Model',
                                          'caption',
                                          0.20,
                                          100)

def test_run_probability_by_level_data(mgmt_data):
    t = Mod_Stoch_FBPH(**mgmt_data)
    val = t.run_probability_analysis_gender_by_level(10, 0.15)
    assert (isinstance(val, pd.DataFrame))


def test_plot_comparison_empirical_probability_proportion_by_level(mock_data):
    modlist = list([Mod_Stoch_FBHP(**mock_data),
                    Mod_Stoch_FBPH(**mock_data)])
    c = Comparison(modlist)
    c.plot_comparison_empirical_probability_proportion_by_level(0.25,100)

def test_plot_overall_probability_proportion(mgmt_data):
    t = Mod_Stoch_FBPH(**mgmt_data)
    d = {'plottype': 'probability proportion',
         'number_of_runs': 10,
         'target': 0.15,
         'caption': '',
         'xlabel':'test x',
         'ylabel': 'test y',
         'title': 'test title',
         'line_width': 2,
         'xmin': 0,
         'ymin': 0,
         'xmax': 40,
         'ymax': 1.0,
         'transparency': 0.25,
         'marker_shape': None,
         'linecolor': 'g',
         'target_plot': True,
         'legend_location': 'upper right',
         'color_target': 'r',
         'percent_line_plot': True,
         'percent_line_value':0.5,
         'color_percent_line': 'k',
         'target_plot_line_style': '--',
         'percent_line_style': '-.',
         'target_plot_linewidth': 2,
         'percent_linewidth': 2,
         'model_legend_label':'model',
         'target_plot_legend_label':'target',
         'percent_legend_label': '50%'
         }
    t.plot_overall_chart(**d)


def test_plot_overall_male_female_numbers(mgmt_data):
    t = Mod_Stoch_FBPH(**mgmt_data)
    d = {'plottype': 'male female numbers',
         'number_of_runs': 10,
         'target': 0.15,
         'caption': '',
         'xlabel':'test x',
         'ylabel': 'test y',
         'title': 'test title',
         'line_width': 2,
         'xmin': 0,
         'ymin': 0,
         'xmax': 40,
         'ymax': 100,
         'transparency': 0.25,
         'marker_shape': None,
         'linecolor': 'g',
         'target_plot': False,
         'legend_location': 'upper right',
         'color_target': 'r',
         'percent_line_plot': False,
         'percent_line_value':0.5,
         'color_percent_line': 'k',
         'target_plot_line_style': '--',
         'percent_line_style': '-.',
         'target_plot_linewidth': 2,
         'percent_linewidth': 2,
         'model_legend_label':'model',
         'target_plot_legend_label':'target',
         'percent_legend_label': '50%',
         'male_female_numbers_plot': True,
         'mf_male_color' : 'k',
         'mf_target_color' : 'r',
         'mf_male_label' : 'Male',
         'mf_target_label': 'Target',
         'mf_male_linestyle' : '-',
         'mf_target_linestyle': '-',
         'mf_male_linewidth' :2,
         'mf_target_linewidth' : 2
         }
    t.plot_overall_chart(**d)


def test_plot_overall_gender_proportion(mgmt_data):
    t = Mod_Stoch_FBPH(**mgmt_data)
    d = {'plottype': 'gender proportion',
         'number_of_runs': 10,
         'target': 0.25,
         'caption': '',
         'xlabel':'test x',
         'ylabel': 'test y',
         'title': 'test title',
         'line_width': 2,
         'xmin': 0,
         'ymin': 0,
         'xmax': 40,
         'ymax': 1.0,
         'alpha_val': 0.25,
         'marker_val': None,
         'color_val': 'g',
         'target_plot': True,
         'legend_location': 'upper right',
         'color_target': 'r',
         'fifty_percent_plot': False,
         'color_fifty_percent': 'r'
         }
    t.plot_overall_chart(**d)

def test_plot_overall_unfilled_vacancies(mgmt_data):
    t = Mod_Stoch_FBPH(**mgmt_data)
    d = {'plottype': 'unfilled vacancies',
         'number_of_runs': 10,
         'target': 0.25,
         'caption': '',
         'xlabel':'test x',
         'ylabel': 'test y',
         'title': 'test title',
         'line_width': 2,
         'xmin': 0,
         'ymin': 0,
         'xmax': 40,
         'ymax': 10,
         'alpha_val': 0.25,
         'marker_val': None,
         'color_val': 'g',
         'target_plot': False,
         'legend_location': 'upper right',
         'color_target': 'r',
         'fifty_percent_plot': False,
         'color_fifty_percent': 'r'
         }
    t.plot_overall_chart(**d)


def test_plot_overall_department_size(mgmt_data):
    t = Mod_Stoch_FBPH(**mgmt_data)
    d = {'plottype': 'department size',
         'number_of_runs': 10,
         'target': 0.25,
         'caption': '',
         'xlabel':'test x',
         'ylabel': 'test y',
         'title': 'test title',
         'line_width': 2,
         'xmin': 0,
         'ymin': 0,
         'xmax': 40,
         'ymax': 100,
         'alpha_val': 0.25,
         'marker_val': None,
         'color_val': 'g',
         'target_plot': True,
         'legend_location': 'upper right',
         'color_target': 'r',
         'fifty_percent_plot': False,
         'color_fifty_percent': 'r',
         'target_plot_line_style': '--',
         'fifty__percent_line_style': '-.'
         }
    t.plot_overall_chart(**d)

def test_plot_bylevel_percentage(mgmt_data):
    t = Mod_Stoch_FBPH(**mgmt_data)
    d = {'plottype' : 'gender number',
         'number_of_runs': 20,
         'target' : 0.25,
         'caption' : '',
         'xlabel_f1': 'label f1',
         'ylabel_f1': 'range f1',
         'xlabel_f2': 'label f2',
         'ylabel_f2': 'range f2',
         'xlabel_f3': 'label f3',
         'ylabel_f3': 'range f3',
         'xlabel_m1':'label m1',
         'ylabel_m1':'range m1',
         'xlabel_m2':'label m2',
         'ylabel_m2':'range m2',
         'xlabel_m3':'label m3',
         'ylabel_m3':'range m3',
         'group_title': 'group title',
         'title_f1':'title f1',
         'title_f2':'title f2',
         'title_f3':'title f3',
         'title_m1':'title m1',
         'title_m2':'title m2',
         'title_m3':'title m3',
         'line_width': 2,
         'xmin_f1':0,
         'ymin_f1':0,
         'xmax_f1':40,
         'ymax_f1':15,
         'xmin_f2':0,
         'ymin_f2':0,
         'xmax_f2':40,
         'ymax_f2':5,
         'xmin_f3':0,
         'ymin_f3':0,
         'xmax_f3':40,
         'ymax_f3':8,
         'xmin_m1':0,
         'ymin_m1':0,
         'xmax_m1':40,
         'ymax_m1':20,
         'xmin_m2':0,
         'ymin_m2':0,
         'xmax_m2':40,
         'ymax_m2':25,
         'xmin_m3':0,
         'ymin_m3':0,
         'xmax_m3':40,
         'ymax_m3':60,
         'legend_location':'upper right',
         'model_legend_label':'model',
         'transparency': 0.25,
         'marker_shape': None,
         'linecolor' : 'g',
         'target_plot' : False,
         'target_color' : 'r',
         'target_plot_line_style' : '--',
         'target_plot_linewidth' : 2,
         'target_plot_legend_label' : 'target',
         'percent_line_plot' : False,
         'percent_line_value': 0.5,
         'color_percent_line':'r',
         'percent_line_style':'-.',
         'percent_linewidth':2,
         'percent_legend_label':'percent'}

    t.plot_level_chart(**d)


def test_comparison_plot_overall(mgmt_data):

    modlist = list([Mod_Stoch_FBHP(**mgmt_data),
                    Mod_Stoch_FBPH(**mgmt_data)])
    c = Comparison(modlist)
    plot_settings = {'plottype': 'male female numbers',
            'number_of_runs': 10,  # number simulations to average over
            'target': 0.25,  # target percentage of women in the department
            'caption': '',
            # Main plot settings
            'xlabel':'Years',
            'ylabel': 'Probability of Achieving Target Proportion',
            'title': 'Figure 4.1.1b: Probability of Achieving 25% Women, Model 1 (Hire-Promote)',
            'line_width': 2,
            'xmin': 0,
            'ymin': 0,
            'xmax': 40,
            'ymax': 100,
            'transparency': [0.25,0.25],
            'marker_shape': [None,None],
            'linecolor': ['g','b'],
            'model_legend_label': ['Female 1', 'Female 2'],
            'legend_location': 'upper right',

            # Optional Settings
            # Target value plot settings
            'target_plot': False,
            'color_target': 'r',
            'color_percent_line': 'r',
            'target_plot_line_style': '--',
            'target_plot_linewidth': 2,
            'target_plot_legend_label': 'Target percentage',

            # Percent plot settings
            'percent_line_plot': False,
            'percent_line_value': 0.5,
            'percent_line_style': '-.',
            'percent_linewidth': 2,
            'percent_legend_label': 'Reference Line',

            # Male Female numbers plot settings
            'male_female_numbers_plot': True,
            'mf_male_color': ['k','m'],
            'mf_target_color': 'r',
            'mf_male_label': ['Male 1', 'Male 2'],
            'mf_target_label': ['Target 1','Target 2'],
            'mf_male_linestyle': '-',
            'mf_target_linestyle': ['-','-.'],
            'mf_male_linewidth':2,
            'mf_target_linewidth': 2
            }
    c.plot_comparision_overall_chart(**plot_settings)

def test_comparision_duration(mgmt_data):
    modlist = list([Mod_Stoch_FBHP(**mgmt_data),
                    Mod_Stoch_FBPH(**mgmt_data)])
    c = Comparison(modlist)
    print(c.testfunc())
