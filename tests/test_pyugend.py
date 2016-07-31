import pytest
from pyugend.Models import Base_model
from pyugend.Mod_Stoch_VSHP import Mod_Stoch_VSHP
from pyugend.Mod_Stoch_FSHP import Mod_Stoch_FSHP
from pyugend.Mod_Stoch_FSPH import Mod_Stoch_FSPH
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
    t = Mod_Stoch_VSHP(**mock_data)
    assert (isinstance(t.run_multiple(10), int))


def test_base_model_multiple_runs_persistent_state(mock_data):
    t = Mod_Stoch_VSHP(**mock_data)
    t.run_multiple(10)
    assert (isinstance(t.mean_matrix, np.ndarray))


def test_base_model_parameter_sweep(mock_data):
    t = Mod_Stoch_FSHP(**mock_data)
    v = t.run_parameter_sweep(12, 'female_pp_1', 0.1, 0.3, 2)
    assert (isinstance(v, int))


def test_base_model_plot_multiple_runs_detail(mock_data):
    t = Mod_Stoch_FBHP(**mock_data)
    t.run_multiple(10)
    t.plot_multiple_runs_detail(10,'Model 1: Hire-Promote', 5)

def test_base_model_plot_multiple_runs_percentage(mock_data):
    t = Mod_Stoch_FBHP(**mock_data)
    t.run_multiple(10)
    t.plot_multiple_runs_detail_percentage(0.25,100)

#
def test_base_model_multiple_runs_gender_prop(mock_data):
     t = Mod_Stoch_FBHP(**mock_data)
     t.run_multiple(100)
     print(t.pct_female_matrix)


def test_basic_stochastic_model(mock_data):
    assert (isinstance(Mod_Stoch_VSHP(**mock_data),
                       Mod_Stoch_VSHP))


def test_basic_stochastic_model_run(mock_data):
    t = Mod_Stoch_VSHP(**mock_data).run_model()
    assert isinstance(t, np.recarray)


def test_basic_stochastic_model_run_with_saved_data(mock_data):
    t = Mod_Stoch_VSHP(**mock_data)
    t.run_model()
    assert isinstance(t.run, np.recarray)


def test_basic_stochastic_model_promotion_probability_recovery(mock_data):
    t = Mod_Stoch_VSHP(**mock_data)
    assert (t.female_promotion_probability_2 == 0.188)


def test_replication_model(mock_data):
    t = Replication_model(**mock_data)
    t.run_model()
    assert (isinstance(t.run, np.ndarray))


# def test_base_model_multiple_runs_gender_prop(mock_data):
#      t = Replication_model(**mock_data)
#      t.run_multiple(10)
#      t.plot_multiple_runs_gender_prop()

def test_excel_export(mock_data):
    t = Mod_Stoch_FBPH(**mock_data)
    t.export_model_run(4)


def test_stochastic_model_with_hiring_first(mock_data):
    t = Mod_Stoch_FSPH(**mock_data)
    t.run_model()
    assert (isinstance(t.run, np.ndarray))


def test_stochastic_model_with_hiring_first_multiple(mock_data):
    t = Mod_Stoch_FBPH(**mock_data)
    t.run_multiple(10)
    assert (isinstance(t.mean_matrix, np.ndarray))


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
    modlist = list([Replication_model(**mock_data),
                    Mod_Stoch_FBHP(**mock_data),
                    Mod_Stoch_FBPH(**mock_data)])
    c = Comparison(modlist)
    c.plot_parameter_sweep_gender_proportion(10, 'female_promotion_probability_2', 0.1, 0.5, 8)


def test_comparison_model_param_sweep_detail(mock_data):
    modlist = list([Replication_model(**mock_data),
                    Mod_Stoch_VSHP(**mock_data),
                    Mod_Stoch_FSPH(**mock_data)])
    c = Comparison(modlist)
    c.plot_parameter_sweep_detail(10, 'female_promotion_probability_2', 0.1, 0.5, 8)


def test_base_model_probability_calc_detail_array(mock_data):
    t = Mod_Stoch_FSPH(**mock_data)
    res = t.run_probability_analysis_parameter_sweep_gender_detail(10,
                                                                   'female_promotion_probability_2',
                                                                   'm2', 0.1, 0.8, 8, 150)
    print(res)
    assert (isinstance(res, pd.DataFrame))


# def test_base_model_probability_calc_plot(mock_data):
#      t = Mod_Stoch_FSPH(**mock_data)
#      t.plot_empirical_probability_group_detail(10,
#                                             'female_promotion_probability_2',
#                                             'm2', 0.1, 0.8, 8, 150)
#
# def test_comparison_empirical_probability_detail_plot(mock_data):
#     modlist = list([Replication_model(**mock_data),
#                     Mod_Stoch_VSHP(**mock_data),
#                     Mod_Stoch_FSPH(**mock_data)])
#     c = Comparison(modlist)
#     c.plot_comparison_empirical_probability_detail(10,
#                                                    'female_promotion_probability_2',
#                                                    'm2', 0.1, 0.8, 20, 150)


def test_plot_dept_size_over_time_shrinking(mock_data):
    t = Mod_Stoch_FSPH(**mock_data)
    t.plot_department_size_over_time_multiple_runs(10, 'Dept Size Shrinking '
                                                       'Model', 'Years',
                                                   'Department Size')

def test_plot_dept_size_over_time_banded(mock_data):
    t = Mod_Stoch_FBPH(**mock_data)
    t.plot_department_size_over_time_multiple_runs(10, 'Dept Size Banded Model',
                                                   'Years',
                                                   'Department Size')

# def test_plot_comparision_department_size(mock_data):
#     modlist = list([Mod_Stoch_FSHP(**mock_data),
#                 Mod_Stoch_VSHP(**mock_data),
#                 Mod_Stoch_FSPH(**mock_data)])
#     c = Comparison(modlist)
#     c.plot_comparison_department_size()

def test_multiple_runs_created_res_array(mock_data):
    t = Mod_Stoch_FSPH(**mock_data)
    t.run_multiple(10)
    assert hasattr(t, 'mean_matrix')


def test_plot_empirical_probability_gender_proportion(mock_data):

    t = Mod_Stoch_FBPH(**mock_data)
    t.plot_empirical_probability_gender_proportion(100, 0.19, 'title',
                                                   'xlabel','ylabel', '')


def test_plot_comparision_empirical_probability_gender_proportion(mock_data):
    modlist = list([Mod_Stoch_VSHP(**mock_data),
                    Mod_Stoch_FSHP(**mock_data),
                    Mod_Stoch_FSPH(**mock_data)])
    c = Comparison(modlist)
    c.plot_comparison_empirical_probability_gender_proportion(100, 0.19)


def test_FBPH_model_run(mock_data):
    t = Mod_Stoch_FBPH(**mock_data)
    t.run_model()
    assert (isinstance(t.res, np.ndarray))

def test_FBPH_plot_dept_size(mock_data):
    t = Mod_Stoch_FBPH(**mock_data)
    t.plot_department_size_over_time_multiple_runs(300)

def test_FBHP_model_run(mgmt_data):
    t = Mod_Stoch_FBHP(**mgmt_data)
    t.run_model()
    assert (isinstance(t.res, np.ndarray))

def test_FBHP_plot_dept_size(mock_data):
    t = Mod_Stoch_FBHP(**mock_data)
    t.plot_department_size_over_time_multiple_runs(20)

def test_FBPH_plot_dept_size(mock_data):
    t = Mod_Stoch_FBPH(**mock_data)
    t.plot_department_size_over_time_multiple_runs(20, 'Dept Size Banded '
                                                        'Model',
                                                   'Years',
                                                   'Department Size')

def test_FBHP_plot_dept_size(mock_data):
    t = Mod_Stoch_FBHP(**mock_data)
    t.plot_department_size_over_time_multiple_runs(20, 'Dept Size Banded '
                                                        'Model',
                                                   'Years',
                                                   'Department Size')

def test_FBPH_plot_unfilled(mock_data):
    t = Mod_Stoch_FBPH(**mock_data)
    t.plot_unfilled_vacancies_over_time_multiple_runs(20, 'Dept Size Banded Model',
                                                   'Years',
                                                   'unfilled vacancies')


def test_FBPH_plot_gender_proportion(mock_data):
    t = Mod_Stoch_FBPH(**mock_data)
    t.plot_multiple_runs_gender_prop('Title', 'years', '% women',0.3,
                                     'This is a test of the caption '
                                     'functionality',20 )

def test_FBPH_plot_pct_gender_detail(mock_data):
    t = Mod_Stoch_FBPH(**mock_data)
    t.plot_multiple_runs_detail_percentage(20, 'pct', 0.15, 'text for caption')

def test_plot_comparision_unfilled_vacancies(mock_data):
    modlist = list([Mod_Stoch_FBPH(**mock_data),
                    Mod_Stoch_FBHP(**mock_data)])
    c = Comparison(modlist)
    c.plot_comparison_spec_parameter('dept_size', 'department size',
                                     'year', 'department size')

def test_FBHP_plot_gend_proportion(mgmt_data):
    t = Mod_Stoch_FBHP(**mgmt_data)
    t.plot_multiple_runs_gender_prop('Gender Proportion', 'Year', 'Percentage Female', 0.2, 100)

def test_FBHP_plot_gend_detail_percentage(mgmt_data):
    t = Mod_Stoch_FBHP(**mgmt_data)
    t.plot_multiple_runs_detail_percentage(100,'Detail Percentages',0.20,'')

def test_FBHP_plot_male_female_numbers(mgmt_data):

    t = Mod_Stoch_FBHP(**mgmt_data)
    t.plot_male_female_total_numbers('Years', 'Numbers',
                                     'Number of Men and Women in a Department',
                                     'caption',
                                     0.30,
                                     10)


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

def test_plot_empirical_probability_analysis_by_level(mgmt_data):
    t = Mod_Stoch_FBPH(**mgmt_data)
    t.plot_multiple_runs_detail_percentage(0.25,10)


def test_plot_comparison_empirical_probability_proportion_by_level(mock_data):
    modlist = list([Mod_Stoch_FBHP(**mock_data),
                    Mod_Stoch_FBPH(**mock_data)])
    c = Comparison(modlist)
    c.plot_comparison_empirical_probability_proportion_by_level(0.25,100)
