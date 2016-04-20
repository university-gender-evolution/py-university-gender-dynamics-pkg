import pytest
from models.Models import Base_model
from models.Basic_stochastic_model import Basic_stochastic_model
from models.Stochastic_model_with_promotion import Stochastic_model_with_promotion
from models.Replication_model import Replication_model
from models.Stochastic_model_with_promotion_and_first_hiring import Stochastic_model_with_promotion_and_first_hiring
from comparisons.Comparison import Comparison
from models.Basic_stochastic_model_fixed_promotion import Basic_stochastic_model_fixed_promotion
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
             'max_threshold': 0.1,
             'prob_random_growth': 0.1,
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
    t = Basic_stochastic_model(**mock_data)
    assert (isinstance(t.run_multiple(10), int))


def test_base_model_multiple_runs_persistent_state(mock_data):
    t = Basic_stochastic_model(**mock_data)
    t.run_multiple(10)
    assert (isinstance(t.mean_matrix, np.ndarray))


def test_base_model_parameter_sweep(mock_data):
    t = Stochastic_model_with_promotion(**mock_data)
    v = t.run_parameter_sweep(12, 'female_pp_1', 0.1, 0.3, 2)
    assert (isinstance(v, int))


def test_base_model_plot_multiple_runs(mock_data):
    t = Basic_stochastic_model(**mock_data)
    t.run_multiple(10)
    t.plot_multiple_runs_detail()


#
# def test_base_model_multiple_runs_gender_prop(mock_data):
#      t = Basic_stochastic_model(**mock_data)
#      t.run_multiple(10)
#      t.plot_multiple_runs_gender_prop()



def test_basic_stochastic_model(mock_data):
    assert (isinstance(Basic_stochastic_model(**mock_data),
                       Basic_stochastic_model))


def test_basic_stochastic_model_run(mock_data):
    t = Basic_stochastic_model(**mock_data).run_model()
    assert isinstance(t, np.recarray)


def test_basic_stochastic_model_run_with_saved_data(mock_data):
    t = Basic_stochastic_model(**mock_data)
    t.run_model()
    assert isinstance(t.run, np.recarray)


def test_basic_stochastic_model_promotion_probability_recovery(mock_data):
    t = Basic_stochastic_model(**mock_data)
    assert (t.female_promotion_probability_2 == 0.188)


def test_replication_model(mock_data):
    t = Replication_model(**mock_data)
    t.run_model()
    assert (isinstance(t.run, np.ndarray))


# def test_base_model_multiple_runs_gender_prop(mock_data):
#      t = Replication_model(**mock_data)
#      t.run_multiple(10)
#      t.plot_multiple_runs_gender_prop()

# def test_excel_export(mock_data):
#     t = Basic_stochastic_model(**mock_data)
#     t.export_model_run()
#     assert(isinstance(t,Basic_stochastic_model))

def test_stochastic_model_with_hiring_first(mock_data):
    t = Stochastic_model_with_promotion_and_first_hiring(**mock_data)
    t.run_model()
    assert (isinstance(t.run, np.ndarray))


def test_stochastic_model_with_hiring_first_multiple(mock_data):
    t = Stochastic_model_with_promotion_and_first_hiring(**mock_data)
    t.run_multiple(10)
    assert (isinstance(t.mean_matrix, np.ndarray))


def test_comparison_model_plot_detail(mock_data):
    modlist = list([Replication_model(**mock_data),
                    Basic_stochastic_model(**mock_data),
                    Stochastic_model_with_promotion_and_first_hiring(**mock_data)])
    c = Comparison(modlist)
    c.plot_comparison_detail(10)


def test_comparison_model_param_sweep(mock_data):
    modlist = list([Replication_model(**mock_data),
                    Basic_stochastic_model(**mock_data),
                    Stochastic_model_with_promotion_and_first_hiring(**mock_data)])
    c = Comparison(modlist)
    c.plot_parameter_sweep_gender_proportion(10, 'female_promotion_probability_2', 0.1, 0.5, 8)


def test_comparison_model_param_sweep_detail(mock_data):
    modlist = list([Replication_model(**mock_data),
                    Basic_stochastic_model(**mock_data),
                    Stochastic_model_with_promotion_and_first_hiring(**mock_data)])
    c = Comparison(modlist)
    c.plot_parameter_sweep_detail(10, 'female_promotion_probability_2', 0.1, 0.5, 8)


def test_base_model_probability_calc_detail_array(mock_data):
    t = Stochastic_model_with_promotion_and_first_hiring(**mock_data)
    res = t.run_probability_analysis_parameter_sweep_gender_detail(10,
                                                                   'female_promotion_probability_2',
                                                                   'm2', 0.1, 0.8, 8, 150)
    print(res)
    assert (isinstance(res, pd.DataFrame))


# def test_base_model_probability_calc_plot(mock_data):
#      t = Stochastic_model_with_promotion_and_first_hiring(**mock_data)
#      t.plot_empirical_probability_group_detail(10,
#                                             'female_promotion_probability_2',
#                                             'm2', 0.1, 0.8, 8, 150)
#
# def test_comparison_empirical_probability_detail_plot(mock_data):
#     modlist = list([Replication_model(**mock_data),
#                     Basic_stochastic_model(**mock_data),
#                     Stochastic_model_with_promotion_and_first_hiring(**mock_data)])
#     c = Comparison(modlist)
#     c.plot_comparison_empirical_probability_detail(10,
#                                                    'female_promotion_probability_2',
#                                                    'm2', 0.1, 0.8, 20, 150)


# def test_plot_dept_size_over_time(mock_data):
#     t = Stochastic_model_with_promotion_and_first_hiring(**mock_data)
#     t.plot_department_size_over_time_multiple_runs(10)


# def test_plot_comparision_department_size(mock_data):
#     modlist = list([Basic_stochastic_model_fixed_promotion(**mock_data),
#                 Basic_stochastic_model(**mock_data),
#                 Stochastic_model_with_promotion_and_first_hiring(**mock_data)])
#     c = Comparison(modlist)
#     c.plot_comparison_department_size()

def test_multiple_runs_created_res_array(mock_data):
    t = Stochastic_model_with_promotion_and_first_hiring(**mock_data)
    t.run_multiple(10)
    assert hasattr(t, 'probability_matrix')
#
# def test_plot_empirical_probability_gender_proportion(mock_data):
#     t = Stochastic_model_with_promotion_and_first_hiring(**mock_data)
#     t.plot_empirical_probability_gender_proportion(100, 0.19)


def test_plot_comparision_empirical_probability_gender_proportion(mock_data):
    modlist = list([Basic_stochastic_model(**mock_data),
                    Basic_stochastic_model_fixed_promotion(**mock_data),
                    Stochastic_model_with_promotion_and_first_hiring(**mock_data)])
    c = Comparison(modlist)
    c.plot_comparison_empirical_probability_gender_proportion(100, 0.19)


def test_basic_stochastic_with_random_dept_growth(mock_data):
    t = Basic_stochastic_model(**mock_data)
    assert (t.max_threshold, 0.1)

def test_basic_stochastic_with_random_dept_growth(mock_data):
    t = Basic_stochastic_model(**mock_data)
    assert (t.max_threshold, 0.1)