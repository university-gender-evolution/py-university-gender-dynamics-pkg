import pytest
from pyugend.Models import Base_model
from pyugend.Mod_Stoch_FBHP import Mod_Stoch_FBHP
from pyugend.Mod_Stoch_FBPH import Mod_Stoch_FBPH
import numpy as np
import pandas as pd

@pytest.mark.usefixtures('mgmt_data')

@pytest.mark.usefixtures('mock_data')

class TestBaseModel:


    def test_Base_model(self, mgmt_data):
        assert isinstance(Base_model(**mgmt_data), Base_model)

    def test_base_model_run(self, mgmt_data):
        t = Base_model(**mgmt_data)
        t.run_model()
        assert (isinstance(t.res, np.ndarray))

    def test_base_model_persistence(self, mgmt_data):
        t = Base_model(**mgmt_data)
        assert (t.nf1 == 3)

    def test_base_model_multiple_runs(self, mgmt_data):
        t = Mod_Stoch_FBPH(**mgmt_data)
        t.run_multiple(5)
        assert (hasattr(t, 'res_array'))

    def test_base_model_multiple_runs_persistent_state(self, mgmt_data):
        t = Mod_Stoch_FBHP(**mgmt_data)
        t.run_multiple(5)
        assert (isinstance(t.results_matrix, pd.DataFrame))

    def test_base_model_parameter_sweep(self, mgmt_data):
        t = Mod_Stoch_FBHP(**mgmt_data)
        t.run_parameter_sweep(10,'bf1',0.05, 0.6,5)
        assert (hasattr(t, 'parameter_sweep_results'))


    #TODO must fix this test case
    @pytest.mark.skip(reason="no way of currently testing this")
    def test_base_model_probability_calc_detail_array(self, mgmt_data):
        t = Mod_Stoch_FBPH(**mgmt_data)
        res = t.run_probability_analysis_parameter_sweep_gender_detail(10,
            'female_promotion_probability_2','m2', 0.1,0.8, 8, 150)
        assert (isinstance(res, pd.DataFrame))

