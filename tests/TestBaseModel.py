import pytest
from pyugend.pyugend.Models import Base_model
from pyugend.pyugend.Mod_Stoch_FBHP import Mod_Stoch_FBHP
from pyugend.pyugend.Mod_Stoch_FBPH import Mod_Stoch_FBPH
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

    def test_base_model_multiple_runs_persistent_state(self,mgmt_data):
        t = Mod_Stoch_FBHP(**mgmt_data)
        t.run_multiple(5)
        assert (isinstance(t.results_matrix, pd.DataFrame))
