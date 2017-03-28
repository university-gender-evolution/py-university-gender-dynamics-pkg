import pytest
from pyugend.pyugend.Models import Base_model
from pyugend.pyugend.Mod_Stoch_FBHP import Mod_Stoch_FBHP
from pyugend.pyugend.Mod_Stoch_FBPH import Mod_Stoch_FBPH
import numpy as np
import pandas as pd

@pytest.mark.usefixtures('mgmt_data')

@pytest.mark.usefixtures('mock_data')


class TestExcelExport:

    def test_excel_export(self,mgmt_data):
        t = Mod_Stoch_FBPH(**mgmt_data)
        t.export_model_run('testexport', 'model test', 10)
