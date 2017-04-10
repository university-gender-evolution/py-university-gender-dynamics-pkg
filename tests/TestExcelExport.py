import pytest
from pyugend.Mod_Stoch_FBHP import Mod_Stoch_FBHP
from pyugend.Mod_Stoch_FBPH import Mod_Stoch_FBPH


@pytest.mark.usefixtures('mgmt_data')

@pytest.mark.usefixtures('mock_data')


class TestExcelExport:

    def test_excel_export_ph(self, mgmt_data):
        t = Mod_Stoch_FBPH(**mgmt_data)
        t.export_model_run('model baseline PH management', 'model baseline PH management', 100)

    def test_excel_export_hp(self, mgmt_data):
        t = Mod_Stoch_FBHP(**mgmt_data)
        t.export_model_run('model baseline HP management', 'model baseline HP management', 100)
