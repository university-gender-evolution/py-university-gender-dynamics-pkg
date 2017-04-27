import pytest
from pyugend.Mod_Stoch_FBHP import Mod_Stoch_FBHP
from pyugend.Mod_Stoch_FBPH import Mod_Stoch_FBPH
from pyugend.Comparison import Comparison

@pytest.mark.usefixtures('mgmt_data')

@pytest.mark.usefixtures('mock_data')


def test_excel_export_ph(mgmt_data):
    modlist = list([Mod_Stoch_FBPH(**mgmt_data)])
    c = Comparison(modlist)
    c.export_model_run('model baseline PH management', 'model baseline PH '
                                                       'management', 100)

def test_excel_export_hp(mgmt_data):

    modlist = list([Mod_Stoch_FBHP(**mgmt_data)])
    c = Comparison(modlist)
    c.export_model_run('model baseline HP management', 'model baseline HP '
                                                       'management', 100)
