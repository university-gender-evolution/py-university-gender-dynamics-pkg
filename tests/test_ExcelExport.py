import pytest
from pyugend.Mod_Stoch_FBHP import Mod_Stoch_FBHP
from pyugend.Mod_Stoch_FBPH import Mod_Stoch_FBPH
from pyugend.ModelGenderDiversity import Model3GenderDiversity
from pyugend.Comparison import Comparison

@pytest.mark.usefixtures('mgmt_data')

@pytest.mark.usefixtures('mock_data')


def test_excel_export_ph(mgmt_data):
    modlist = list([Model3GenderDiversity(**mgmt_data)])
    modlist[0].init_default_hiring_rate()
    c = Comparison(modlist)
    c.export_model_run('new model baseline management', 'new model baseline '
                                                       'management', 3)

# def test_excel_export_hp(mgmt_data):
#
#     modlist = list([Mod_Stoch_FBHP(**mgmt_data)])
#     c = Comparison(modlist)
#     c.export_model_run('model baseline HP management', 'model baseline HP '
#                                                        'management', 100)
