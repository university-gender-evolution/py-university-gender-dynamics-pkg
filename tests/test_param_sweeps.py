import pytest


@pytest.mark.usefixtures('mgmt_data')
class Test_param_sweeps:


    def test_test_Base_model(self):
        assert 1


    def test_Base_model(self, mgmt_data):
        from pyugend.pyugend.Models import Base_model
        t = Base_model(**mgmt_data)
        assert isinstance(t, Base_model)
