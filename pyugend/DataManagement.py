

from .abcDepartmentData import abcDepartmentData
import pandas as pd



class DataManagement(abcDepartmentData):

    def __init__(self):
        super().__init__()
        self.data = self.load_data()

    def load_data(self):
        a = pd.read_csv('/media/krishnab/jaimini1/development/gender_equity/university_gender_dynamics/py_package_ugend/pyugend/pyugend/data/mgmt_data.csv')
        return a

    def get_data(self):
        return self.data

    def get_field(self, field):
        return self.data.loc[:, field].tolist()

