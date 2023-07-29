
from hodmdExperiments.experiment import Experiment

import pytest
import pandas as pd

@pytest.fixture
def covid_data():
    df = pd.read_csv("./Dataset/CleanedData.csv",header=1)
    data = df.iloc[:,1:].T
    return data

def test_covid_data_dimensions(covid_data):
    #201 countries with 1032 days of cumulative count
    assert covid_data.shape == (201,1032)



class TestExperiment:
    
    def test_when_countries_of_intrest_is_none(self,covid_data):
        exp = Experiment(covid_data)
        assert exp.countries_of_intrest == None
        assert exp.train_data.shape == (201,exp.train_days)

    def test_when_countries_of_intrest_is_not_none(self,covid_data):
        coi =  ["China","India","Bangladesh", "Afghanistan","United Arab Emirates","United Kingdom"]
        exp = Experiment(covid_data, countries_of_intrest= coi)
        assert exp.train_data.shape == (len(coi), exp.train_days)



