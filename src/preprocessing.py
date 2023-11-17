import pandas as pd
from config import options
from experiment import sliding_pred


def get_data(data_path):
    df = pd.read_csv(data_path,header=1)
    df = df.fillna(method="ffill")
    df = df.drop("0",axis=1)
    daily_cases = df.ewm(span=60,axis=0).mean().T
    daily_cases.drop(["UT","DD","UN"],axis=0,inplace=True)
    cumulative_cases = daily_cases.cumsum(axis=1)
    states = daily_cases.T.columns

    return daily_cases,cumulative_cases,states

