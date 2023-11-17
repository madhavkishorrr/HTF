
import optuna
from optuna import Trial
from experiment import sliding_pred,Experiment
from metrics import mae_mape
import pandas as pd

def objective(y_true:pd.DataFrame,lam=0.5):
    def df_objective(trial:Trial):
        train_days = trial.suggest_int("train_days",20,150)
        pred_days = trial.suggest_int("pred_days",15,60)
        d = trial.suggest_int("d",3,train_days-2)
        t,y,exps,lambdas = sliding_pred(y_true,train_days=train_days,pred_days=pred_days,d=d)
        mae,mape = mae_mape(y_true.iloc[:, t],y)
        score = ((1-lam)*Experiment.MAPE_full(y_true.iloc[:, t],y).sum()) + lam*(train_days/pred_days)
        return score
    return df_objective


