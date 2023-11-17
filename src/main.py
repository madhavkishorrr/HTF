from config import options
from preprocessing import get_data
from find_hyperparms import objective
from experiment import sliding_pred
import os
import optuna
import json
from metrics import *
import pandas as pd
import datetime
from plot_figs import plot_scores,plot_actual_vs_predicted,plot_error_box,plot_lambs_true

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    #datapath
    data_dir = options["data_path"]
    daily_cases,cumulative_cases,states = get_data(data_dir)
    if options["data"] == "DAILY":
        data=daily_cases
    if options["data"] == "CUM":
        data=cumulative_cases
    
    if options["find_params"] is True:
        study = optuna.create_study()
        study.optimize(objective(data,lam=0.5), n_trials=250,n_jobs=-1)
        params = study.best_params
        # Save the dictionary to a JSON file
        with open(f"{options['data']}.json", "w") as f:
            json.dump(study.best_params, f)

   
    if options["find_params"] is False:
        if not os.path.exists(f"{options['data']}.json"):
            raise Exception("Please run Hyperparameter optimization atleast once for this data")
        if  os.path.exists(f"{options['data']}.json"):
            # Load the dictionary from a JSON file
            with open(f"{options['data']}.json", "r") as f:
                params = json.load(f)
    

    #Predict
    t,y,exps,lambdas = sliding_pred(data,**params)


    mse,rmse = mse_rmse(data.iloc[:,t],y)
    mae,mape = mae_mape(data.iloc[:,t],y)
    r2 = r_square(data.iloc[:,t],y)
    scores =  pd.DataFrame({"r2": r2, "mae": mae,"mape":mape ,"mse":mse,"rmse":rmse,})

        # Set some formatting options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.precision', 2)

    # Print and save the dataframe
    print(scores.to_string())
    os.makedirs(options["results_path"],exist_ok=True)
    run_dir = os.path.join( options["results_path"], f"run_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" )
    os.makedirs( run_dir )
    #Save Metrics
    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(metrics_dir)
    scores.to_csv(os.path.join(metrics_dir , f"{options['data']}.csv"))

    #plot figs for run

    figs_dir = os.path.join(run_dir,"figures")
    os.makedirs(figs_dir)
    plot_scores(scores,figs_dir)
    plot_actual_vs_predicted(data,y,t,figs_dir,options['data'])
    plot_error_box(data,y,t,figs_dir)
    plot_lambs_true(t,data.iloc[:,t],lambdas,figs_dir,st="RJ")