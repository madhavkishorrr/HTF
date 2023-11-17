import numpy as np
from pydmd import HODMD
import matplotlib.pyplot as plt
import pandas as pd
import numpy.matlib as mb

class Experiment():
    
    def __init__(self, X,
                 train_params = {"start_day":0, "train_days":40, "pred_days":5},
                 HODMD_params = { "svd_rank":0, "tlsq_rank":0, "exact":False, "opt":False,
                 "rescale_mode":None, "forward_backward":False, "d":35,
                 "sorted_eigs":False, "reconstruction_method":"first",
                 "svd_rank_extra":0},
                row_mask=None):
        self.start_day = train_params["start_day"]
        self.train_days = train_params["train_days"]
        self.pred_days = train_params["pred_days"]
        self.X = X
        self.DMD = HODMD(**HODMD_params)
        
        self.row_mask = row_mask
        self.train_data ,self.ground_truth = Experiment.train_test_split(self.X,self.train_days,self.pred_days,self.start_day,self.row_mask)
        self.metrics = {}

    @staticmethod
    def train_test_split(df,train_days,pred_days,start_day,row_mask=None):
        if row_mask==None:
            train = df.loc[:,start_day:start_day+train_days-1]
            test = df.loc[:, start_day+train_days: start_day+train_days+pred_days-1]
        else:
            try:
                train = df.loc[row_mask,start_day:start_day+train_days-1]
                test = df.loc[row_mask, start_day+train_days: start_day+train_days+pred_days-1]
            except:
                raise Exception("mask should be compatible with the indices of rows")
    
        return train,test

    def fit(self):

        snapshot = self.train_data.to_numpy()
        self.DMD = self.DMD.fit(snapshot)
        #print(self.DMD.eigs.shape)
        #self.DMD.original_time['t0'] = self.DMD.dmd_time['t0'] = self.start_day
        self.DMD.original_time['tend'] = self.DMD.dmd_time['tend'] = (self.train_days-1)
        self.reconstructed_data = self.DMD.reconstructed_data.real
        
    def pred(self):
        #self.DMD.dmd_time['t0'] = self.start_day
      
        self.DMD.dmd_time['tend'] = ( self.train_days + self.pred_days - 1)
        
        self.pred_data =  self.DMD.reconstructed_data.real[:, self.train_days:(  self.train_days + self.pred_days)]
        #self.pred_data =  self.DMD.reconstructed_data.real
        
     
       # self.MAPE();
        return self.pred_data

    def MAPE(self):
        gt = self.ground_truth.to_numpy() 
        self.metrics["MAPE"] = np.abs((gt-self.pred_data)/(gt+0.00000001)).mean(axis=1)   
        return self.metrics["MAPE"]

    @staticmethod
    def MAPE_full(ground_truth,pred_data):
        gt =  ground_truth.to_numpy() 
        return np.abs((gt- pred_data)/(gt+0.00000001)).mean(axis=1)   

    def plot_pred(self,row_num=1,ax=None):
        #plt.plot(np.arange(self.start_day+self.train_days,(self.start_day + self.train_days + self.pred_days)),self.pred_data[row_num,:])

        #plt.plot(np.arange(self.start_day,self.start_day+self.train_days ),self.reconstructed_data[row_num,:])


        gt_time_steps = np.arange(self.start_day+self.train_days ,self.start_day+self.train_days +self.pred_days )
        train_time_steps = np.arange(self.start_day,self.start_day+self.train_days  )

        #plt.plot(gt_time_steps,self.ground_truth.iloc[row_num,:],'g')
    
        #plt.plot(train_time_steps, self.train_data.iloc[row_num,:],'r')
       # plt.plot(train_time_steps, self.reconstructed_data[row_num,:],'b')

        plt.plot(gt_time_steps , self.pred_data[row_num,:])
      
        #plt.legend([ "Predicted" , "Reconstructed", "Ground Truth", "Train Data"])
        
       # plt.legend(["GT","Train","Pred"])


def sliding_pred(X,train_days=60,pred_days=30,d=20):
    start_day=0
    train_params = {"start_day":start_day, "train_days":train_days, "pred_days":pred_days}
    HODMD_params = { "svd_rank":X.shape[0], "tlsq_rank":0, "exact":False, "opt":True,
                    "rescale_mode":None, "forward_backward":False, "d":d,
                    "sorted_eigs":False, "reconstruction_method":"first",
                    "svd_rank_extra":X.shape[0]}
    exps = []
    preds = []
    exp = Experiment(X,train_params=train_params,HODMD_params=HODMD_params);
    exp.fit()
    preds = exp.pred()
    pred_timesteps = np.arange(train_params["start_day"]+train_params["train_days"],train_params["start_day"]+train_params["train_days"]+ train_params["pred_days"])
    lambdas = mb.repmat(exp.DMD.eigs.reshape(-1,1),1,train_params["pred_days"] )
    for epoch_start_day in range(train_params["start_day"], X.shape[1]-(train_params["pred_days"]+train_params["train_days"]), train_params["pred_days"] ):  
        train_params_tilde = {"start_day":epoch_start_day, "train_days":train_days, "pred_days":pred_days}
        pred_timesteps = np.concatenate( (pred_timesteps, np.arange(epoch_start_day+train_params["train_days"],epoch_start_day+train_params["train_days"]+ train_params["pred_days"])))
        exp = Experiment(X,train_params=train_params_tilde,HODMD_params=HODMD_params);
        exp.fit()
        preds = np.concatenate((preds, exp.pred()), axis=1)
        exps.append(exp)
        lambdas = np.concatenate((lambdas,   mb.repmat(exp.DMD.eigs.reshape(-1,1),1,train_params_tilde["pred_days"] )    ), axis=1 )

    return pred_timesteps,preds,exps,lambdas