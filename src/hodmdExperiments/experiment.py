from pydmd import HODMD

class Experiment():
    
    def __init__(self, X,
                 train_params = {"start_day":0, "train_days":40, "pred_days":5},
                 HODMD_params = { "svd_rank":0, "tlsq_rank":0, "exact":False, "opt":False,
                 "rescale_mode":None, "forward_backward":False, "d":1,
                 "sorted_eigs":False, "reconstruction_method":"first",
                 "svd_rank_extra":0},
                countries_of_intrest=None):
        self.start_day = train_params["start_day"]
        self.train_days = train_params["train_days"]
        self.pred_days = train_params["pred_days"]
        self.X = X
        self.DMD = HODMD(**HODMD_params)
        
        self.countries_of_intrest = countries_of_intrest
        
        if countries_of_intrest is not None:
        
            self.train_data = self.X.loc[countries_of_intrest, self.start_day:self.start_day+self.train_days-1]
            
        if countries_of_intrest is None:
            
            self.train_data = self.X.loc[:, self.start_day:self.start_day+self.train_days-1]