import numpy as np
import sys

from paths import ML_Path
sys.path.append(ML_Path)

from mllib import MLUtilities
from mlalgos import Sequential
# import copy,pickle

import gc

class NeuralRatioEstimator(MLUtilities):
    """ Base class to construct neural ratio estimator using provided training sample. """ 
    def __init__(self,params={}):
        self.nparam = params.get('param_dim',None)
        self.ndata = params.get('data_dim',None)
        self.seed = params.get('seed',None)

        self.check_init()

        self.params_seq = {} # feed to Sequential

        self.rng = np.random.RandomState(seed=self.seed)

    def check_init(self):
        if self.nparam is None:
            raise Exception("Need to specify param_dim in NeuralRatioEstimator.")
        
        if self.ndata is None:
            raise Exception("Need to specify data_dim in NeuralRatioEstimator.")
        
        return
    
    # force this method to be defined explicitly 
    def simulator(self,theta):
        prnt_strng = "Need to define NeuralRatioEstimator.simulator with input theta (self.nparam,nsamp)"
        prnt_strng += " and output X_sim (self.ndata,nsamp)"
        raise NotImplementedError()

    # force this method to be defined explicitly 
    def prior(self,nsamp):
        prnt_strng = "Need to define NeuralRatioEstimator.prior with input nsamp"
        prnt_strng += " and output theta (self.nparam,nsamp)"
        raise NotImplementedError()

    def gen_train(self,theta):
        """ Generate complete training sample using provided theta samples drawn from prior p(theta).
            -- theta: parameter sample of shape (self.nparam,nsamp)
            Returns Xtheta [(self.ndata+self.nparam,2*nsamp)], Y [(1,2*nsamp)]
            organised so that Xtheta[:,:nsamp] combines input theta with output of self.simulator(theta), thus sampling p(x,theta),
            and Xtheta[:,nsamp:] is the same but with X and theta order both shuffled, thus sampling p(x)p(theta).
            Correspondingly, Y[0,:nsamp] = 1 and Y[0,nsamp:] = 0.
        """
        if theta.shape[0] != self.nparam:
            raise Exception("NeuralRatioEstimator.gen_train expected first axis of input of dimension {0:d}, got {1:d}"
                            .format(self.nparam,theta.shape[0]))

        nsamp = theta.shape[1]
        X = self.simulator(theta)
        
        if X.shape[0] != self.ndata:
            raise Exception("Expected output of NeuralRatioEstimator.simulator with first axis of dimension {0:d}, got {1:d}"
                            .format(self.ndata,X.shape[0]))
        if X.shape[1] != nsamp:
            raise Exception("Expected output of NeuralRatioEstimator.simulator with second axis of dimension {0:d}, got {1:d}"
                            .format(nsamp,X.shape[1]))

        Xtheta_orig = np.concatenate((X,theta),axis=0)

        # shuffle
        ind_shuff_theta = np.arange(nsamp)
        self.rng.shuffle(ind_shuff_theta)
        ind_shuff_X = np.arange(nsamp)
        self.rng.shuffle(ind_shuff_X)        
        Xtheta_shuff = np.concatenate((X[:,ind_shuff_X],theta[:,ind_shuff_theta]),axis=0)

        Xtheta = np.concatenate((Xtheta_orig,Xtheta_shuff),axis=1) # note original first, shuffled second
        Y = np.zeros((1,Xtheta.shape[1]))
        Y[0,:nsamp] = 1.0
        
        return Xtheta,Y


    def train(self,params={}):
        return 
