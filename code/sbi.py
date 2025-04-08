import numpy as np
import sys,os

from paths import ML_Path
sys.path.append(ML_Path)

from pathlib import Path
import copy,pickle

from mllib import MLUtilities,Utilities
from mlalgos import Sequential

import gc

class NeuralRatioEstimator(MLUtilities,Utilities):
    """ Base class to construct neural ratio estimator using provided training sample. """
    #############################
    def __init__(self,params={}):
        """ Neural ratio estimation using Sequential NN.
            params should be dictionary with a subset of following keys:
            -- params['data_dim']: int, input data dimension
            -- params['param_dim']: int, input model parameter dimension
            -- params['use_external']: boolean, whether or not to use externally defined training sample (default False).
                                       If False, then methods simulator() and prior() must be defined by user.
                                       If True, then these methods not used but training set should have correct format.
            -- params['Lh']: int, L >= 1, number of hidden layers
            -- params['n_hidden_layer']: list of Lh int, number of units in each hidden layer.
            -- params['hidden_atypes']: list of Lh str, activation type in each hidden layer 
                                        chosen from ['sigm','tanh','relu','lrelu','sm','lin'] or 'custom...'.
                                        If 'custom...', then also define dictionary params['custom_atypes']
            -- params['custom_atypes']: dictionary with keys matching 'custom...' entry in params['hidden_atypes']
                                        with items being activation module instances.
            -- params['standardize']: boolean, whether or not to standardize training data in train() (default True)
            -- params['adam']: boolean, whether or not to use adam in GD update (default True)
            -- params['lrelu_slope']: float in (-1,1), slope of leaky ReLU if used (default 1e-2).
            -- params['wt_decay']: float, weight decay coefficient (should be non-negative; default 0.0)
            -- params['decay_norm']: int, norm of weight decay coefficient, either 2 or 1 (default 2)
            -- params['reg_fun']: str, type of regularization.
                                  Accepted values ['bn','drop','none'] for batch-normalization, dropout or no reg, respectively.
                                  If 'drop', then value of 'p_drop' must be specified. Default 'none'.
            -- params['p_drop']: float between 0 and 1, drop probability.
                                 Only used if 'reg_fun' = 'drop'.
                                 Default value 0.5, but not clear if this is a good choice.
            -- params['seed']: int, random number seed.
            -- params['file_stem']: str, common stem for generating filenames for saving (should include full path).
            -- params['nreal']: int, number of realisations to average over (default 1).
            -- params['share_train_data']: bool, whether or not to share training data across multiple realisations (default True).
            -- params['parallel']: bool, whether or not to use parallel calculation in ensemble averaging (default False).
            -- params['verbose']: boolean, whether of not to print output (default True).
            -- params['logfile']: None or str, file into which to print output (default None, print to stdout)
        """
        self.params = params
        self.nparam = params.get('param_dim',None)
        self.ndata = params.get('data_dim',None)
        self.use_external = params.get('use_external',False)
        self.seed = params.get('seed',None)
        self.Lh = int(params.get('Lh',1))
        self.n_hidden_layer = params.get('n_hidden_layer',[1]) 
        self.hidden_atypes = params.get('hidden_atypes',['relu'])
        custom_atypes = params.get('custom_atypes',None) 
        adam = params.get('adam',True)
        lrelu_slope = params.get('lrelu_slope',1e-2)
        reg_fun = params.get('reg_fun','none')
        p_drop = params.get('p_drop',0.5)
        wt_decay = params.get('wt_decay',0.0)
        decay_norm = int(params.get('decay_norm',2))
        self.standardize = params.get('standardize',True)
        self.seed = params.get('seed',None)
        self.file_stem = params.get('file_stem','net')
        self.verbose = params.get('verbose',True)
        self.logfile = params.get('logfile',None)

        self.nreal = params.get('nreal',1)
        self.share_train_data = params.get('share_train_data',True)
        
        self.parallel = params.get('parallel',False)
        self.nproc = np.min([os.cpu_count(),self.nreal]) 

        # for consistency with self.save and self.load
        self.params['standardize'] = self.standardize 
        self.params['parallel'] = self.parallel 
        self.params['nreal'] = self.nreal
        self.params['share_train_data'] = self.share_train_data
        
        self.file_stems = {r:self.file_stem+'/r{0:d}'.format(r) for r in range(1,self.nreal+1)} if self.nreal > 1 else {1:self.file_stem}
        if self.standardize:
            self.params['X_mean'] = {}
            self.params['X_std'] = {}
            self.params['theta_mean'] = {}
            self.params['theta_std'] = {}
            self.X_mean = {}
            self.X_std = {}
            self.theta_mean = {}
            self.theta_std = {}
            
        self.check_init()

        ##########################
        # feed to Sequential
        self.params_seq = {}
        self.net = {}
        for r in range(1,self.nreal+1):
            self.params_seq[r] = {'data_dim':self.ndata+self.nparam,'L':self.Lh+1,
                                  'n_layer':self.n_hidden_layer+[1],'atypes':self.hidden_atypes+['sigm'],'custom_atypes':custom_atypes,
                                  'loss_type':'nll','neg_labels':False,'standardize':False,# note False
                                  'adam':adam,'lrelu_slope':lrelu_slope,'reg_fun':reg_fun,'p_drop':p_drop,
                                  'wt_decay':wt_decay,'decay_norm':decay_norm,'seed':self.seed,'file_stem':self.file_stems[r]+'/net',
                                  'verbose':self.verbose,'logfile':self.logfile} 
            self.net[r] = Sequential(params=self.params_seq[r])
            self.net[r].net_type = 'reg'
            self.net[r].modules[-1].net_type = 'reg'
        ##########################
        
        self.rng = np.random.RandomState(seed=self.seed)

        return
    #############################

    #############################
    def check_init(self):
        if self.nparam is None:
            raise Exception("Need to specify param_dim in NeuralRatioEstimator.")
        
        if self.ndata is None:
            raise Exception("Need to specify data_dim in NeuralRatioEstimator.")

        for r in range(1,self.nreal+1):
            Path(self.file_stems[r]).mkdir(parents=True, exist_ok=True)
        
        return
    #############################
    
    #############################
    # force these methods to be defined explicitly, unless external training set to be used
    def simulator(self,theta):
        if not self.use_external:
            prnt_strng = "Need to define NeuralRatioEstimator.simulator with input theta (self.nparam,nsamp)"
            prnt_strng += " and output X_sim (self.ndata,nsamp)"
            raise NotImplementedError()

    def prior(self,nsamp):
        if not self.use_external:
            prnt_strng = "Need to define NeuralRatioEstimator.prior with input nsamp"
            prnt_strng += " and output theta (self.nparam,nsamp)"
            raise NotImplementedError()
    #############################

    #############################
    def gen_train(self,theta_input,r):
        """ Generate complete training sample for one realisation using provided theta samples drawn from prior p(theta).
            -- theta_input: parameter sample of shape (self.nparam,nsamp)
            -- r: int >= 1, realisation number
            Returns Xtheta [(self.ndata+self.nparam,2*nsamp)], Y [(1,2*nsamp)]
            organised so that Xtheta[:,:nsamp] combines input theta with output of self.simulator(theta), thus sampling p(x,theta),
            and Xtheta[:,nsamp:] is the same but with X and theta order both shuffled, thus sampling p(x)p(theta).
            Correspondingly, Y[0,:nsamp] = 1 and Y[0,nsamp:] = 0.
        """
        theta = theta_input.copy() # may be modified below, so copy here
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

        if self.standardize:
            ##########################
            self.X_std[r] = np.std(X,axis=1)
            self.X_mean[r] = np.mean(X,axis=1)
            self.params['X_mean'][r] = self.X_mean[r]
            self.params['X_std'][r] = self.X_std[r]
            X = (X.T - self.X_mean[r]).T
            X = (X.T/(self.X_std[r] + 1e-15)).T
            
            self.theta_std[r] = np.std(theta,axis=1)
            self.theta_mean[r] = np.mean(theta,axis=1)
            self.params['theta_mean'][r] = self.theta_mean[r]
            self.params['theta_std'][r] = self.theta_std[r]
            theta = (theta.T - self.theta_mean[r]).T
            theta = (theta.T/(self.theta_std[r] + 1e-15)).T
            ##########################
            
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
    #############################


    #############################
    def check_sample(self,Xtheta,Y,nsamp):
        """ Simle utility to check user-defined training sample. """
        if Xtheta is None:
            prnt_str = "Since use_external = True, need Xtheta defined with shape = ({0:d},{1:d})".format(self.ndata+self.nparam,2*nsamp)
            raise Exception(prnt_str)
        if Xtheta.shape != (self.ndata+self.nparam,2*nsamp):
            prnt_str = "Need Xtheta.shape = ({0:d},{1:d}), got (".format(self.ndata+self.nparam,2*nsamp)
            prnt_str += ','.join([str(s) for s in Xtheta.shape])+")"
            raise Exception(prnt_str)
        if Y is None:
            prnt_str = "Since use_external = True, need Y defined with shape = (1,{0:d})".format(2*nsamp)
            raise Exception(prnt_str)
        if Y.shape != (self.ndata+self.nparam,2*nsamp):
            prnt_str = "Need Y.shape = ({0:d},{1:d}), got (".format(self.ndata+self.nparam,2*nsamp)
            prnt_str += ','.join([str(s) for s in Y.shape])+")"
            raise Exception(prnt_str)
        
        return
    #############################

    #############################
    def train(self,nsamp,params={}):
        """ NRE training.
            -- nsamp: int, number of samples to simulate.
            -- params: dictionary compatible with input to Sequential.train(). 
                       If self.use_external = True, then params should contain keys 'Xtheta' and 'Y' with values
                       being arrays of shape (self.ndata+self.nparam,2*nsamp*self.nreal) and (1,2*nsamp*self.nreal), respectively,
                       organised such that each of self.nreal chunks of length 2*nsamp along last axis 
                       are compatible with output of self.gen_train().
        """
        ##########################
        # setup sample
        if self.verbose:
            self.print_this('Sample setup...',self.logfile)
            
        if self.use_external:
            Xtheta_all = params.get('Xtheta',None)
            Y_all = params.get('Y',None)
            nsamp_all = nsamp if self.share_train_data else nsamp*self.nreal
            self.check_sample(Xtheta_all,Y_all,nsamp_all) 

        if self.share_train_data:
            if self.verbose:
                self.print_this('Sharing data between {0:d} realisations...'.format(self.nreal),self.logfile)
        else:
            if self.verbose:
                self.print_this('Using independent data for each of {0:d} realisations...'.format(self.nreal),self.logfile)
                
        if self.parallel:
            ##########################
            # continue setup sample
            Xtheta_r = []
            Y_r = []
            for r in range(1,self.nreal+1):
                if self.use_external:
                    Xtheta = Xtheta_all if self.share_train_data else Xtheta_all[:,(r-1)*2*nsamp:r*2*nsamp]
                    Y = Y_all if self.share_train_data else Y_all[:,(r-1)*2*nsamp:r*2*nsamp]
                else:
                    if (r == 1) | (not self.share_train_data):
                        theta = self.prior(nsamp)
                        Xtheta,Y = self.gen_train(theta,r)
                Xtheta_r.append(Xtheta)
                Y_r.append(Y)

            tasks = [(Xtheta_r[r],Y_r[r],params,copy.deepcopy(self.net[r+1])) for r in range(self.nreal)]

            del Xtheta_r,Y_r
            gc.collect()

            if self.verbose:
                self.print_this('Training {0:d} realisations with {1:d} processors...'.format(self.nreal,self.nproc),self.logfile)
            # train networks
            updated_net_dict = self.run_processes(tasks,self.queue_train,self.nproc)
            for r in range(1,self.nreal+1):
                self.net[r] = updated_net_dict[r]

            del tasks,updated_net_dict
            gc.collect()
            ##########################
        else:
            ##########################
            # continue setup sample
            for r in range(1,self.nreal+1):
                if self.verbose:
                    self.print_this('Training realisation {0:d} of {1:d}...'.format(r,self.nreal),self.logfile)
                if self.use_external:
                    Xtheta = Xtheta_all if self.share_train_data else Xtheta_all[:,(r-1)*2*nsamp:r*2*nsamp]
                    Y = Y_all if self.share_train_data else Y_all[:,(r-1)*2*nsamp:r*2*nsamp]
                else:
                    if (r == 1) | (not self.share_train_data):
                        theta = self.prior(nsamp)
                        Xtheta,Y = self.gen_train(theta,r)

                # train network
                self.net[r].train(Xtheta,Y,params=params)
            ##########################
        
        return 
    #############################

    #############################
    def queue_train(self,r,X,Y,params,net,mdict):
        """ Convenience function for use with MLUtilities.run_processes(). Expect r >= 0."""
        net.train(X,Y,params)
        mdict[r+1] = net
        return
    #############################

    #############################
    def predict(self,X,theta):
        """ Predict neural ratio r(X,theta) =  p(theta|x)/p(theta) = p(x|theta)/p(x).
            -- X: input data array of shape (self.ndata,nsamp) [typically nsamp=1 in this case] 
            -- theta: parameter array of shape (self.nparam,nsamp)
            Returns r(X,theta) of shape (nsamp,).
        """        
        ##########################
        ratios = np.zeros(self.nreal)
        weights = np.zeros(self.nreal)
        for r in range(1,self.nreal+1):
            X_use = X.copy()
            theta_use = theta.copy()
            if self.standardize:
                r_use = 1 if self.share_train_data else r
                X_use = (X_use.T - self.X_mean[r_use]).T
                X_use = (X_use.T/(self.X_std[r_use] + 1e-15)).T
                theta_use = (theta_use.T - self.theta_mean[r_use]).T
                theta_use = (theta_use.T/(self.theta_std[r_use] + 1e-15)).T

            Xtheta = np.concatenate((X_use,theta_use),axis=0)

            s = self.net[r].predict(Xtheta)
            ratios[r-1] = s/(1 - s + 1e-15)

            min_loss = self.net[r].val_loss[self.net[r].val_loss > 0.0].min() + 1e-15
            weights[r-1] = 1/min_loss
            
        weights /= weights.sum()

        ratio = np.sum(weights*ratios)
        ##########################
        
        return ratio        
    #############################

    #############################
    def save(self):
        """ Save current weights and setup params to file(s). """
        
        ##########################
        for r in range(1,self.nreal+1):
            self.net[r].save()
        ##########################
        
        with open(self.file_stem + '/params.pkl', 'wb') as f:
            pickle.dump(self.params,f)            
        return    
    #############################

    #############################
    # to be called after generating instance of NeuralRatioEstimator() with correct setup params,
    # e.g. after invoking self.save().
    def load(self):
        """ Load weights and setup params from file(s). """
        ##########################
        for r in range(1,self.nreal+1):
            self.net[r].load()
            self.net[r].load_loss_history()
        ##########################
        # self.net.load()
        
        with open(self.file_stem + '/params.pkl', 'rb') as f:
            self.params = pickle.load(f)
            
        self.nparam = self.params['param_dim']
        self.nreal = self.params['nreal']
        self.standardize = self.params['standardize']
        self.share_train_data = self.params['share_train_data']
        self.parallel = self.params['parallel']
        if self.standardize:
            ##########################
            for r in range(1,self.nreal+1):
                r_use = 1 if self.share_train_data else r
                self.X_std[r] = self.params['X_std'][r_use]
                self.X_mean[r] = self.params['X_mean'][r_use]
                self.theta_std[r] = self.params['theta_std'][r_use]
                self.theta_mean[r] = self.params['theta_mean'][r_use]
            ##########################
        
        return
    #############################

    #############################
    def save_train(self,params_train):
        """ Save training params to file. """
        with open(self.file_stem + '/train.pkl', 'wb') as f:
            pickle.dump(params_train,f)
    #############################

    #############################
    def load_train(self):
        """ Load training params from file. """
        with open(self.file_stem + '/train.pkl', 'rb') as f:
            params_train = pickle.load(f)
        return params_train
    #############################
    
#################################
