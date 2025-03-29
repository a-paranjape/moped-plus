import numpy as np
import scipy.linalg as linalg
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory

import sys
from paths import ML_Path
sys.path.append(ML_Path)
from mlstats import Chi2Like

#########################################
class NRELike(Likelihood):
    #########################################
    def initialize(self):
        pass
    #########################################

    #########################################
    def get_requirements(self):
        """ Theory code should return model array. """
        return {'model': None}
    #########################################

    #########################################
    def logp(self,**params_values_dict):
        out = self.provider.get_model()
        return out
    #########################################

#########################################

   
#########################################
class NRETheory(Theory):
    nre = None
    data = None
    keys = [] # list of parameter names
    #########################################
    def initialize(self):
        pass
    #########################################

    #########################################
    def calculate(self,state, want_derived=False, **param_dict):
        params = self.nre.cv([param_dict[key] for key in self.keys])
        logp = np.log(self.nre.predict(self.data,params))
        state['model'] = logp
    #########################################

    #########################################
    def get_model(self):
        return self.current_state['model']
    #########################################

    #########################################
    def get_allow_agnostic(self):
        return True
    #########################################


#########################################


#########################################
