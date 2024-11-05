import numpy as np
import scipy.linalg as linalg

class MOPED(object):
    """ Class implementing MOPED algorithm from Heavens+2000: """
    def __init__(self,data_pack={}):
        """ MOPED algorithm from Heavens+2000 (https://ui.adsabs.harvard.edu/abs/2000MNRAS.317..965H/abstract).
            data_pack should be a dictionary with the following keys:
            --     'data' [mandatory]: 1-d array of shape (N,) containing data vector
            -- 'data_cov' [optional] : if specified, should be (N,N) symmetric array with data covariance matrix C
            -- 'dmdtheta' [mandatory]: (M,N array) containing derivatives of N-dimensional model prediction wrt M parameters
            Outputs stored in:
            -- self.eig_vec: eigenvectors, shape (M,N). self.eig_vec[m] is the m-th optimized eigenvector
            -- self.data_comp: compressed data, shape (M,). self.data_comp[m] = self.eig_vec[m] . self.data
            -- self.eig_unnorm: non-optimized eigenvectors, shape (M,N). self.eig_unnorm[m] = C^-1 dmdtheta[m]
        """
        for key in ['data','dmdtheta']:
            if key not in data_pack.keys():
                raise Exception("data_pack missing a mandatory key:"+key)                
            
        for key in ['data_cov']:
            if key not in data_pack.keys():
                print("data_pack missing an optional key: "+key+". Defaults will be used.")                
            
        self.data_pack = data_pack
        self.data = data_pack.get('data',None)
        self.data_cov = data_pack.get('data_cov',None)
        self.dmdtheta = data_pack.get('dmdtheta',None)

        self.check_init()
        
        self.N = self.data.shape[0]
        self.M = self.dmdtheta.shape[0]

        self.L_cholesky = linalg.cholesky(self.data_cov,lower=True) # so C = L L^T

        self.moped() # implements MOPED and creates self.eig_vec, self.data_comp and self.eig_unnorm

    def check_init(self):
        """ Convenience function to check inputs. """
        if self.data is None:
            raise Exception("data must be specified in MOPED.")

        if len(self.data.shape) != 1:
            raise Exception("data must be 1-d array in MOPED.")

        if self.dmdtheta is None:
            raise Exception("dmdtheta must be specified in MOPED.")

        if len(self.dmdtheta.shape) != 2:
            raise Exception("dmdtheta must be 2-d array in MOPED.")

        if self.dmdtheta.shape[1] != self.data.shape[0]:
            raise Exception("Need dmdtheta.shape[1] == data.shape[0] in MOPED.")

        if self.data_cov is None:
            print("Data covariance not specified in MOPED. Assuming identity.")
            self.data_cov = np.eye(self.data.shape[0])

        if self.data_cov.shape != (self.data.shape[0],self.data.shape[0]):
            raise Exception("Need data_cov.shape == (data.shape[0],data.shape[0]) in MOPED.")

        for i in range(self.data.shape[0]-1):
            for j in range(i+1,self.data.shape[0]):
                if self.data_cov[i,j] != self.data_cov[j,i]:
                    raise Exception("Asymmetry detected in data covariance at (i,j) = ({0:d},{1:d}) in MOPED.".format(i,j))
            
        return

    def moped(self):
        """ MOPED algorithm to calculate eigenvectors and compressed data points. """

        self.eig_unnorm = np.zeros_like(self.dmdtheta) # storage for C^-1 dmdtheta
        for m in range(self.M):
            self.eig_unnorm[m] = linalg.cho_solve((self.L_cholesky,True),self.dmdtheta[m],check_finite=False)
            # solves (L L^T) eig_unnorm[m] = dmdtheta[m] or eig_unnorm[m] = C^-1 dmdtheta[m], shape (N,)
        
        self.eig_vec = np.zeros_like(self.dmdtheta)

        self.eig_vec[0] = self.eig_unnorm[0]/np.sqrt(np.sum(self.dmdtheta[0]*self.eig_unnorm[0]))
        for m in range(1,self.M):
            sum_vector = np.zeros_like(self.data)
            sum_scalar = 0.0
            for q in range(m):
                dmu_dot_b = np.sum(self.dmdtheta[m]*self.eig_vec[q])
                sum_vector += dmu_dot_b*self.eig_vec[q]
                sum_scalar += dmu_dot_b**2
            numerator = self.eig_unnorm[m] - sum_vector
            denominator = np.sum(self.dmdtheta[m]*self.eig_unnorm[m]) - sum_scalar + 1e-15
            self.eig_vec[m] = numerator / np.sqrt(denominator)

        self.data_comp = np.sum(self.eig_vec*self.data,axis=1)
        # print('these should be positive:')
        # for m in range(self.M):
        #     print(np.sum(self.dmdtheta[m]*self.eig_unnorm[m]))
            
        return
