from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np
from math import log
from scipy.special import rel_entr
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, csr_matrix, vstack

class Jensen_Shannon():

    def __init__(self, Pi1 = 0.5, Pi2= 0.5):
        
        assert Pi1 + Pi2 == 1
        
        self.Pi1 = Pi1
        self.Pi2= Pi2
     

    def linear_JSD(self, P, Q, cte=1e-10):
    
        """
        Input : P and Q are Probability distribution vectors
        Output : Jensen Divergence between all individual dimension of vectors -- list
        """
        
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        #print(_P)
        _M = self.Pi1 * _P + self.Pi2 * _Q
        
        indiv_JDS = []
        for i in range(len(_M)):
            JSD_i = -_M[i]*log(_M[i]+cte) + self.Pi1*_P[i]*log(_P[i]+cte) + self.Pi2*_Q[i]*log(_Q[i]+cte)
            indiv_JDS.append(JSD_i)
    
        return indiv_JDS
    
    def JSDiv(self, P, Q):
        """
        Input: P and Q are probability distribution vectors
            P is the known distribution and Q is the novel distribution.
        """
        # Convert P and Q to numpy arrays
        _P = np.asarray(P)
        _Q = np.asarray(Q)

        # Ensure that the sum of P and Q are not zero
        sum_P = np.sum(_P)
        sum_Q = np.sum(_Q)

        # Replace zeros with a small number to avoid division by zero
        if sum_P == 0:
            _P = np.ones_like(_P) * np.finfo(float).eps  # Assign small values if sum is zero
        else:
            _P /= sum_P  # Normalize P

        if sum_Q == 0:
            _Q = np.ones_like(_Q) * np.finfo(float).eps  # Assign small values if sum is zero
        else:
            _Q /= sum_Q  # Normalize Q

        # Compute the mixture distribution
        _M = self.Pi1 * _P + self.Pi2 * _Q

        # Compute the JS divergence using relative entropy (KL divergence)
        js_div = self.Pi1 * np.sum(rel_entr(_P, _M)) + self.Pi2 * np.sum(rel_entr(_Q, _M))

        return js_div
    

    def JSDiv_fC(self, P, Q):
    
        """
        Input : P and Q are Probability distribution vectors
                P is the known distribution and Q should be the novel distribrution
        """
        P = P.toarray().flatten()
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = self.Pi1 * _P + self.Pi2 * _Q
        return self.Pi1 * entropy(_P, _M) + self.Pi2 * entropy(_Q, _M)

