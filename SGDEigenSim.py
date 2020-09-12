import time

import numpy as np
import pandas as pd


class SGDEigenSim:
    def __init__(self, feedback, random_state=1):
        self._r = feedback.copy()
        self._rr_pseudo = self._r.dot(np.linalg.pinv(self._r.A))
        self._n_users = feedback.shape[0]
        self._n_items = feedback.shape[1]
        r = np.random.RandomState(random_state)
        self.eigenval_diag = r.rand(self._n_users)
    
    def get_appr(self):
        """ Return approximation for initial feedback matrix"""
        return self._rr_pseudo.dot(
            self.eigenval_diag.reshape((self._n_users, 1)) 
            * self._r.A
        )
    
    def get_model(self):
        """ Return model matrix for Eigenvalue analogy 'S' """
        return np.linalg.pinv(self._r.A).dot(
            self.eigenval_diag.reshape((self._n_users, 1)) 
            * self._r.A
        )
    
    def get_eigenvalues(self):
        """ Return array of eigenvalues approximation """
        return self.eigenval_diag
    
    def _give_grad(self, pos_u, pos_i, gm):
        grad = np.zeros(self._n_users)
        for i in pos_u:
            for j in pos_i:
                ivi = 2 * (self._r[i, j] 
                           - np.sum(
                               self.eigenval_diag
                               * self._r.A[:, j]
                               * self._rr_pseudo[i, :]))
        
                vec = self._rr_pseudo[i, :] * self._r.A[:, j]
                grad += -1 * ivi * vec
                
        if gm is not None:
            grad += 2 * gm * self.eigenval_diag
        
        return grad / (pos_u.size**2)
    
    def _give_error(self, r_approximate):
        return (1 / (self._n_users * self._n_items) 
                * np.sum((self._r.A - r_approximate)**2))    
    
    def fit(self,
            alpha=0.5,
            batch_size=5,
            error=1e-4,
            num_iter=150,
            show_step=(True, 10),
            gm=None):
        
        r_appr = self.get_appr()
        got_error = self._give_error(r_appr)
        n_iter = 0
        print("Begin calculations:\n")
        start = time.time()
        
        while got_error > error:
            self.eigenval_diag -= alpha * self._give_grad(
                np.random.randint(self._n_users, size=batch_size),
                np.random.randint(self._n_items, size=batch_size),
                gm=gm,
            )
                
            r_appr = self.get_appr()
            got_error = self._give_error(r_appr)
            n_iter += 1
            if show_step[0]:
                if n_iter % show_step[1] == 0:
                    end = time.time()
                    print(
                        f"Iteration: {n_iter}, "
                        + f"Error: {got_error}, " 
                        + f"Time for {show_step[1]} steps: {end - start}\n\n"
                    )
                    start = time.time()

            if n_iter == num_iter:
                print("Ended with iterations\n")
                break
                
        print(f"Got error = {got_error};\n")       
        return
    
    def predict_top_n(self, user, num=10):
        pass
    
        
