import numpy as np
from scipy.sparse import csr_matrix, diags, eye, block_diag, bmat
from scipy.sparse.linalg import norm


class RecWalk:
    def __init__(self, item_model, rating_matrix, alpha=0.005):
        inter_item = item_model / norm(item_model, np.inf)
            
        # stochasticity adjustment
        adjustment = 1 - inter_item.sum(axis=1).A.squeeze()
        inter_item += diags(adjustment,
                            shape=item_model.shape,
                            format='csr')
        # M matrix
        transition = block_diag(
            (eye(rating_matrix.shape[0], format='csr'),
             inter_item),
            format = 'csr',
            dtype='float64',
        )
        # H matrix
        walk_model = bmat(
            [[None, rating_matrix], [rating_matrix.T, None]],
            format='csr',
            dtype='float64',
        )
        k = np.reciprocal(walk_model.sum(axis=1).A.squeeze())
        walk_model = diags(k, format='csr').dot(walk_model)
        self._p = alpha * walk_model + (1-alpha) * transition

    def get_model(self):
        return self._p
    
    def make_steps(self, k_steps=10):
        pass