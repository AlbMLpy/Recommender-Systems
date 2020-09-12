import numpy as np

def sample_unseen(pool_size, sample_size, exclude):
    """Efficient sampling from a range with exclusion"""
    
    assert (pool_size-len(exclude)) >= sample_size 
    src = np.random.rand(pool_size)
    np.put(src, exclude, -1) # will never get to the top
    return np.argpartition(src, -sample_size)[-sample_size:]


def topk_idx(arr, topk, unsorted=False):
    'Select top-k elements. Sort for raniking metrics.'
    
    top_unsorted = np.argpartition(arr, -topk)[-topk:]
    if unsorted:
        return top_unsorted
    return top_unsorted[np.argsort(-arr[top_unsorted])]


def evaluate(scores, holdout_items, holdout_unseen, topk=10):
    """
    Evaluation of RS model with HR & ARHR.
    
    Input:
        - scores -> Numpy.ndarray[-, -] (new full rating matrix) 
        - holdout_items -> Numpy.ndarray[-] (items to be checked per every user)
        - holdout_unseen -> Numpy.ndarray[-, -] (unseen 999 items per every user)
    
    Output:
        return -> hr, arhr - these are numbers
    """
    
    rows = np.arange(len(holdout_items))
    holdout_scores = scores[rows, holdout_items]
    random_scores = scores[
        np.broadcast_to(rows[:, None], holdout_unseen.shape),
        holdout_unseen
    ]
    
    test_scores = np.concatenate((holdout_scores[:, None], random_scores), axis=1)
    top_recs = np.apply_along_axis(topk_idx, 1, test_scores, topk)        
    _, rec_pos = np.where(top_recs == 0) # holdout has index 0 by construction
    
    hr = len(rec_pos) / len(holdout_items)
    arhr = np.reciprocal(rec_pos + 1.).sum() / len(holdout_items)
    return hr, arhr