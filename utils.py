import numpy as np

def sample_unseen(pool_size, sample_size, exclude):
    '''Efficient sampling from a range with exclusion'''
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