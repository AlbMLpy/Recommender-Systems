import numpy as np


def split_holdout(data,
                  userid='userid',
                  feedback=None,
                  sample_max_rated=False,
                  random_state=None):
    """ 
    Sample 1 item per every user according to the rule sample_max_rated.
    
    Input:
        - data -> Pandas.Dataframe
    Output:
        - observed -> Pandas.Dataframe
        - holdout -> Pandas.Dataframe
    """
    
    idx_grouper = (
        data
        .sample(frac=1, random_state=random_state) # randomly permute data
        .groupby(userid, as_index=False, sort=False)
    )
    if sample_max_rated: # take single item with the highest score
        idx = idx_grouper[feedback].idxmax()
    else: # data is already permuted - simply take the 1st element
        idx = idx_grouper.head(1).index # sample random element
    
    observed = data.drop(idx.values)
    holdout = data.loc[idx.values]
    return observed, holdout


def sample_unseen_items(item_group, item_pool, n, random_state):
    """
    Helper function to run on pandas dataframe grouper.
    
    Input:
        - item_group -> Numpy.ndarray (items seen by a user) 
        - item_pool -> Numpy.ndarray (all the items)
    
    Output:
        return -> random n unseen items per user
    """
    
    seen_items = item_group.values
    candidates = np.setdiff1d(item_pool, seen_items, assume_unique=True)
    return random_state.choice(candidates, n, replace=False)


def sample_unseen_interactions(data,
                               item_pool,
                               userid='userid',
                               itemid='itemid',
                               n_random=999,
                               seed=None):
    
    """
    Randomized sampling of unseen items per every user in data.
    It is essential to conduct experiment regularly.
    
    Input:
        - data -> Pandas.Dataframe
    Output:
        return -> Pandas.Series      
    """
    random_state = np.random if seed is None else np.random.RandomState(seed)
    return (
        data
        .groupby(userid, sort=False)[itemid]
        .apply(sample_unseen_items, item_pool, n_random, random_state)
    )