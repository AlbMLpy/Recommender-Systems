# Recommender-Systems


# Preprocessing Yahoo Data from the RecWalk Repo
> Julia
```julia
using SparseArrays, DataFrames, CSV, MAT

DATA = matread("yahoo.mat")
TrainSet = DATA["TrainSet"]
Holdout = DATA["Holdout"]

users, items, vals = findnz(TrainSet)
train_df = DataFrame([:userid => users, :itemid => items])
user_idx = sort(unique(users))
holdout_df = DataFrame([:userid => user_idx, :itemid => Holdout]);

CSV.write("_train_df.csv", train_df)
CSV.write("_holdout_df.csv", holdout_df)
```

> Python
```python
import pandas as pd

train_df = pd.read_csv("_train_df.csv")
train_df['is_holdout'] = False

holdout_df = pd.read_csv("_holdout_df.csv")
holdout_df['is_holdout'] = True

full_df = train_df.append(holdout_df, ignore_index=True)
full_df.to_csv('yahoo_data_full.gz', index=False)
```