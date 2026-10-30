import os
import pandas as pd

if not os.path.exists(f'MMRec/data/Sports/Sports.inter'):   #Baby Beauty Office Sports
    train = pd.read_csv(f'MMRec/data/Sports/train_final_indexed.tsv', sep='\t', header=None)
    val = pd.read_csv(f'MMRec/data/Sports/val_final_indexed.tsv', sep='\t', header=None)
    test = pd.read_csv(f'MMRec/data/Sports/test_final_indexed.tsv', sep='\t', header=None)
    train.columns = ['userID', 'itemID', 'rating']
    val.columns = ['userID', 'itemID', 'rating']
    test.columns = ['userID', 'itemID', 'rating']
    train['x_label'] = pd.Series([0] * len(train))
    val['x_label'] = pd.Series([1] * len(val))
    test['x_label'] = pd.Series([2] * len(test))
    df = pd.concat([train, val, test])[['userID', 'itemID', 'x_label']]
    df.to_csv(f'MMRec/data/Sports/Sports.inter', index=False, sep='\t') 

#Command to run: python MMRec/create_dataset.py