import argparse
import os
import numpy as np
import pandas as pd

def int_or_float(x):
    try:
        val = int(x)
        assert val >= 0
    except ValueError:
        val = float(x)
        assert 0.0 <= val <= 1.0
    return val

def do_simple_split(path_train_csv, path_test_csv, train_size, df_all, vprint=False):
    
    if os.path.exists(path_train_csv) and os.path.exists(path_test_csv):
        os.remove(path_train_csv)
        os.remove(path_test_csv)
        vprint('Removing existing train/test splits.')
    if train_size == 0:
        df_test = df_all
        df_train = df_all[0:0]  # empty DataFrame but with columns intact
    else:
        if isinstance(train_size, int):
            idx_split = train_size
        elif isinstance(train_size, float):
            idx_split = round(len(df_all)*train_size)
        else:
            raise AttributeError
        df_train = df_all[:idx_split]
        df_test = df_all[idx_split:]
    vprint('train/test sizes: {:d}/{:d}'.format(len(df_train), len(df_test)))

    df_train.to_csv(path_train_csv, index=False)
    df_test.to_csv(path_test_csv, index=False)
    vprint('saved:', path_train_csv)
    vprint('saved:', path_test_csv)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_csv', help='path to dataset CSV')
    parser.add_argument('dst_dir', help='destination directory of dataset split')
    parser.add_argument('--kfolds', type=int, default=-1, help='if you want k-fold validation specify number of folds')
    parser.add_argument('--val_split', type=float, default=0.1, help='if you dont use cross-validation define how to split data')
    parser.add_argument('--train_size', type=int_or_float, default=0.8, help='training set size as int or faction of total dataset size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--no_shuffle', action='store_true', help='random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    opts = parser.parse_args()

    vprint = print if opts.verbose else lambda *a, **kw: None   
    
    #name = os.path.basename(opts.src_csv).split('.')[0] #opts.src_csv="pytorch_fnet/data/csvs/${DATASET}.csv"
    #path_store_split = os.path.join(opts.dst_dir, name) #opts.dst_dir="pytorch_fnet/data/csvs"
    path_store_split = opts.dst_dir

    rng = np.random.RandomState(opts.seed)
    df_all = pd.read_csv(opts.src_csv)
    
    if not opts.no_shuffle:
        df_all = df_all.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    
    if not os.path.exists(path_store_split):
        os.makedirs(path_store_split)
    
    # split data into train and val only once
    if opts.kfolds==1:
        path_train_csv = os.path.join(path_store_split, 'train_1.csv')
        path_test_csv = os.path.join(path_store_split, 'val_1.csv')
        do_simple_split(path_train_csv, path_test_csv, 1.-opts.val_split, df_all, vprint)
    
    # or into k-folds
    elif opts.kfolds>1:
        
        split_percentage = 100/opts.kfolds # e.g. 100/5=20%
        split_samples = round(len(df_all)*split_percentage/100) # e.g. 11
        for i in range(opts.kfolds):
            path_train_csv = os.path.join(path_store_split, 'train_{}.csv'.format(i+1))
            path_test_csv = os.path.join(path_store_split, 'val_{}.csv'.format(i+1))
            if os.path.exists(path_train_csv) and os.path.exists(path_test_csv):
                os.remove(path_train_csv)
                os.remove(path_test_csv)
                vprint('Removing existing train/test splits.')
            if i==0:
                idx_start=0
            idx_end=idx_start+split_samples
            df_train = pd.concat([df_all[:idx_start], df_all[idx_end:]]) 
            df_test = df_all[idx_start:idx_end]

            df_train.to_csv(path_train_csv, index=False)
            df_test.to_csv(path_test_csv, index=False)
            vprint('saved:', path_train_csv)
            vprint('saved:', path_test_csv)
            
            idx_start=idx_end

    # else it will be -1 and we do the original train test split
    else:
        path_train_csv = os.path.join(path_store_split, 'train.csv')
        path_test_csv = os.path.join(path_store_split, 'test.csv')
        do_simple_split(path_train_csv, path_test_csv, opts.train_size, df_all, vprint)

    

if __name__ == '__main__':
    main()


