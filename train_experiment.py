import argparse
import os
import json
import optuna
import csv
import pandas as pd
from train_model import Trainer

def get_best_hyperparams(config, results_file, fine_tune=False):
    '''
    This function computes the best hyperparameters for training the final model

    After training and doing a repeated hyperparameter search and cross-validation
    the best hyperaparameters need to be computed. This is for most cases computed 
    as the average of all runs, except for the categorical parameters (majority voting).
    The config is updated with these hyperparameters and returned.

    Parameters
    ----------
    config : dict
        The config file - usually config.yaml
    results_file : str
        Will be e.g. output-path/dataset/run/fine_tuned/hyperparameters.csv and hold
        best hyperparameters for all runs
    fine_tine: bool
        Whether we are computing the best hyperparameters for the fine-tuning case 
        or training from scratch (different hyperparameters for each) 
        
    Returns
    -------
    dict
        The updated config with the best hyperparameters
    '''
    
    df = pd.read_csv(results_file, delimiter=',')
    best_hyperparams = dict()

    best_hyperparams['lr'] = df['learning rate'].mean()
    best_hyperparams['loss_weight'] = df['loss weight'].mean()
    best_hyperparams['dropout'] = df['dropout'].mean()
    best_hyperparams['resampling_probability'] = df['resampling probability'].mean()
    best_hyperparams['threshold_backround'] = df['resampling threshold'].mean()
    
    if fine_tune:
        best_hyperparams['num_freeze_layers'] = round(df['freeze layers'].mean())
    else:
        best_hyperparams['depth'] = round(df['depth'].mean())
        best_hyperparams['patch_size'] = [int(df['patch size'].mode().values[0]), int(df['patch size'].mode().values[0])]
    
    for key in best_hyperparams.keys():
        config[key] = best_hyperparams[key]

    return config

def run_experiment(config, path_run_dir, results_file, fine_tune=False, verbose=False):
    '''
    This main function - conducts the whole experiment for either fine-tuning or training
    from scratch.

    In this function first a hyperparameter search is performed (repeated config['searches_per_fold']
    times for each k-fold) and the best hyperaparemeters for each run are stored. Then the 
    best overall hyperparameters are computed (as an average of all runs) and a model is trained
    with these hyperparameters.

    Parameters
    ----------
    config : dict
        The config file - usually config.yaml
    path_run_dir : str
        The path where training outputs will be stored, either 
        output_path/dataset/run/fine_tuned or output_path/dataset/run/train_from_scratch
    results_file: str
        Will be e.g. output-path/dataset/run/fine_tuned/hyperparameters.csv and hold
        best hyperparameters for all runs
    fine_tine: bool
        Whether we are computing the best hyperparameters for the fine-tuning case 
        or training from scratch (different hyperparameters for each) 
    verbose: bool
        Whether or not to print training updates

    '''
    with open(results_file, 'w') as csv_file:
       
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['cv fold','search id', 'loss weight', 'learning rate', 'depth', 'patch size', 'dropout', 'resampling probability', 'resampling threshold', 'freeze layers'])
        
        #create trainer instance
        trainer = Trainer(config, path_run_dir=path_run_dir, fine_tune=fine_tune)
        path_dataset_csv_paths = os.path.join(config['data_path'], 'csvs', config['dataset'])

        for fold_id in range(config['kfolds']):
            print('Starting search on fold {}/{}'.format(fold_id+1, config['kfolds']))
            # set train and validation dataset paths
            path_dataset_train_csv = ('/').join([path_dataset_csv_paths,"train_{}.csv".format(fold_id+1)])
            path_dataset_val_csv = ('/').join([path_dataset_csv_paths, "val_{}.csv".format(fold_id+1)])
            trainer.set_train_val_sets(path_dataset_train_csv, path_dataset_val_csv)

            for search_id in range(config['searches_per_fold']):
                print('Starting search {}/{} of fold {}/{}'.format(search_id+1,config['searches_per_fold'],fold_id+1, config['kfolds']))
                # create optuna study for hyperparameter search
                study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner()) #minimize
                # and train x config['training']['num_trials'] times to find best hyperparameter configuration
                study.optimize(trainer.train_for_search, n_trials=config['training']['num_trials'])

                pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
                complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                trial = study.best_trial
                best_params = study.best_params
                if verbose:
                    print("Study statistics: ")
                    print("  Number of finished trials: ", len(study.trials))
                    print("  Number of pruned trials: ", len(pruned_trials))
                    print("  Number of complete trials: ", len(complete_trials))
                    print("Best trial:")
                    print("  Value: ", trial.value)
                    print("  Params: ")
                    for key, value in trial.params.items():
                        print("    {}: {}".format(key, value))
            
                trainer.reset_trial_id()
                
                # write best hyperparameters to results file
                if fine_tune:
                    writer.writerow([fold_id, search_id, best_params['loss_weight'],best_params['lr'], None, None, best_params['dropout'], best_params['resample'], best_params['threshold'], best_params['freeze_layers']])
                else:
                    writer.writerow([fold_id, search_id, best_params['loss_weight'], best_params['lr'], best_params['depth'], best_params['patch'], best_params['dropout'], best_params['resample'], best_params['threshold'], None])

    csv_file.close()
    
    print('Going to train a model with the average of the best hyperparameter')

    config = get_best_hyperparams(config, results_file, fine_tune) # from all best hyperparameters get average
    trainer.update_config(config) # and update the config file with those hyperparameters
    
    # update train-val dataset to use entire split for training - no validation
    path_dataset_train_csv = ('/').join([path_dataset_csv_paths,"train.csv"])
    trainer.set_train_val_sets(path_dataset_train_csv, None)
    trainer.train_best() #run one time with newly computed hyperparameters
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config',type=str, help='config dictionary')
    parser.add_argument('--fine_tune', action='store_true', help='set to true if you wish to finetune existing model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    opts = parser.parse_args()
    
    if opts.fine_tune:
        print('Starting fine-tuning')
    else:
        print('Starting training from scratch')
    
    # load the config file
    with open(opts.config, "r") as fp:
        config = json.load(fp)

    # set run path depending on if we are fine-tuning or not
    if opts.fine_tune:
        path_run_dir = os.path.join(config['output_path'], config['dataset'], config['run'], 'fine_tuned')
    else:
        path_run_dir = os.path.join(config['output_path'], config['dataset'], config['run'], 'train_from_scratch')
    if not os.path.exists(path_run_dir):
        os.makedirs(path_run_dir)
    
    # create a results file to save the hyperparameter search results from training
    results_file = os.path.join(path_run_dir, 'hyperaparams.csv')
    # run experiment and save best model
    run_experiment(config, path_run_dir, results_file, opts.fine_tune, opts.verbose) 

