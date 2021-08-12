import argparse
import os
import json
import optuna
import csv
import pandas as pd
from train_model import Trainer

def get_best_hyperparams(config, results_file, fine_tune=False):
    df = pd.read_csv(results_file)
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
        best_hyperparams['patch_size'] = df['patch size'].mode()
    
    for key in best_hyperparams.keys():
        config[key] = best_hyperparams[key]
    return config

def run_experiment(config, path_run_dir, results_file, fine_tune=False):

    with open(results_file, 'w', newline='') as csv_file:

        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['cv fold','search id', 'loss weight', 'learning rate', 'depth', 'patch size', 'dropout', 'resampling probability', 'resampling threshold', 'freeze layers'])

        trainer = Trainer(config, path_run_dir=path_run_dir, fine_tune=fine_tune)
        path_dataset_csv_paths = os.path.join(config['data_path'], 'csvs', config['dataset'])

        for fold_id in range(config['kfolds']):
            print('Starting search on fold ', fold_id+1)
            path_dataset_train_csv = ('/').join([path_dataset_csv_paths,"train_{}.csv".format(fold_id+1)])
            path_dataset_val_csv = ('/').join([path_dataset_csv_paths, "val_{}.csv".format(fold_id+1)])
            trainer.set_train_val_sets(path_dataset_train_csv, path_dataset_val_csv)

            for search_id in range(5):
                print('Starting search {} of fold {}'.format(search_id+1,fold_id+1))
                study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner()) #minimize
                study.optimize(trainer.train_for_search, n_trials=config['training']['num_trials'])

                pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
                complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
                trial = study.best_trial
                if opts.v:
                    print("Study statistics: ")
                    print("  Number of finished trials: ", len(study.trials))
                    print("  Number of pruned trials: ", len(pruned_trials))
                    print("  Number of complete trials: ", len(complete_trials))
                    print("Best trial:")
                    print("  Value: ", trial.value)
                    print("  Params: ")
                    for key, value in trial.params.items():
                        print("    {}: {}".format(key, value))
            
            if fine_tune:
                writer.writerow([fold_id, search_id, trial.lr, None, None, trial.dropout, trial.resample, trial.threshold, trial.freeze_layers])
            else:
                writer.writerow([fold_id, search_id, trial.lr, trial.depth, trial.patch, trial.dropout, trial.resample, trial.threshold, None])

    csv_file.close()

    config = get_best_hyperparams(config, results_file, fine_tune) # from all best hyperparameters get average
    trainer.update_config(config)
    path_dataset_train_csv = ('/').join([path_dataset_csv_paths,"train.csv"])
    trainer.set_train_val_sets(path_dataset_train_csv, None)
    trainer.train_best() #run one time with newly computed hyperparameters
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config',type=str, help='config dictionary')
    parser.add_argument('--fine_tune', action='store_true', help='set to true if you wish to finetune existing model')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    opts = parser.parse_args()

    with open(opts.config, "r") as fp:
        config = json.load(fp)

    if opts.fine_tune:
        path_run_dir = os.path.join(config['output_path'], config['dataset'], config['run'], 'fine_tuned')
    else:
        path_run_dir = os.path.join(config['output_path'], config['dataset'], config['run'], 'train_from_scratch')

    results_file = os.path.join(path_run_dir, 'hyperaparams.cvs')
    # run experiment and save best model
    run_experiment(config, path_run_dir, results_file, opts.fine_tune) 