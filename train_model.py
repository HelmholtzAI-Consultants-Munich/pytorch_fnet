import argparse
import os

import fnet.data
import fnet.fnet_model
from fnet.functions import compute_dataset_min_max_ranges, pearsonr
from fnet.transforms import Propper

import json
import logging
import numpy as np

import pdb
import sys
import time
import torch
import warnings

import optuna
import math
import joblib
from datetime import datetime

class Trainer(object):

    def __init__(self, config, fine_tune=False, path_run_dir=None, path_dataset_train_csv=None, path_dataset_val_csv=None, verbose=False):
        self.config = config
        self.path_run_dir = path_run_dir
        self.path_dataset_train_csv = path_dataset_train_csv
        self.path_dataset_val_csv = path_dataset_val_csv
        self.fine_tune = fine_tune
        self.verbose = verbose
        self.config['gpu_ids'] = [self.config['gpu_ids']] if isinstance(self.config['gpu_ids'], int) else self.config['gpu_ids']
        self.device = torch.device('cuda', self.config['gpu_ids'][0]) if self.config['gpu_ids'][0] >= 0 else torch.device('cpu')
        #Setup logging
        self.setup_logger()
        self.trial_id = 0
        
    def reset_trial_id(self):
        self.trial_id = 0
    
    def set_run_dir(self, path_run_dir):
        self.path_run_dir = path_run_dir

    def set_train_val_sets(self, path_dataset_train_csv, path_dataset_val_csv):
        self.path_dataset_train_csv = path_dataset_train_csv
        self.path_dataset_val_csv = path_dataset_val_csv

    def get_dataloader(self, remaining_iterations, validation=False):
        
        min_max_bright, min_max_infection, min_max_dapi = compute_dataset_min_max_ranges(self.path_dataset_train_csv, self.path_dataset_val_csv)
        min_max_bright_norm, _, _ = compute_dataset_min_max_ranges(self.path_dataset_train_csv, self.path_dataset_val_csv, norm=True)

        transform_signal=[]
        for t in self.config['preprocessing']['transform_signal']:
            if t=='fnet.transforms.AdaptRange':
                t = 'fnet.transforms.AdaptRange({:f},{:f})'.format(min_max_bright_norm[0], min_max_bright_norm[1])  
            transform_signal.append(eval(t))
        transform_target = [eval(t) for t in self.config['preprocessing']['transform_target']]
        transform_thresh = []
        
        if validation:
            propper = Propper(action='+') 
            print(propper)
            transform_signal.append(propper)
            transform_target.append(propper)
            transform_thresh.append(propper)
            
        ds = getattr(fnet.data, self.config['class_dataset'])(
            path_csv = self.path_dataset_train_csv if not validation else self.path_dataset_val_csv,
            transform_source = transform_signal,
            transform_target = transform_target,
            transform_thresh = transform_thresh,
            min_max_bright = min_max_bright, 
            min_max_dapi = min_max_dapi, 
            min_max_infection = min_max_infection,
            augmentations = self.config['training']['augmentations'] if not validation else False
        )
        #print(ds)
        
        if not validation:
            assert len(ds)>=self.config['buffer_size'], 'buffer_size cannot be larger than \
            the training data. Please set buffer_size in the config smaller or equal to your \
            training data size and try again. Exiting.'
            ds_patch = fnet.data.BufferedPatchDataset(
                dataset = ds,
                patch_size = self.config['patch_size'],
                buffer_size = self.config['buffer_size'],
                buffer_switch_frequency = self.config['buffer_switch_frequency'],
                npatches = remaining_iterations*self.config['batch_size'], #None, #
                verbose = False,
                shuffle_images = self.config['shuffle_images'],
                threshold_backround = self.config['threshold_backround'], #0.2
                resampling_probability = self.config['resampling_probability'], #0.7,
                **self.config['bpds_kwargs'],
            )
        else:
            ds_patch = fnet.data.AllPatchesDataset(
                dataset = ds,
                patch_size = self.config['patch_size'],
                buffer_size = len(ds),
                buffer_switch_frequency = -1,
                verbose = False
        )
        dataloader = torch.utils.data.DataLoader(
            ds_patch, #ds
            batch_size = self.config['batch_size'],
        )
        return dataloader

    def update_config(self, config):
        self.config = config

    def setup_logger(self):
        self.logger = logging.getLogger('model training')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.path_run_dir, 'run.log'), mode='a')
        sh = logging.StreamHandler(sys.stdout)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

    def train_for_search(self, trial):
        self.trial_id += 1
        print('Starting trial {}/{}'.format(self.trial_id, self.config['training']['num_trials']))
        # define hyperparameters to tune
        if self.fine_tune:
            self.config['num_freeze_layers'] = trial.suggest_int('freeze_layers', 1,106) #97
        else:
            self.config['depth'] = trial.suggest_int('depth', 2, 6) #3
            patch_l = trial.suggest_categorical('patch', [128,256])
            self.config['patch_size'] = [patch_l, patch_l] #[256, 256]
        
        self.config['lr'] = trial.suggest_loguniform("lr", 1e-5, 1e-1) #0.00369
        self.config['resampling_probability'] = trial.suggest_float('resample', 0.5, 0.9) #0.6263
        self.config['threshold_backround'] = trial.suggest_float('threshold', 0.01, 0.5) #0.3463
        self.config['dropout'] = trial.suggest_float('dropout', 0.1, 0.5) #0.1078
        self.config['loss_weight'] = trial.suggest_float('loss_weight', 0.5, 0.9)

        #Set random seed
        if self.config['seed'] is not None:
            np.random.seed(self.config['seed'])
            torch.manual_seed(self.config['seed'])
            torch.cuda.manual_seed_all(self.config['seed'])

        #Instantiate Model
        if self.fine_tune:
            saved_model_path = self.config['path_model_dir'][0]
        else:
            saved_model_path = self.path_run_dir
            
        if os.path.exists(os.path.join(saved_model_path, 'model.p')):
            model = fnet.load_model_from_dir(saved_model_path, gpu_ids=self.config['gpu_ids'], in_channels=self.config['in_channels'], out_channels=self.config['out_channels'])
            self.logger.info('model loaded from: {:s}'.format(saved_model_path))
            # freeze first layers
            for freeze_idx, param in enumerate(model.net.parameters()):
                if freeze_idx<self.config['num_freeze_layers']:
                    param.requires_grad = False

        else:
            model = fnet.fnet_model.Model(
                nn_module=self.config['nn_module'],
                lr=self.config['lr'],
                gpu_ids=self.config['gpu_ids'],
                dropout= self.config['dropout'],
                in_channels=self.config['in_channels'],
                out_channels=self.config['out_channels'],
                depth = self.config['depth'],
                loss_weight=self.config['loss_weight'],
                min_dif=self.config['min_dif'],
                max_dif=self.config['max_dif']
            )

        n_remaining_iterations = max(0, (self.config['training']['n_iter'] - model.count_iter))
        dataloader_train = self.get_dataloader(n_remaining_iterations)
        if self.path_dataset_val_csv is not None:
            dataloader_val = self.get_dataloader(n_remaining_iterations, validation=True)
            criterion_val = model.criterion_fn(reduction='none')
        
        loss_train = 0
        pearson_train = 0
        
        for i, (signal, target, dapi_signal, dif_dapi_inf, _) in enumerate(dataloader_train, model.count_iter): #dna_channel
        
            if self.config['in_channels']==2:
                signal = torch.cat([signal, dapi_signal],  dim=1)
            loss_batch, pearson_batch = model.do_train_iter(signal, target, dif_dapi_inf)
            loss_train += loss_batch 
            pearson_train += pearson_batch
            
            if ((i + 1) % self.config['print_iter'] == 0) or ((i + 1) == self.config['training']['n_iter']):
                if self.verbose: print('For {}/{} iterations, average training loss: {:.3f} and average Pearson Correlation Coefficient: {:3f}'.format(i, n_remaining_iterations, loss_train/self.config['print_iter'], pearson_train/self.config['print_iter']))
                loss_train = 0 
                pearson_train = 0
                
                if self.path_dataset_val_csv is not None:
                    loss_val = 0
                    pearson = 0
                    pearson_idx =0
                    
                    for idx_val, sample in enumerate(dataloader_val):
                        patch, is_last = sample['patch'], sample['is_last']
                        signal_val = patch[0].to(device=self.device)
                        target_val = patch[1].to(device=self.device)
                        dif_dapi_inf = patch[-2].to(device=self.device)
                        
                        if self.config['in_channels']==2:
                            signal_val = torch.cat([signal_val, patch[2]],  dim=1)
                            
                        pred_val = model.predict(signal_val)
                        dif_mean = torch.mean(dif_dapi_inf, dim=(1,2,3))
    
                        for it,dif_mean_it in enumerate(dif_mean):
                            if dif_mean_it>self.config['min_dif'] and dif_mean_it<self.config['max_dif']:
                                dif_mean[it] = self.config['loss_weight']
                            else:
                                dif_mean[it] = 1-self.config['loss_weight']
                                
                        #dif_mean = torch.tensor(dif_mean, dtype=torch.float32, device=self.device)
                        dif_mean = dif_mean.to(device=self.device)
                        
                        loss_val_batch = criterion_val(pred_val, target_val)
                        loss_val_batch = torch.mean(loss_val_batch, dim=(1,2,3)) * dif_mean
                        loss_val += torch.mean(loss_val_batch).item()
                        
                        pearson_val = pearsonr(pred_val, target_val)
                        if not math.isnan(pearson_val):
                            pearson += pearson_val
                            pearson_idx += 1                  
                                            
                    loss_val = loss_val/idx_val #len(dataloader_val)
                    print('Validation loss: {:.3f}'.format(loss_val))
                    if pearson_idx==0:
                        pearson=0
                        print('Validation Pearson Correlation Coefficient is nan everywhere.')
                    else:
                        pearson=pearson/pearson_idx
                        print('Validation Pearson Correlation Coefficient: {:.3f}'.format(pearson))
                
                trial.report(loss_val, i)
        
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        return pearson

    def train_best(self):
        
        #Set random seed
        if self.config['seed'] is not None:
            np.random.seed(self.config['seed'])
            torch.manual_seed(self.config['seed'])
            torch.cuda.manual_seed_all(self.config['seed'])
        #Instantiate Model
        if self.fine_tune:
            saved_model_path = self.config['path_model_dir'][0]
        else:
            saved_model_path = self.path_run_dir
            
        path_model = os.path.join(self.path_run_dir, 'model.p')
        
        if os.path.exists(os.path.join(saved_model_path,'model.p')):
            model = fnet.load_model_from_dir(saved_model_path, gpu_ids=self.config['gpu_ids'], in_channels=self.config['in_channels'], out_channels=self.config['out_channels'])
            self.logger.info('model loaded from: {:s}'.format(saved_model_path))
            # freeze first layers
            for freeze_idx, param in enumerate(model.net.parameters()):
                if freeze_idx<self.config['num_freeze_layers']:
                    param.requires_grad = False
                    
        else:
            model = fnet.fnet_model.Model(
                nn_module=self.config['nn_module'],
                lr=self.config['lr'],
                gpu_ids=self.config['gpu_ids'],
                dropout= self.config['dropout'],
                in_channels=self.config['in_channels'],
                out_channels=self.config['out_channels'],
                depth = self.config['depth'],
                loss_weight=self.config['loss_weight'],
                min_dif=self.config['min_dif'],
                max_dif=self.config['max_dif']
            )
            self.logger.info('Model instianted from: {:s}'.format(self.config['nn_module']))
        self.logger.info(model)
        
        #Load saved history if it already exists
        path_losses_csv = os.path.join(self.path_run_dir, 'losses.csv')
        if os.path.exists(path_losses_csv):
            fnetlogger = fnet.FnetLogger(path_losses_csv)
            self.logger.info('History loaded from: {:s}'.format(path_losses_csv))
        else:
            fnetlogger = fnet.FnetLogger(columns=['num_iter', 'loss_batch'])
    
        n_remaining_iterations = max(0, (self.config['training']['n_iter'] - model.count_iter))
        dataloader_train = self.get_dataloader(n_remaining_iterations)
        
        with open(os.path.join(self.path_run_dir, 'train_options.json'), 'w') as fo:
            json.dump(self.config, fo, indent=4, sort_keys=True)
        
        loss_train = 0
        pearson_train = 0
        
        for i, (signal, target, dapi_signal, dif_dapi_inf, _) in enumerate(dataloader_train, model.count_iter): #dna_channel

            if self.config['in_channels']==2:
                signal = torch.cat([signal, dapi_signal],  dim=1)
            loss_batch, pearson_batch = model.do_train_iter(signal, target, dif_dapi_inf)
                            
            fnetlogger.add({'num_iter': i + 1, 'loss_batch': loss_batch})
            loss_train += loss_batch
            pearson_train += pearson_batch
            
            if ((i + 1) % self.config['print_iter'] == 0) or ((i + 1) == self.config['training']['n_iter']):
                fnetlogger.to_csv(path_losses_csv)
                self.logger.info('loss log saved to: {:s}'.format(path_losses_csv))
                print('For {}/{} iterations, average training loss: {:.3f} and average Pearson Correlation Coefficient: {:3f}'.format(i, n_remaining_iterations, loss_train/self.config['print_iter'], pearson_train/self.config['print_iter']))
                loss_train = 0 
                pearson_train = 0

        self.logger.info('model saved to: {:s}'.format(path_model))
        model.save_state(path_model)