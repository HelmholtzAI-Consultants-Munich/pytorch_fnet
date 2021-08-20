import argparse
import fnet.data
from fnet.functions import pearsonr, compute_dataset_min_max_ranges
import importlib
import json
import numpy as np
import os
import pandas as pd
import tifffile
import time
import torch
import warnings
import pdb

def set_warnings():
    warnings.filterwarnings('ignore', message='.*zoom().*')
    warnings.filterwarnings('ignore', message='.*end of stream*')
    warnings.filterwarnings('ignore', message='.*multiple of element size.*')

def get_dataset(config, propper):

    path_csvs = os.path.join(config['data_path'], 'csvs')
    path_dataset_csv = os.path.join(path_csvs, ('.').join([config['dataset'],'csv'])) 
    path_test_dataset_csv = os.path.join(config['data_path'], 'csvs', config['dataset'], 'test.csv') #path_dataset_test_csv: ["./data/csvs/new_exp_eg/test.csv"]
    min_max_bright, min_max_infection, min_max_dapi = compute_dataset_min_max_ranges(path_dataset_csv)
    min_max_bright_norm, _, _ = compute_dataset_min_max_ranges(path_dataset_csv, norm=True)

    transform_signal=[]
    for t in config['preprocessing']['transform_signal']:
        if t=='fnet.transforms.AdaptRange':
            t = 'fnet.transforms.AdaptRange({:f},{:f})'.format(min_max_bright_norm[0], min_max_bright_norm[1])  
        transform_signal.append(eval(t))
    transform_target = [eval(t) for t in config['preprocessing']['transform_target']]
    transform_thresh = []

    transform_signal.append(propper)
    transform_target.append(propper) 
    transform_thresh.append(propper)
        
    ds = getattr(fnet.data, config['class_dataset'])(
        path_csv = path_test_dataset_csv,
        transform_source = transform_signal,
        transform_target = transform_target,
        transform_thresh = transform_thresh,
        min_max_bright = min_max_bright, 
        min_max_dapi = min_max_dapi, 
        min_max_infection = min_max_infection)

    
    ds_patch = fnet.data.AllPatchesDataset(
        dataset = ds,
        patch_size = [2048, 2048], #config['patch_size'],
        buffer_size = 1, #opts.n_images,
        buffer_switch_frequency = 1, #-1,
        verbose = False
    )
    
    dataloader = torch.utils.data.DataLoader(
        ds_patch, #ds
        batch_size = 1, #['batch_size'],
    )
    return dataloader, ds_patch


def save_tiff_and_log(tag, ar, path_tiff_dir, entry, path_log_dir):
    if not os.path.exists(path_tiff_dir):
        os.makedirs(path_tiff_dir)
    path_tiff = os.path.join(path_tiff_dir, '{:s}.tiff'.format(tag))
    print('saved:', path_tiff)
    tifffile.imsave(path_tiff, ar)
    entry['path_' + tag] = os.path.relpath(path_tiff, path_log_dir)

def get_prediction_entry(ds, index):
    info = ds.get_information(index)
    # In the case where 'path_signal', 'path_target' keys exist in dataset information,
    # replace with 'path_signal_dataset', 'path_target_dataset' to avoid confusion with
    # predict.py's 'path_signal' and 'path_target'.
    if isinstance(info, dict):
        if 'path_signal' in info:
            info['path_signal_dataset'] = info.pop('path_signal')
        if 'path_target' in info:
            info['path_target_dataset'] = info.pop('path_target')
        return info
    if isinstance(info, str):
        return {'information': info}
    raise AttributeError

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',type=str, help='config dictionary')
    opts = parser.parse_args()
    with open(opts.config, "r") as fp:
        config = json.load(fp)

    if config['class_dataset'] == 'TiffDataset':
        if config['prediction']['propper_kwargs'] == '-':
            config['prediction']['propper_kwargs']['n_max_pixels'] = 6000000
    propper = fnet.transforms.Propper(action='+') #, padding=[4,0])#**opts.propper_kwargs)
    print(propper)
    model = None
    dataset, ds = get_dataset(config, propper)
    entries = []
    predicted_patches = []
    
    indices = len(dataset) if config['prediction']['n_images'] < 0 else min(config['prediction']['n_images'], len(dataset))
    
    pearson=dict()
    for key in config['path_run_dir']:
        pearson[key] = dict()
    
    for idx, sample in enumerate(dataset):
        if idx==indices:
            break
        patch, is_last = sample['patch'], sample['is_last']
        signal = patch[0]
        target = patch[1] if (len(patch) > 2) else None

        for path_model_dir, path_run_dir in zip(config['path_model_dir'], config['path_run_dir']):
            path_save_dir = os.path.join(path_run_dir, 'results')
            
            if os.path.exists(os.path.join(path_save_dir, 'pearson.json')):
                continue
            
            if (path_model_dir is not None) and (model is None or len(config['path_model_dir']) > 1):
                model = fnet.load_model(path_model_dir, config['gpu_ids'], module=config['module_fnet_model'], in_channels=config['in_channels'], out_channels=config['out_channels'])
                print(model)
                name_model = os.path.basename(path_model_dir)
            if config['in_channels']==2:
                signal = torch.cat([signal, patch[2]],  dim=1)
            prediction = model.predict(signal) if model is not None else None

            if is_last:
                predicted_patches.append(prediction)
                entry = get_prediction_entry(ds, idx)
                filename=entry['file'].split('/')[-1].split('.')[0]
                prediction = ds.repatch(predicted_patches)
                # save images
                path_tiff_dir = os.path.join(path_save_dir, filename)
                signal_img, target_img = ds.get_current_image_target(idx)
                
                if target is not None:
                    pearson[path_run_dir][entry['file']] = pearsonr(prediction, target)
                    target_img = target_img.numpy()[0, ]
                    
                signal_img = signal_img.numpy()[0, ]
    
                if not config['prediction']['no_signal']:
                    save_tiff_and_log('signal', signal_img, path_tiff_dir, entry, path_save_dir)
                if not config['prediction']['no_target'] and target is not None:
                    save_tiff_and_log('target', target_img.astype(np.float32), path_tiff_dir, entry, path_save_dir)
                
                prediction = prediction.squeeze().numpy()#[0, ]
                if not config['prediction']['no_prediction'] and prediction is not None:
                    save_tiff_and_log('prediction_{:s}'.format(name_model), prediction.astype(np.float32), path_tiff_dir, entry, path_save_dir)
                if not config['prediction']['no_prediction_unpropped']:
                    #ar_pred_unpropped = propper.undo_last(prediction.numpy()[0, ])
                    ar_pred_unpropped = propper.undo_last(prediction)
                    save_tiff_and_log('prediction_{:s}_unpropped'.format(name_model), ar_pred_unpropped.astype(np.float32), path_tiff_dir, entry, path_save_dir)
                
                entries.append(entry)
                predicted_patches = []
            else:
                predicted_patches.append(prediction)

        pd.DataFrame(entries).to_csv(os.path.join(path_save_dir, 'predictions.csv'), index=False)

    if config['prediction']['return_score']:
        for path_run_dir in config['path_run_dir']:
            if not os.path.exists(os.path.join(path_run_dir, 'results', 'pearson.json')):
                with open(os.path.join(path_run_dir, 'results', 'pearson.json'), 'w') as fo:
                    json.dump(pearson[path_run_dir], fo)

if __name__ == '__main__':

    main()