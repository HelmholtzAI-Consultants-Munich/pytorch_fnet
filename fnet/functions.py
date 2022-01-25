import importlib
import json
import os
import pdb
import sys
import fnet
import pandas as pd
import tifffile
import numpy as np
from fnet.transforms import normalize

def pearson_loss(x, y):
    #x = output
    #y = target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return cost

# code retrieved on 21.05.21 from: https://github.com/pytorch/pytorch/issues/1254
def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    
    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    
    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    
    """
    x = x.detach().cpu().numpy().flatten() #pred
    y = y.detach().cpu().numpy().flatten() #target
    pearson_img = np.corrcoef(x,y)
    r_val = pearson_img[0,1]

    return r_val


def load_model(path_model, gpu_ids=0, module='fnet_model', in_channels=1, out_channels=1):
    module_fnet_model = importlib.import_module('fnet.' + module)
    if os.path.isdir(path_model):
        path_model = os.path.join(path_model, 'model.p')
    model = module_fnet_model.Model(in_channels=in_channels, out_channels=out_channels)
    model.load_state(path_model, gpu_ids=gpu_ids)
    return model

def load_model_from_dir(path_model_dir, gpu_ids=0, in_channels=1, out_channels=1):
    assert os.path.isdir(path_model_dir)
    path_model_state = os.path.join(path_model_dir, 'model.p')
    model = fnet.fnet_model.Model(in_channels=in_channels, out_channels=out_channels)
    model.load_state(path_model_state, gpu_ids=gpu_ids)
    return model

def compute_dataset_min_max_ranges(train_path, val_path=None, norm=False):
    
    df_train = pd.read_csv(train_path)
    if val_path is not None:
        df_val = pd.read_csv(val_path)
        df=pd.concat([df_train, df_val])
    else:
        df=df_train
    
    min_bright=[]
    max_bright =[]
    min_inf = []
    max_inf = []
    min_dapi = []
    max_dapi = []
    
    if df.iloc[0,:]['target_channel'] is None: no_target = True
    else: no_target = False
        
    if df.iloc[0,:]['dapi_channel'] is None: no_dapi = True
    else: no_dapi = False
        
    for index in range(len(df)):
        element=df.iloc[index, :]
        image = tifffile.imread(element['file'])
        
        if not no_target:
            image_infection = image[element['target_channel'],:,:]
            if norm:
                min_inf.append(np.min(normalize(image_infection)))
                max_inf.append(np.max(normalize(image_infection)))
            else:
                min_inf.append(np.min(image_infection))
                max_inf.append(np.max(image_infection))
        
        if not no_dapi:
            image_dapi = image[element['dapi_channel'],:,:]
            min_dapi.append(np.min(image_dapi))
            max_dapi.append(np.max(image_dapi))
            
        image_bright = image[element['signal_channel'],:,:] 
        if norm:
            image_bright = normalize(image_bright)
        min_bright.append(np.min(image_bright))
        max_bright.append(np.max(image_bright))
    
    min_inf = np.min(np.array(min_inf)) if not no_target else None
    max_inf = np.max(np.array(max_inf)) if not no_target else None

    min_dapi = np.min(np.array(min_dapi)) if not no_dapi else None
    max_dapi = np.max(np.array(max_dapi)) if not no_dapi else None
    
    min_bright = np.min(np.array(min_bright))
    max_bright = np.max(np.array(max_bright))

    return [min_bright, max_bright], [min_inf, max_inf], [min_dapi, max_dapi]        