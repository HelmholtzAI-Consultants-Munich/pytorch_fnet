import os
import torch
import importlib
import pdb
from fnet.functions import pearsonr
import math


class Model(object):
    def __init__(
            self,
            nn_module = None,
            init_weights = True,
            lr = 0.001,
            criterion_fn = torch.nn.MSELoss, #torch.nn.torch.nn.SmoothL1Loss, #DiceLoss, #torch.nn.L1Loss, # #(reduction='none'), #CrossEntropyLoss,
            dropout=0,
            in_channels=1,
            out_channels=1,
            depth = 4,
            gpu_ids = -1,
            loss_weight = 0.5,
            min_dif=0.08,
            max_dif=0.35,
            nn_kwargs={}
    ):
        self.nn_module = nn_module
        self.nn_kwargs = nn_kwargs
        self.dropout = dropout
        self.input_channels = in_channels
        self.out_channels = out_channels
        self.init_weights = init_weights
        self.lr = lr
        self.criterion_fn = criterion_fn
        self.count_iter = 0
        self.gpu_ids = [gpu_ids] if isinstance(gpu_ids, int) else gpu_ids
        self.device = torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else torch.device('cpu')
        
        self.criterion = criterion_fn(reduction='none')
        self._init_model(nn_kwargs=self.nn_kwargs)
        
        self.loss_weight = loss_weight
        self.min_dif = min_dif
        self.max_dif = max_dif
        
    def _init_model(self, nn_kwargs={}):
        if self.nn_module is None:
            self.net = None
            return
        self.net = importlib.import_module('fnet.nn_modules.' + self.nn_module).Net(dropout_p=self.dropout, in_channels=self.input_channels, out_channels=self.out_channels)
        if self.init_weights:
            self.net.apply(_weights_init)
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def __str__(self):
        out_str = '{:s} | {:s} | iter: {:d}'.format(
            self.nn_module,
            str(self.nn_kwargs),
            self.count_iter,
        )
        return out_str

    def get_state(self):
        return dict(
            nn_module = self.nn_module,
            nn_kwargs = self.nn_kwargs,
            nn_state = self.net.state_dict(),
            optimizer_state = self.optimizer.state_dict(),
            count_iter = self.count_iter,
        )

    def to_gpu(self, gpu_ids):
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else torch.device('cpu')
        self.net.to(self.device)
        _set_gpu_recursive(self.optimizer.state, self.gpu_ids[0])  # this may not work in the future

    def save_state(self, path_save):
        curr_gpu_ids = self.gpu_ids
        dirname = os.path.dirname(path_save)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.to_gpu(-1)
        torch.save(self.get_state(), path_save)
        self.to_gpu(curr_gpu_ids)

    def load_state(self, path_load, gpu_ids=-1):
        state_dict = torch.load(path_load)
        self.nn_module = state_dict['nn_module']
        self.nn_kwargs = state_dict.get('nn_kwargs', {})
        self._init_model(nn_kwargs=self.nn_kwargs)
        self.net.load_state_dict(state_dict['nn_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.count_iter = state_dict['count_iter']
        self.to_gpu(gpu_ids)

    def do_train_iter(self, signal, target, dif_dapi_inf):
        self.net.train()
        
        dif_mean = torch.mean(dif_dapi_inf, dim=(1,2,3))

        for it,dif_mean_it in enumerate(dif_mean):
            if dif_mean_it>self.min_dif and dif_mean_it<self.max_dif:
                dif_mean[it] = self.loss_weight
            else:
                dif_mean[it] = 10-self.loss_weight #1-self.loss_weight
        dif_mean = dif_mean.clone().detach().requires_grad_(True).to(self.device)

        signal = signal.clone().detach().requires_grad_(True).to(self.device)
        target = target.clone().detach().requires_grad_(True).to(self.device)
        
        if len(self.gpu_ids) > 1:
            module = torch.nn.DataParallel(
                self.net,
                device_ids = self.gpu_ids,
            )
        else:
            module = self.net
        self.optimizer.zero_grad()
        output = module(signal)
        loss = self.criterion(output, target)
        # get mean per sample in batch
        loss = torch.mean(loss, dim=(1,2,3))
        loss = loss * dif_mean
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()
        self.count_iter += 1
        
        pearson_train = pearsonr(output, target)
        if math.isnan(pearson_train):
            pearson_train=0
            
        return loss.item(), pearson_train
    
    def predict(self, signal):
        #signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        signal = signal.to(device=self.device)
        
        if len(self.gpu_ids) > 1:
            module = torch.nn.DataParallel(
                self.net,
                device_ids = self.gpu_ids,
            )
        else:
            module = self.net
        module.eval()
        with torch.no_grad():
            prediction = module(signal) #.cpu()
        return prediction

def _weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0) 

def _set_gpu_recursive(var, gpu_id):
    """Moves Tensors nested in dict var to gpu_id.

    Modified from pytorch_integrated_cell.

    Parameters:
    var - (dict) keys are either Tensors or dicts
    gpu_id - (int) GPU onto which to move the Tensors
    """
    for key in var:
        if isinstance(var[key], dict):
            _set_gpu_recursive(var[key], gpu_id)
        elif torch.is_tensor(var[key]):
            if gpu_id == -1:
                var[key] = var[key].cpu()
            else:
                var[key] = var[key].cuda(gpu_id)