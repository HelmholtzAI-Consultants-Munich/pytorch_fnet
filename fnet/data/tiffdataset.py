import torch.utils.data
from fnet.data.fnetdataset import FnetDataset
from fnet.data.tifreader import TifReader
import fnet.transforms as transforms
import pandas as pd
import numpy as np
import pdb

class TiffDataset(FnetDataset):
    """Dataset for Tif files."""

    def __init__(self, dataframe: pd.DataFrame = None, path_csv: str = None, 
                    transform_source = [transforms.normalize],
                    transform_target = None,
                    transform_thresh = None,
                    min_max_bright = [13677, 65535], 
                    min_max_dapi = [655, 18328],
                    min_max_infection = [379, 65535],
                    augmentations=False,
                    validation=False):
        
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv)
        assert all(i in self.df.columns for i in ['signal_channel', 'target_channel', 'dapi_channel'])
        
        if self.df.iloc[0,:]['target_channel'] is None:
            self.no_target = True
        else:
            self.no_target = False
            
        if self.df.iloc[0,:]['dapi_channel'] is None:
            self.no_dapi = True
        else:
            self.no_dapi = False
        
        self.transform_source = transform_source
        self.transform_target = transform_target
        self.transform_thresh = transform_thresh
        
        self.augmentations = augmentations
        if self.augmentations:
            self.random_affine = transforms.RandomAffine(degrees=30, translate=(0.3,0.3), scale=(0.8, 1.2))
            #transforms.RandomAffine(degrees=45, translate=(0.3,0.3), scale=(0.8, 1.2))
            #transforms.RandomAffine(degrees=10, translate=(0.2,0.2), scale=(0.9, 1.2))
        
        self.min_max_bright = min_max_bright
        self.min_max_dapi = min_max_dapi
        self.min_max_infection = min_max_infection
        self.which_dataset = self.df.iloc[0,:]['dataset']
        
    def get_dataset_info(self):
        return self.which_dataset

    def __getitem__(self, index):
        element = self.df.iloc[index, :]
        im_all_channels = TifReader(element['file']).get_image()        
        im_out = [im_all_channels[element['signal_channel']]]
        thresh_img = transforms.threshold(im_out[0], self.min_max_bright[0], self.min_max_bright[1])
        
        # if we are just doing inference there may be no target or dapi channel
        if not self.no_target:
            im_out.append(im_all_channels[element['target_channel']])
            patch_inf_bin = transforms.threshold(im_out[1], self.min_max_infection[0], self.min_max_infection[1]) # get inf. binary mask 
        if not self.no_dapi:
            im_out.append(im_all_channels[element['dapi_channel']]) 
            patch_dapi_bin = transforms.threshold(im_out[2], self.min_max_dapi[0], self.min_max_dapi[1]) # get DAPI binary mask 
        
        # if the channels exist get the difference of the masks - this should have much information for areas where e.g. there are cells which are not infected
        if not self.no_target and not self.no_dapi:
            dif = abs(patch_dapi_bin-patch_inf_bin)
            
        if self.transform_source is not None:
            for t in self.transform_source:
                im_out[0]=t(im_out[0])
        
        if self.transform_thresh is not None:
            for t in self.transform_thresh:
                thresh_img = t(thresh_img)
                dif = t(dif)
        # apply same transfroms to target and dapi
        if self.transform_target is not None and (len(im_out) > 1):
            for t in self.transform_target: 
                im_out[1] = t(im_out[1])
                im_out[2] = t(im_out[2])
        
        if not self.no_target and not self.no_dapi:
            im_out.append(dif)
        im_out.append(thresh_img)
        
        im_out = [torch.from_numpy(im).float() for im in im_out]
        #unsqueeze to make the first dimension be the channel dimension
        for idx, im in enumerate(im_out):
            if len(im.size())<3:
                im_out[idx] = torch.unsqueeze(im, 0) 
                
        if self.augmentations:
            im_out = self.random_affine(im_out)
            
        return im_out #im_out=[im_bright, im_inf, im_dapi, dif_dapi_inf,im_bright_thresh] or im_out=[im_bright,im_bright_thresh]
    
    def __len__(self):
        return len(self.df)

    def get_information(self, index):
        return self.df.iloc[index, :].to_dict()