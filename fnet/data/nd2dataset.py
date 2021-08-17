import torch.utils.data
from fnet.data.fnetdataset import FnetDataset
import fnet.transforms as transforms

import pandas as pd

#import pdb
from nd2reader import ND2Reader


class ND2Dataset(FnetDataset):
    """Dataset for Tif files."""

    def __init__(self, dataframe: pd.DataFrame = None, path_csv: str = None, 
                    transform_source = [transforms.normalize],
                    transform_target = None):
        
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv)
        assert all(i in self.df.columns for i in ['signal_channel', 'target_channel'])
        
        self.transform_source = transform_source
        self.transform_target = transform_target

    def __getitem__(self, index):
     
        element = self.df.iloc[index, :]
   
        with ND2Reader(element['file']) as images:
            images.bundle_axes = 'zyx' # check what convention is for CZI dataset
            images.iter_axes ='t'
            images.default_coords['c'] = element['signal_channel'] # for brightfield
            im_out = [images[0]] # [:16,:,:]
            images.default_coords['c'] = element['target_channel'] # for cy3 RED (-2 # for DAPI)
            im_out.append(images[0]) # [:16,:,:]
            images.default_coords['c'] = element['target_channel']-2
            im_out.append(images[0])
            
        
        if self.transform_source is not None:
            for t in self.transform_source: 
                im_out[0] = t(im_out[0])

        if self.transform_target is not None and (len(im_out) > 1):
            for t in self.transform_target: 
                im_out[1] = t(im_out[1])
                im_out[2] = t(im_out[2])
        
        im_out = [torch.from_numpy(im).float() for im in im_out]
        
        #unsqueeze to make the first dimension be the channel dimension
        im_out = [torch.unsqueeze(im, 0) for im in im_out]
        
        return im_out
    
    def __len__(self):
        return len(self.df)

    def get_information(self, index):
        return self.df.iloc[index, :].to_dict()