from fnet.data.fnetdataset import FnetDataset
import numpy as np
import torch

from tqdm import tqdm

import pdb

import random

def decision(probability):
    return random.random() < probability


class BufferedPatchDataset(FnetDataset):
    """Dataset that provides chunks/patchs from another dataset."""

    def __init__(self, 
                 dataset,
                 patch_size, 
                 buffer_size = 1,
                 buffer_switch_frequency = 720, 
                 npatches = 100000,
                 verbose = False,
                 transform = None,
                 shuffle_images = True,
                 dim_squeeze = None,
                 threshold_backround = 0.2,
                 resampling_probability = 0.7
    ):
        self.threshold_back = threshold_backround
        self.resampling_prob = resampling_probability
        
        self.counter = 0
        
        self.dataset = dataset
        self.transform = transform
        
        self.buffer_switch_frequency = buffer_switch_frequency
        
        self.npatches = npatches
        
        self.buffer = list()
        
        self.verbose = verbose
        self.shuffle_images = shuffle_images
        self.dim_squeeze = dim_squeeze
        
        shuffed_data_order = np.arange(0, len(dataset))

        if self.shuffle_images:
            np.random.shuffle(shuffed_data_order)
        
        
        pbar = tqdm(range(0, buffer_size))
                       
        self.buffer_history = list()
            
        for i in pbar:
            #convert from a torch.Size object to a list
            if self.verbose: pbar.set_description("buffering images")

            datum_index = shuffed_data_order[i]
            datum = dataset[datum_index]
            
            datum_size = datum[0].size()
            
            self.buffer_history.append(datum_index)
            self.buffer.append(datum)
            
        self.remaining_to_be_in_buffer = shuffed_data_order[i+1:]
            
        self.patch_size = patch_size #[datum_size[0]] + patch_size # e.g. [1,16,128,128]
        self.img_size = datum_size[1], datum_size[2]
        
            
    def __len__(self):
        return self.npatches

    def __getitem__(self, index):
        self.counter +=1
        
        if (self.buffer_switch_frequency > 0) and (self.counter % self.buffer_switch_frequency == 0):
            if self.verbose: print("Inserting new item into buffer")
                
            self.insert_new_element_into_buffer()
        patch = self.get_random_patch()
        
        found_patch = False
        
        while not found_patch:
            patch = self.get_random_patch()
            if decision(self.resampling_prob): # will be true self.resampling_prob*100% of the time
                # if patch is more than threshold_back % foreground
                if torch.sum(patch[-1]) > self.threshold_back * np.prod(self.patch_size): 
                    # keep this patch
                    found_patch = True
                # else sample another patch
                else:
                    found_patch = False
            else:
                found_patch = True
       
        return patch
                       
    def insert_new_element_into_buffer(self):
        #sample with replacement
                       
        self.buffer.pop(0)
        
        if self.shuffle_images:
            
            if len(self.remaining_to_be_in_buffer) == 0:
                self.remaining_to_be_in_buffer = np.arange(0, len(self.dataset))
                np.random.shuffle(self.remaining_to_be_in_buffer)
            
            new_datum_index = self.remaining_to_be_in_buffer[0]
            self.remaining_to_be_in_buffer = self.remaining_to_be_in_buffer[1:]
            
        else:
            new_datum_index = self.buffer_history[-1]+1
            if new_datum_index == len(self.dataset):
                new_datum_index = 0
                             
        self.buffer_history.append(new_datum_index)
        self.buffer.append(self.dataset[new_datum_index])
        
        if self.verbose: print("Added item {0}".format(new_datum_index))


    def get_random_patch(self):
        
        buffer_index = np.random.randint(len(self.buffer))
                                   
        datum = self.buffer[buffer_index] # datum will be a list with input, target, dapi, threshold

        # start and end set at random loc each time
        starts_xy = [np.random.randint(0, d - p + 1) if d - p + 1 >= 1 else 0 for d, p in zip(self.img_size, self.patch_size)] # d in [1,20,1109,1107] p in [1,16,128,128]
        ends_xy = [x+y for x,y in zip(starts_xy,self.patch_size)]
        
        patch = []
        
        for d in datum:
            starts = np.array([0, starts_xy[0], starts_xy[1]])
            ends = np.array([d.size()[0], ends_xy[0], ends_xy[1]])
            #thank you Rory for this weird trick
            index = [slice(s, e) for s,e in zip(starts,ends)]
            temp = d[tuple(index)]
            patch.append(temp)

        #patch = [d[tuple(index)] for d in datum]
        
        if self.dim_squeeze is not None:
            patch = [torch.squeeze(d, self.dim_squeeze) for d in patch]

        return patch
    
    def get_buffer_history(self):
        return self.buffer_history
    
def _test():
    # dims_chunk = (2,3,4)
    dims_chunk = (4,5)
    ds_test = ChunkDatasetDummy(
        None,
        dims_chunk = dims_chunk,
    )
    print('Dataset len', len(ds_test))
    for i in range(3):
        print('***** {} *****'.format(i))
        element = ds_test[i]
        print(element[0])
        print(element[1])
    
if __name__ == '__main__':
    _test()