from fnet.data.fnetdataset import FnetDataset
import numpy as np
import torch

from tqdm import tqdm

import pdb


class AllPatchesDataset(FnetDataset):
    """Dataset that provides chunks/patchs from another dataset."""

    def __init__(self, 
                 dataset,
                 patch_size, 
                 buffer_size = 1,
                 buffer_switch_frequency = 7200000, 
                 verbose = False,
                 transform = None,
                 dim_squeeze = None,
    ):            
        self.counter = 0
        
        self.dataset = dataset
        self.transform = transform
        self.buffer_switch_frequency = buffer_switch_frequency
        
        self.buffer = list()
        
        self.verbose = verbose
        self.dim_squeeze = dim_squeeze
        shuffed_data_order = np.arange(0, len(dataset))
        
        pbar = tqdm(range(0, buffer_size))
                       
        self.buffer_history = list()
                    
        for i in pbar:
            #convert from a torch.Size object to a list
            if self.verbose: pbar.set_description("buffering images")
            datum_index = shuffed_data_order[i]
            datum = dataset[datum_index]
            self.datum_size = [datum[0].size()[1], datum[0].size()[2]]
            self.buffer_history.append(datum_index)
            self.buffer.append(datum)

        self.remaining_to_be_in_buffer = shuffed_data_order[i+1:]
        # if patch has not be defined take the whole image(padded to square)
        if len(patch_size)==0:
            self.patch_size = [max(self.datum_size[0]), max(self.datum_size[1])]
        else:
            self.patch_size = patch_size
        self.img_patch_ratios = [x/y for x,y in zip(self.datum_size, self.patch_size)]
        
        self.patches_per_img = round(np.prod(self.img_patch_ratios))
        print('Patches per img', self.patches_per_img)
        self.npatches = len(self.dataset) * self.patches_per_img
        print('Total number of patches', self.npatches)
        
        self.which_dataset = self.dataset.get_dataset_info()
            
    def __len__(self):
        return self.npatches

    def __getitem__(self, index):
        
        self.counter +=1
        
        # figure out later how to adapt
        if (self.buffer_switch_frequency > 0) and (self.counter % self.buffer_switch_frequency == 0) and index!=0:
            if self.verbose: print("Inserting new item into buffer")
                
            self.insert_new_element_into_buffer()
        
        patch, is_last = self.get_patch(index)
        return {'patch': patch, 'is_last': is_last}
    
    def get_current_image_target(self, index):
        
        image_index = index // self.patches_per_img
        if image_index >= len(self.buffer):
            image_index = image_index%len(self.buffer)
        datum = self.buffer[image_index]
        if len(datum)>2:
            return datum[0], datum[1]
        else:
            return datum[0], None
    
    
    def repatch(self, pred_list):
        image = torch.zeros((pred_list[0].size()[1],self.datum_size[0], self.datum_size[1]))
        for patch_index, patch in enumerate(pred_list):
            patch_i = patch_index // self.img_patch_ratios[1]
            patch_j = patch_index - patch_i * self.img_patch_ratios[1]
            i, j = int(patch_i * self.patch_size[0]), int(patch_j * self.patch_size[1])
            i_p, j_p = int((patch_i+1) * self.patch_size[0]), int((patch_j+1) * self.patch_size[1])
            image[:,i:i_p, j:j_p] = patch.squeeze()
            
        return image
            
    def get_information(self, index):
        return self.dataset.get_information(index // self.patches_per_img)
    
    def get_patch(self, index):
        
        # only works for 2d now
        image_index = index // self.patches_per_img
        if image_index >= len(self.buffer):
            image_index = image_index%len(self.buffer)
        
        datum = self.buffer[image_index]
        patch_index = index % self.patches_per_img
        patch_i = patch_index // self.img_patch_ratios[1]
        patch_j = patch_index - patch_i * self.img_patch_ratios[1]
        i, j = int(patch_i * self.patch_size[0]), int(patch_j * self.patch_size[1])
        starts_xy = [i, j]
        i_p, j_p = int((patch_i+1) * self.patch_size[0]), int((patch_j+1) * self.patch_size[1])
        ends_xy = [i_p, j_p]
        
        patch = []
        
        for d in datum:
            starts = np.array([0, starts_xy[0], starts_xy[1]])
            ends = np.array([d.size()[0], ends_xy[0], ends_xy[1]])
            #thank you Rory for this weird trick
            index = [slice(s, e) for s,e in zip(starts,ends)]
            temp = d[tuple(index)]
            patch.append(temp)
        
        is_last= False
        if patch_index == self.patches_per_img - 1:
            is_last=True
            
        return patch, is_last
        
                       
    def insert_new_element_into_buffer(self):
        #sample with replacement
                       
        self.buffer.pop(0)
        
        new_datum_index = self.buffer_history[-1]+1
        if new_datum_index == len(self.dataset):
            new_datum_index = 0
                             
        self.buffer_history.append(new_datum_index)
        self.buffer.append(self.dataset[new_datum_index])
        
        if self.verbose: print("Added item {0}".format(new_datum_index))


    def get_random_patch(self):
        
        buffer_index = np.random.randint(len(self.buffer))
                                   
        datum = self.buffer[buffer_index]
        
        # start and end set at random loc each time
        starts_xy = np.array([np.random.randint(0, d - p + 1) if d - p + 1 >= 1 else 0 for d, p in zip(self.datum_size, self.patch_size)]) # d in [20,1109,1107] p in [16,128,128]
        ends_xy = starts + np.array(self.patch_size)
        patch = []
        
        for d in datum:
            starts = np.array([0, starts_xy[0], starts_xy[1]])
            ends = np.array([d.size()[0], ends_xy[0], ends_xy[1]])
            #thank you Rory for this weird trick
            index = [slice(s, e) for s,e in zip(starts,ends)]
            temp = d[tuple(index)]
            patch.append(temp)
        
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
