import torch.utils.data

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lens = [len(d) for d in self.datasets] 

    def __getitem__(self, i):
        running_len = 0
        index_d = -1
        for idx, len_d in enumerate(self.lens):
            if i > running_len:
                running_len += len_d
            elif i==running_len:
                index_d = idx
                break
            else:
                index_d = idx-1
                break
        return self.datasets[index_d][i-running_len]

    def __len__(self):
        return sum(self.lens)

'''
train_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 datasets.ImageFolder(traindir_A),
                 datasets.ImageFolder(traindir_B)
             ),
             batch_size=args.batch_size, shuffle=True,
             num_workers=args.workers, pin_memory=True)
'''
