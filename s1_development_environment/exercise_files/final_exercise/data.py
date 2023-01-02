import torch
import numpy as np

def mnist():
    # exchange with the corrupted mnist dataset
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784)
    tr0 = np.load('data/corruptmnist/train_0.npz')
    tr1 = np.load('data/corruptmnist/train_1.npz')
    tr2 = np.load('data/corruptmnist/train_2.npz')
    tr3 = np.load('data/corruptmnist/train_3.npz')
    tr4 = np.load('data/corruptmnist/train_4.npz')
    trimagestmp = np.concatenate((tr0['images'],tr1['images'],tr2['images'],tr3['images'],tr4['images']))

    mean_px = trimagestmp.mean().astype(np.float32)
    std_px = trimagestmp.std().astype(np.float32)
    trimagestmp = (trimagestmp - mean_px)/(std_px)

    trimages = torch.from_numpy(trimagestmp)
    trlabelstmp = np.concatenate((tr0['labels'],tr1['labels'],tr2['labels'],tr3['labels'],tr4['labels']))
    trlabels = torch.from_numpy(trlabelstmp)
    loadtest = np.load('data/corruptmnist/test.npz')
    tsimagestmp = loadtest['images']
    tsimages = torch.from_numpy(tsimagestmp)

    mean_px = tsimagestmp.mean().astype(np.float32)
    std_px = tsimagestmp.std().astype(np.float32)
    tsimagestmp = (tsimagestmp - mean_px)/(std_px)
    
    tslabelstmp = loadtest['labels']
    tslabels = torch.from_numpy(tslabelstmp)
    # train = trimages.type(torch.float32),trlabels.type(torch.float32)
    # test = tsimages.type(torch.float32),tslabels.type(torch.float32)
    #train = trimages,trlabels
    train = [[trimages[i],trlabels[i]] for i in range(len(trimages))]
    #test = tsimages,tslabels
    test = [[tsimages[i],tslabels[i]] for i in range(len(tsimages))]
    return train, test




