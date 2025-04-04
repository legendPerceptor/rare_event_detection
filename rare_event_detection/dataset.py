import numpy as np
import torchvision
import torch
import h5py
import sys
from torch.utils.data import Dataset
from .utility import ge_raw2array_fabio, ge_raw2array, ge_raw2patch
from pathlib import Path

def data_transforms(psz):
    # get a set of data augmentation transformations 
    data_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                          torchvision.transforms.RandomVerticalFlip(p=0.5),
                                          torchvision.transforms.RandomErasing(p=0.2, scale=(1/psz, 4/psz), ratio=(0.5, 2)),
                                          torchvision.transforms.RandomRotation(degrees=180)])
    return data_transforms

class BraggDatasetMIDAS(Dataset):
    def __init__(self, ifn, psz=-1, train=True, tv_split=1):
        self.transform = data_transforms(psz)
        with h5py.File(ifn, 'r') as h5:
            train_N = int(tv_split * h5['patch'].shape[0])
            if train:
                sidx, eidx = 0, train_N
            else:
                sidx, eidx = train_N, None
            patches = h5['patch'][sidx:eidx]
            peakLoc = h5['peakLoc'][sidx:eidx]

        nPeaks = np.array([_pl.shape[0]//2 for _pl in peakLoc])
        sel_peak_patches = patches[nPeaks >= 1]
        _min = sel_peak_patches.min(axis=(1, 2))[:,np.newaxis,np.newaxis]
        _max = sel_peak_patches.max(axis=(1, 2))[:,np.newaxis,np.newaxis] + 1e-10
        self.patches = ((sel_peak_patches - _min) / (_max - _min)).astype(np.float32)[:,np.newaxis]

        self.fpsz    = self.patches.shape[-1]

        self.psz = self.fpsz if psz <= 0 else psz

        if self.psz > self.fpsz:
            sys.exit(f"It's impossible to make patch with ({self.psz}, {self.psz}) from ({self.fpsz}, {self.fpsz})")

    def __getitem__(self, idx):
        sr  = np.random.randint(0, self.fpsz-self.psz+1)
        sc  = np.random.randint(0, self.fpsz-self.psz+1)
        c_patch = self.patches[idx, :, sr:(sr+self.psz), sc:(sc+self.psz)]

        c_patch = torch.from_numpy(c_patch)
        view1 = c_patch
        view2 = self.transform(c_patch)

        return view1, view2

    def __len__(self):
        return self.patches.shape[0]



class BraggDataset(Dataset):
    def __init__(self, irawt, irawd, thold, data_folder=Path("."), psz=-1, train=True, tv_split=1):
        self.transform = data_transforms(psz)

        # read the raw scan and dark file and output a h5 file for later processing
        if irawd != "default_dark":
            print(f"Reading dark file from {irawd} ... ")
            dark = ge_raw2array_fabio(irawd, skip_frm=0).mean(axis=0).astype(np.float32)
            print(f"Done with reading dark file from {irawd}")
        else:
            print(f"no dark file provided, skip dark file reading")

        outFile = data_folder / "test.h5"
        print(f"Reading training file from {irawt} ... ")
        if irawd != "default_dark":
            ge_raw2patch(gefname=irawt, ofn=outFile, dark=dark, thold=thold, psz=15, skip_frm=0, \
                         min_intensity=0, max_r=None)
        else:
            ge_raw2patch(gefname=irawt, ofn=outFile, dark=irawd, thold=thold, psz=15, skip_frm=0, \
                         min_intensity=0, max_r=None)
        print(f"Done with reading training file from {irawt}")

        #with h5py.File('../sam9_all_init.edf.h5', 'r') as h5:
        with h5py.File(outFile, 'r') as h5:
            train_N = int(tv_split * h5['patch'].shape[0])
            if train:
                sidx, eidx = 0, train_N
            else:
                sidx, eidx = train_N, None
            patches = h5['patch'][sidx:eidx]

        _min = patches.min(axis=(1, 2))[:,np.newaxis,np.newaxis]
        _max = patches.max(axis=(1, 2))[:,np.newaxis,np.newaxis]
        self.patches = ((patches - _min) / (_max - _min)).astype(np.float32)[:,np.newaxis]

        self.fpsz    = self.patches.shape[-1]

        self.psz = self.fpsz if psz <= 0 else psz

        if self.psz > self.fpsz:
            sys.exit(f"It's impossible to make patch with ({self.psz}, {self.psz}) from ({self.fpsz}, {self.fpsz})")

    def __getitem__(self, idx):
        sr  = np.random.randint(0, self.fpsz-self.psz+1)
        sc  = np.random.randint(0, self.fpsz-self.psz+1)
        c_patch = self.patches[idx, :, sr:(sr+self.psz), sc:(sc+self.psz)]

        c_patch = torch.from_numpy(c_patch)
        view1 = c_patch
        view2 = self.transform(c_patch)

        return view1, view2

    def __len__(self):
        return self.patches.shape[0]
