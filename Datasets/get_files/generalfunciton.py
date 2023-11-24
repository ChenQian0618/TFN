import numpy as np
# import pywt
from scipy import signal
import pandas as pd
from torch.utils.data import Dataset
import random

# random.seed(999)
seed = 999
np.random.seed(seed)
random.seed(seed)

def add_noise(sig,SNR): # add noise to sig
    noise = np.random.randn(*sig.shape)
    noise_var = sig.var() / np.power(10,(SNR/20))
    noise = noise /noise.std() * np.sqrt(noise_var)
    return sig + noise


def data_transforms1d(aug,dataset_type="train", normlize_type="1-1",aug_flag = False):
    if aug_flag:
        train_trans = aug.Compose([
            aug.to_numpy(),
            aug.Normalize(normlize_type),
            aug.RandomAddGaussian(),
            aug.RandomScale(),
            aug.RandomStretch(),
            aug.RandomCrop(),
            aug.Retype()
        ])
    else:
        train_trans = aug.Compose([
            aug.to_numpy(),
            aug.Normalize(normlize_type),
            aug.Retype()
        ])
    transforms = {
        'train': train_trans,
        'val': aug.Compose([
            aug.to_numpy(),
            aug.Normalize(normlize_type),
            aug.Retype()
        ])
    }
    return transforms[dataset_type]

class sig_process(object):
    nperseg = 30
    adjust_flag = False
    def __init__(self):
        super(sig_process,self).__init__()

    @classmethod
    def time(cls,x):
        return x

    @classmethod
    def fft(cls,x):
        x = x - np.mean(x)
        x = np.fft.fft(x)
        x = np.abs(x) / len(x)
        x = x[range(int(x.shape[0] / 2))]
        x[1:-1] = 2*x[1:-1]
        return x

    @classmethod
    def slice(cls,x):
        w = int(np.sqrt(len(x)))
        img = x[:w**2].reshape(w,w)
        return img

    @classmethod
    def STFT(cls,x,verbose=False):
        while not cls.adjust_flag:
            _,_, Zxx = signal.stft(x, nperseg=cls.nperseg)
            if abs(Zxx.shape[0] - Zxx.shape[1]) < 2:
                cls.adjust_flag = True
            elif Zxx.shape[0] > Zxx.shape[1]:
                cls.nperseg -= 1
            else:
                cls.nperseg += 1
        f, t, Zxx = signal.stft(x, nperseg=cls.nperseg)
        img = np.abs(Zxx) / len(Zxx)
        if verbose:
            return f, t, img
        else:
            return img

    @classmethod
    def STFT8(cls,x,Nc=8):
        f, t, Zxx = signal.stft(x, nperseg=Nc*2-1,noverlap=Nc*2-2)
        img = np.abs(Zxx) / len(Zxx)
        return img
    @classmethod
    def STFT16(cls,x):
        return sig_process.STFT8(x,Nc=16)

    @classmethod
    def STFT32(cls,x):
        return sig_process.STFT8(x, Nc=32)

    @classmethod
    def STFT64(cls,x):
        return sig_process.STFT8(x, Nc=64)

    @classmethod
    def STFT128(cls,x):
        return sig_process.STFT8(x, Nc=128)

    @classmethod
    def mySTFT(cls,x,verbose=False,nperseg=256,noverlap = None):
        if not noverlap:
            noverlap = nperseg//2
        f, t, Zxx = signal.stft(x, nperseg=nperseg,noverlap=noverlap)
        img = np.abs(Zxx) / len(Zxx)
        if verbose:
            return f, t, img
        else:
            return img

def balance_label(df,labelcol = 'label', number=None):
    value_count = df[labelcol].value_counts()
    if number == None:
        number = value_count.to_numpy().min()
    new_df = pd.concat([df.iloc[df.index[df[labelcol] == lab].to_numpy()[:number]] for lab in value_count.index.sort_values().to_list()])
    return new_df