"""
Created on 2023/11/23
@author: Chen Qian
@e-mail: chenqian2020@sjtu.edu.cn
"""

from torch import nn
import torch
import numpy as np
import random
import os
import torch.nn.init as init
import math
import torch.nn.functional as F

# random seed
seed = 999
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# set the min and max frequency of the kernel function
fmin = 0.03
fmax = 0.45
random_scale = 2e-3


# traditional convolution layer with Real-Imaginary mechanism
class BaseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.stride = stride
        self.padding = padding

        self.phases = ['real', 'imag']
        self.weight = torch.Tensor(len(self.phases), out_channels, in_channels, self.kernel_size)
        if bias:
            self.bias = torch.Tensor(len(self.phases), out_channels)
        else:
            self.bias = None

        for phase in self.phases:
            self.weight[self.phases.index(phase)] = torch.Tensor(out_channels, in_channels, self.kernel_size)
            init.kaiming_uniform_(self.weight[self.phases.index(phase)], a=math.sqrt(5))  # initial weight
            if bias:
                self.bias[self.phases.index(phase)] = torch.Tensor(out_channels)
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[self.phases.index(phase)])  # initial bias
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias[self.phases.index(phase)], -bound, bound)

        if self.__class__.__name__ == 'BaseConv1d':
            self.weight = torch.nn.Parameter(self.weight)
            if self.bias is not None:
                self.bias = torch.nn.Parameter(self.bias)

    def forward(self, input):
        result = {}
        # the output of the convolution layer is the square root of the sum of the squares of the real and imaginary parts
        for phase in self.phases:
            if self.bias is None:
                result[phase] = F.conv1d(input, self.weight[self.phases.index(phase)],
                                         stride=self.stride, padding=self.padding)
            else:
                result[phase] = F.conv1d(input, self.weight[self.phases.index(phase)],
                                         bias=self.bias[self.phases.index(phase)],
                                         stride=self.stride, padding=self.padding)
        output = torch.sqrt(result[self.phases[0]].pow(2) + result[self.phases[1]].pow(2))
        return output


# prepare for kernel function by adding limits and rewrite forward function
class BaseFuncConv1d(BaseConv1d):
    def __init__(self, *pargs, **kwargs):
        kwargs_new = {}
        for k in kwargs.keys():
            if k in ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias']:
                kwargs_new[k] = kwargs[k]
        super().__init__(*pargs, **kwargs_new)
        if self.__class__.__name__ == 'BaseFuncConv1d':
            self.weight = torch.nn.Parameter(self.weight)
            if self.bias is not None:
                self.bias = torch.nn.Parameter(self.bias)
            self.superparams = self.weight

    def _clamp_parameters(self):
        with torch.no_grad():
            for i in range(len(self.params_bound)):
                self.superparams.data[:, :, i].clamp_(self.params_bound[i][0], self.params_bound[i][1])

    def WeightForward(self):
        if self.clamp_flag:
            self._clamp_parameters()
        l00 = []
        for phase in self.phases:
            l0 = []
            for i in range(self.superparams.shape[0]):
                l1 = []
                for j in range(self.superparams.shape[1]):
                    l1.append(self.weightforward(self.kernel_size, self.superparams[i, j], phase).unsqueeze(0))
                l0.append(torch.vstack(l1).unsqueeze(0))
            l00.append(torch.vstack(l0).unsqueeze(0))
            self.weight = torch.vstack(l00)

    def forward(self, input):
        if self.__class__.__name__ != 'BaseFuncConv1d':
            self.WeightForward()
        return super().forward(input)


# kernel functions have differences in the weightforward function and the _reset_parameters function
class TFconv_STTF(BaseFuncConv1d):  # kernel_size = out_channels * 2 - 1
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False,
                 clamp_flag=True, params_bound=((0, 0.5),)):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.clamp_flag = clamp_flag
        self.params_bound = params_bound
        self.superparams = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, len(params_bound)))
        self._reset_parameters()
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias)

    def _reset_parameters(self):
        with torch.no_grad():
            shape = self.superparams.data[:, :, 0].shape
            temp0 = (torch.linspace(fmin, fmax, shape.numel())).reshape(shape)
            self.superparams.data[:, :, 0] = temp0
            self.WeightForward()

    def weightforward(self, lens, params, phase):
        if isinstance(lens, torch.Tensor):
            lens = int(lens.item())
        T = torch.arange(-(lens // 2), lens - (lens // 2)).to(params.device)
        sigma = torch.tensor(0.52).to(params.device)
        if self.phases.index(phase) == 0:
            result = torch.exp(-(T / (lens // 2)).pow(2) / sigma.pow(2) / 2) * torch.cos(2 * math.pi * params[0] * T)
        else:
            result = torch.exp(-(T / (lens // 2)).pow(2) / sigma.pow(2) / 2) * torch.sin(2 * math.pi * params[0] * T)
        return result


class TFconv_Chirplet(BaseFuncConv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False,
                 clamp_flag=True):
        max_t = kernel_size // 2
        params_bound = ((0, 0.5), (-1 / max_t ** 2, 1 / max_t ** 2))
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.clamp_flag = clamp_flag
        self.params_bound = params_bound
        self.superparams = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, len(params_bound)))
        self._reset_parameters()
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias)

    def _reset_parameters(self):
        with torch.no_grad():
            shape = self.superparams.data[:, :, 0].shape
            temp0 = (torch.linspace(fmin, fmax, shape.numel())).reshape(shape)
            self.superparams.data[:, :, 0] = temp0
            self.superparams.data[:, :, 1].normal_(0, 1e-4)
            self.WeightForward()

    def weightforward(self, lens, params, phase):
        if isinstance(lens, torch.Tensor):
            lens = int(lens.item())
        T = torch.arange(-(lens // 2), lens - (lens // 2)).to(params.device)
        sigma = torch.tensor(0.52).to(params.device)
        if self.phases.index(phase) == 0:
            result = torch.exp(-(T / (lens // 2)).pow(2) / sigma.pow(2) / 2) * torch.cos(
                2 * math.pi * (params[1] / 2 * T.pow(2) + params[0] * T))
        else:
            result = torch.exp(-(T / (lens // 2)).pow(2) / sigma.pow(2) / 2) * torch.sin(
                2 * math.pi * (params[1] / 2 * T.pow(2) + params[0] * T))
        return result


class TFconv_Morlet(BaseFuncConv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False,
                 clamp_flag=True, params_bound=((0.4, 10),)):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.clamp_flag = clamp_flag
        self.params_bound = params_bound
        self.superparams = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, len(params_bound)))
        self._reset_parameters()
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias)

    def _reset_parameters(self):
        with torch.no_grad():
            shape = self.superparams.data[:, :, 0].shape
            temp_f = torch.pow(torch.tensor(2), (torch.linspace(np.log2(fmin), np.log2(fmax), shape.numel())))
            temp_s = (0.2 / temp_f).reshape(shape)
            self.superparams.data[:, :, 0] = temp_s
            self.WeightForward()

    def weightforward(self, lens, params, phase):
        if isinstance(lens, torch.Tensor):
            lens = int(lens.item())
        T = torch.arange(-(lens // 2), lens - (lens // 2)).to(params.device)
        T = T / params[0]
        sigma = torch.tensor(0.6).to(params.device)
        fc_len = int(self.out_channels)
        if self.phases.index(phase) == 0:
            result = params[0].pow(-1) * torch.exp(-(T / fc_len).pow(2) / sigma.pow(2) / 2) * torch.cos(
                2 * math.pi * 0.2 * T)
        else:
            result = params[0].pow(-1) * torch.exp(-(T / fc_len).pow(2) / sigma.pow(2) / 2) * torch.sin(
                2 * math.pi * 0.2 * T)
        return result


if __name__ == '__main__':
    # test TFconv_STTF, TFconv_Chirplet, TFconv_Morlet
    for item in [TFconv_STTF, TFconv_Chirplet, TFconv_Morlet]:
        print(item.__name__)
        model = TFconv_STTF(1, 8, 15, padding=7, stride=1,
                            bias=False, clamp_flag=True)  # kernel_size should
        input = torch.randn([1, 1, 1024])
        out = model(input)
        out.mean().backward()
        print(out.shape)
        print(f'{item.__name__:s} test pass')
