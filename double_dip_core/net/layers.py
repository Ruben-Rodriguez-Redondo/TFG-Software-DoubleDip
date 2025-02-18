import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from double_dip_core.net.downsampler import Downsampler


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module_ in enumerate(args):
            self.add_module(str(idx), module_)

    def forward(self, input_):
        inputs = []
        for module_ in self._modules.values():
            inputs.append(module_(input_))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun='LeakyReLU'):
    """
        Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    """
    Creates a 2D batch normalization layer for a given number of features.
    """
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    """
    Creates a 2D convolutional layer with optional padding and downsampling.
    Supports different downsampling modes: stride, average pooling, max pooling, and Lanczos filtering.
    """
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5,
                                      preserve_size=True)
        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = [x for x in [padder, convolver, downsampler] if x is not None]
    return nn.Sequential(*layers)


class VarianceLayer(nn.Module):
    # TODO: make it pad-able
    def __init__(self, patch_size=5, channels=1):
        self.patch_size = patch_size
        super(VarianceLayer, self).__init__()
        mean_mask = np.ones((channels, channels, patch_size, patch_size)) / (patch_size * patch_size)
        self.mean_mask = nn.Parameter(torch.tensor(mean_mask, dtype=torch.float32, device='cuda'), requires_grad=False)
        mask = np.zeros((channels, channels, patch_size, patch_size))
        mask[:, :, patch_size // 2, patch_size // 2] = 1.
        self.ones_mask = nn.Parameter(torch.tensor(mask, dtype=torch.float32, device='cuda'), requires_grad=False)

    def forward(self, x):
        Ex_E = F.conv2d(x, self.ones_mask) - F.conv2d(x, self.mean_mask)
        return F.conv2d((Ex_E) ** 2, self.mean_mask)


class CovarianceLayer(nn.Module):
    def __init__(self, patch_size=5, channels=1):
        self.patch_size = patch_size
        super(CovarianceLayer, self).__init__()
        mean_mask = np.ones((channels, channels, patch_size, patch_size)) / (patch_size * patch_size)
        self.mean_mask = nn.Parameter(torch.tensor(mean_mask, dtype=torch.float32, device='cuda'), requires_grad=False)

        mask = np.zeros((channels, channels, patch_size, patch_size))
        mask[:, :, patch_size // 2, patch_size // 2] = 1.
        self.ones_mask = nn.Parameter(torch.tensor(mask, dtype=torch.float32, device='cuda'), requires_grad=False)

    def forward(self, x, y):
        return F.conv2d((F.conv2d(x, self.ones_mask) - F.conv2d(x, self.mean_mask)) *
                        (F.conv2d(y, self.ones_mask) - F.conv2d(y, self.mean_mask)), self.mean_mask)


class GrayscaleLayer(nn.Module):
    def __init__(self):
        super(GrayscaleLayer, self).__init__()

    def forward(self, x):
        return torch.mean(x, 1, keepdim=True)


def add_module(self, module_):
    """
    Adds a new module to the current model, assigning it a unique name based on the number of existing modules.
    """
    self.add_module(str(len(self) + 1), module_)


torch.nn.Module.add = add_module
