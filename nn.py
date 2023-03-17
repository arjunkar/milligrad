"""
Implementation of a small neural network library
powered by the milligrad engine.

Built from the Tensor class which allows for
training by backpropagation in the PyTorch style.

Largely taken from Karpathy's micrograd project, an
educational resource for understanding the high level
functioning of a modern autograd engine.
https://github.com/karpathy/micrograd
"""

import random
import numpy as np
import torch # Only imported for random tensor creation
from engine import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, d_in, d_out, activate=True) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.activate = activate
        self.W = Tensor([[random.uniform(-1/d_in**0.5, 1/d_in**0.5) for _ in range(d_out)] 
                        for _ in range(d_in)])
        self.b = Tensor([random.uniform(-1/d_in**0.5, 1/d_in**0.5) for _ in range(d_out)])

    def __call__(self, input: Tensor) -> Tensor:
        preact = input.matmul(self.W) + self.b
        return preact.relu() if self.activate else preact

    def parameters(self):
        return [self.W, self.b]

class FeedForward(Module):
    def __init__(self, dims) -> None:
        super().__init__()
        self.dims = dims
        # dims is a list of integers [input_dim, hidden_1, ..., hidden_n, output_dim]
        # where the last linear layer has no ReLU activation to facilitate cross entropy
        self.layers = [Linear(dims[i],dims[i+1],activate=(i < len(dims)-2)) 
                        for i in range(len(dims)-1)]
        
    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params

class ConstantPad2d(Module):
    def __init__(self, padding, value) -> None:
        super().__init__()
        self.padding = padding
        # padding is a 4-tuple (pad_left, pad_right, pad_top, pad_bottom)
        self.value = value
    
    def __call__(self, x: Tensor) -> Tensor:
        # Expects [batch_dim, num_channels, height_dim, width_dim]
        # or [num_channels, height_dim, width_dim].
        # Will pad the last two dimensions.
        shape_l, shape_r = list(x.shape), list(x.shape)
        shape_l[-1] = self.padding[0]
        shape_r[-1] = self.padding[1]

        shape_t, shape_b = list(x.shape), list(x.shape)
        shape_t[-1] += self.padding[0] + self.padding[1]
        shape_b[-1] += self.padding[0] + self.padding[1]
        shape_t[-2] = self.padding[2]
        shape_b[-2] = self.padding[3]

        pad_l = Tensor(np.zeros(tuple(shape_l), dtype='float32') + self.value)
        pad_r = Tensor(np.zeros(tuple(shape_r), dtype='float32') + self.value)
        pad_t = Tensor(np.zeros(tuple(shape_t), dtype='float32') + self.value)
        pad_b = Tensor(np.zeros(tuple(shape_b), dtype='float32') + self.value)

        padded_lr = pad_l.cat(x.cat(pad_r, axis=-1), axis=-1)
        padded_ud = pad_t.cat(padded_lr.cat(pad_b, axis=-2), axis=-2)

        return padded_ud

def im2col(x: Tensor, K, s) -> Tensor:
        # x, kernel_size, stride
        (N, C, H, W) = x.shape
        # Use as_strided to generate windows as flattened columns.
        # This requires introducing two additional dimensions, which we flatten in the end.
        (sN, sC, s0, s1) = x.strides
        # Output height and width
        Hp = 1 + (H-K)//s
        Wp = 1 + (W-K)//s
        rolled_out_shape = (N, C, K, K, Hp, Wp)
        rolled_out_strides = (sN, sC, s0, s1, s*s0, s*s1)
        out_view = x.as_strided(shape=rolled_out_shape, strides=rolled_out_strides)
        return out_view.reshape(N, 1, C, K**2, Hp*Wp)

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0) -> None:
        # Only supports square kernels, strides, and paddings for now.
        super().__init__()
        self.in_channels = in_channels # C
        self.out_channels = out_channels # C'
        self.kernel_size = kernel_size # K
        self.stride = stride
        self.padding = (padding, padding, padding, padding)
        self.padder = ConstantPad2d(self.padding, 0)
        initial_weights = torch.nn.init.uniform_(torch.empty(
            size=(out_channels, in_channels, kernel_size, kernel_size)
            ), -1/kernel_size, 1/kernel_size)
        # kernels are [C', C, K, K]
        self.kernels = Tensor(initial_weights.numpy())
        initial_bias = torch.nn.init.uniform_(torch.empty(
            size=(out_channels,)
            ), -1/out_channels**0.5, 1/out_channels**0.5)
        self.b = Tensor(initial_bias.numpy())

    def __call__(self, x: Tensor) -> Tensor:
        # Expects [N,C,H,W] after padding, output e.g. [N, C', H', W']
        x = self.padder(x)
        (N, C, H, W) = x.shape
        K = self.kernel_size
        Cp = self.out_channels
        s = self.stride
        Hp = 1 + (H-K)//s
        Wp = 1 + (W-K)//s
        # out_view_cols is [N, 1, C, K**2, H'*W']
        out_view_cols = im2col(x, K, s)
        # vector_kernel is [1, C', C, 1, K**2]
        vector_kernel = self.kernels.reshape(1, Cp, C, 1, K**2)
        # matmul(vector_kernel, out_view_cols) size is [N, C', C, 1, H'*W']
        mult_output = vector_kernel.matmul(out_view_cols)
        in_channel_reduction = mult_output.sum(axis=2)
        return in_channel_reduction.reshape(N, Cp, Hp, Wp) + self.b.reshape(1, Cp, 1, 1)

    def parameters(self):
        return [self.kernels, self.b]

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=1, padding=0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (padding, padding, padding, padding)
        self.padder = ConstantPad2d(self.padding, -np.inf)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.padder(x)
        (N, C, H, W) = x.shape
        K = self.kernel_size
        s = self.stride
        Hp = 1 + (H-K)//s
        Wp = 1 + (W-K)//s
        # out_view_cols is [N, 1, C, K**2, H'*W']
        out_view_cols = im2col(x, K, s)
        max_output = out_view_cols.max(axis=-2) # Take max over flattened kernel
        return max_output.reshape(N, C, Hp, Wp)

class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()
    
class Flatten(Module):
    def __init__(self) -> None:
        super().__init__()
        # Follows the NumPy flatten convention rather than torch.
        # Flatten all dimensions except first.
        # Expecting [batch_dim, *]

    def __call__(self, x: Tensor) -> Tensor:
        n = x.shape[0]
        return x.reshape(n, -1)

class Sequential(Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params

class CrossEntropyLoss():
    def __init__(self) -> None:
        pass

    def __call__(self, logits: Tensor, true_classes: Tensor) -> Tensor:
        # log-softmax calculation
        exp = (logits - logits.max(axis=-1).unsqueeze(axis=-1)).exp()  
        # see engine for max behavior
        norm = exp.sum(axis=-1).unsqueeze(axis=-1)
        log_probs = (exp / norm).log()
        # cross entropy calculation, expects [batch_dim, num_classes]
        batch_dim = logits.shape[0]
        preds = log_probs[np.arange(batch_dim), true_classes.data]
        return -preds.sum() / batch_dim