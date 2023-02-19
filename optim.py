"""
A basic optimizer for use with milligrad's autograd engine
and neural network library.

Designed to follow PyTorch's optim class.

Implements stochastic gradient descent, momentum, and other
common optimization schemes.
"""

import numpy as np

class Optimizer:
    def __init__(self, params, lr=5e-3) -> None:
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.grad)

class SGD(Optimizer):
    def __init__(self, params, lr) -> None:
        super().__init__(params, lr)

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad