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
    def __init__(self, params, lr, momentum=0, dampening=0, 
                weight_decay=0, nesterov=False, *, maximize=False) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        self.steps = 0
        self.b = [np.zeros_like(p.data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            grads = p.grad.copy() if not self.maximize else (-p.grad).copy()
            # do not modify the gradient, keep it
            # as a read-only variable for SGD
            if self.weight_decay != 0:
                grads += self.weight_decay * p.data
            if self.momentum != 0:
                if self.steps > 0:
                    self.b[i] = self.momentum * self.b[i] + (1-self.dampening) * grads
                else:
                    self.b[i] = grads.copy()
                if self.nesterov:
                    grads += self.momentum * self.b[i]
                else:
                    grads = self.b[i]
            p.data -= self.lr * grads
        self.steps += 1