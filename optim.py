"""
A basic optimizer for use with milligrad's autograd engine
and neural network library.

Designed to follow PyTorch's optim class.

Implements stochastic gradient descent, momentum, and other
common optimization schemes including RMSprop and Adam.
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
    def __init__(self, params, lr=5e-3, momentum=0, dampening=0, 
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

class RMSprop(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0,
                momentum=0, centered=False, maximize=False) -> None:
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps 
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.maximize = maximize
        self.b = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.grad_ave = [np.zeros_like(p.data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            grads = p.grad.copy() if not self.maximize else (-p.grad).copy()
            if self.weight_decay != 0:
                grads += self.weight_decay * p.data
            self.v[i] = self.alpha * self.v[i] + (1-self.alpha) * grads**2
            vt = self.v[i].copy()
            if self.centered:
                self.grad_ave[i] = self.alpha * self.grad_ave[i] + (1-self.alpha)*grads
                vt -= self.grad_ave[i]**2
            if self.momentum > 0:
                self.b[i] = self.momentum * self.b[i] + grads/(vt**0.5 + self.eps)
                p.data -= self.lr * self.b[i]
            else:
                p.data -= self.lr * grads / (vt**0.5 + self.eps)

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                amsgrad=False, maximize=False) -> None:
        super().__init__(params, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.v_hat_max = [np.zeros_like(p.data) for p in params]
        self.steps = 1

    def step(self):
        for i, p in enumerate(self.params):
            grads = p.grad.copy() if not self.maximize else (-p.grad).copy()

            if self.weight_decay != 0:
                grads += self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * grads
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * grads**2

            bias_correction1 = 1 - self.beta1**self.steps
            bias_correction2 = 1 - self.beta2**self.steps
            step_size = self.lr / bias_correction1

            if self.amsgrad:
                self.v_hat_max[i] = np.maximum(self.v_hat_max[i], self.v[i])
                v_hat_m = self.v_hat_max[i].copy() / bias_correction2
                p.data -= step_size * self.m[i] / (v_hat_m**0.5 + self.eps)
            else:
                v_hat = self.v[i].copy() / bias_correction2
                p.data -= step_size * self.m[i] / (v_hat**0.5 + self.eps)

            self.steps += 1

class StepLR():
    def __init__(self, optimizer, step_size, gamma=0.1) -> None:
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0
        self.initial_lr = self.optimizer.lr

    def step(self):
        num = self.epoch // self.step_size
        self.optimizer.lr = self.initial_lr * (self.gamma**num)
        self.epoch += 1