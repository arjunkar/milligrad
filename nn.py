"""
Implementation of various classes needed to define a
simple feedforward neural network with a loss
function that will be effective on MNIST or FashionMNIST
classification problems.

Built from the Tensor class which allows for
training by backpropagation in the PyTorch style.

Largely taken from Karpathy's micrograd project, an
educational resource for understanding the high level
functioning of a modern autograd engine.
https://github.com/karpathy/micrograd
"""

import random
import numpy as np
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

class CrossEntropyLoss():
    def __init__(self) -> None:
        pass

    def __call__(self, logits: Tensor, true_classes: Tensor) -> Tensor:
        # log-softmax calculation
        exp = logits.exp()
        norm = exp.sum(axis=-1).unsqueeze(axis=-1)
        log_probs = (exp / norm).log()
        # cross entropy calculation, expects [batch_dim, num_classes]
        batch_dim = logits.shape[0]
        preds = log_probs[np.arange(batch_dim), true_classes.data]
        return -preds.sum() / batch_dim