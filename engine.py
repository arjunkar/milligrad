"""
An implementation of backpropagation in the style of
PyTorch for a single tensor variable.

Inspired by Karpathy's micrograd, much of the code is
taken directly from that project.
https://github.com/karpathy/micrograd

We will support operations involved in a simple feedforward network
that uses ReLU activation functions.
This includes weight matrix multiplication, bias vector addition,
and element-wise ReLU activation.
"""

import numpy as np

class Tensor:

    def __init__(self, data, _children=()) -> None:
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.grad = np.zeros_like(data)
        self.shape = self.data.shape
        # To construct the autograd graph, several internal
        # variables are needed.
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        # Sizes of self and other must allow addition broadcasting
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            def shape_sum(tensor: Tensor):
                if tensor.shape == out.shape:
                    tensor.grad += out.grad
                else: # tensor was broadcasted to facilitate addition into out
                    diff = len(out.shape) - len(tensor.shape)
                    pos = [axis for axis,dim in 
                        enumerate([1 for _ in range(diff)] + list(tensor.shape))
                        if dim == 1]
                    tensor.grad += np.sum(out.grad, axis=tuple(pos)).reshape(tensor.shape)
                    # tensor position receives gradients from all out positions
                    # over which it was broadcasted
            shape_sum(self)
            shape_sum(other)
        out._backward = _backward

        return out

    def matmul(self, other):
        # Sizes of self and other must allow matrix multiplication and broadcasting.
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), (self, other))

        def _backward():
            if len(self.shape) == 1 and len(other.shape) == 1:
                # dot product of vectors
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            elif len(self.shape) > 1 and len(other.shape) == 1:
                # matrix multiplying vector
                self.grad += np.expand_dims(out.grad, axis=-1) * other.data
                other.grad += (np.expand_dims(out.grad, axis=-1) * self.data
                                ).reshape(-1,other.shape[0]).sum(axis=0)
            elif len(self.shape) == 1 and len(other.shape) > 1:
                # vector multiplying matrix from left
                self.grad += (
                    np.expand_dims(out.grad, axis=-1) * np.swapaxes(other.data,
                    axis1=-1,axis2=-2)).reshape(-1,other.shape[-2]).sum(axis=0)
                other.grad += np.expand_dims(out.grad, axis=-2) * np.expand_dims(self.data, axis=-1)
            else: 
                # matrix multiplication
                self.grad += np.matmul(out.grad, np.swapaxes(other.data, axis1=-1, axis2=-2))
                other.grad += np.matmul(np.swapaxes(self.data, axis1=-1,axis2=-2), out.grad)
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.), (self,))

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self, gradient=None):

        # Constructing reverse topological order by depth first search
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Initial gradient is dT/dT = 1, chain rule to generate children
        self.grad =  gradient if gradient is not None else np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"