"""
An implementation of automatic differentiation in the style of
PyTorch for a single tensor variable.

Inspired by Karpathy's micrograd, much of the code is
taken directly from that project.
https://github.com/karpathy/micrograd

We will support operations involved in a simple feedforward network
that uses ReLU activation functions.
This includes weight matrix multiplication, bias vector addition,
and element-wise ReLU activation.
Furthermore, we will implement a cross entropy loss which requires
further support of indexing, arithmetic, axis sums, and exp and log functions.
"""

import numpy as np

class Tensor:

    def __init__(self, data, _children=()) -> None:
        self.data = data if isinstance(data, np.ndarray) else np.array(data, dtype='float32')
        self.grad = np.zeros_like(data, dtype='float32')
        self.shape = self.data.shape
        # To construct the autograd graph, several internal
        # variables are needed.
        self._backward = lambda: None
        self._prev = set(_children)

    def __getitem__(self, key):
        out = Tensor(self.data[key], (self,))

        def _backward():
            grads = np.zeros_like(self.grad)
            grads[key] = out.grad
            self.grad += grads
        out._backward = _backward

        return out

    def __add__(self, other):
        # Sizes of self and other must allow broadcasting
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            def shape_sum(tensor: Tensor):
                # tensor may be broadcasted to facilitate addition into out
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

    def __neg__(self): 
        # define -self
        return self * -1

    def __sub__(self, other): 
        # subtract based on negation addition
        return self + (-other)

    def __mul__(self, other):
        # Sizes of self and other must allow broadcasting
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            def shape_mul(tensor1: Tensor, tensor2: Tensor):
                # tensor may be broadcasted to facilitate multiplication into out
                diff = len(out.shape) - len(tensor1.shape)
                pos = [axis for axis,dim in 
                    enumerate([1 for _ in range(diff)] + list(tensor1.shape))
                    if dim == 1]
                tensor1.grad += np.sum(out.grad * tensor2.data, axis=tuple(pos)).reshape(tensor1.shape)
                # tensor position receives gradients from all out positions
                # over which it was broadcasted
            shape_mul(self, other)
            shape_mul(other, self)
        out._backward = _backward

        return out

    def __pow__(self, other):
        # broadcasting power follows same rules as other arithmetic
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data ** other.data, (self, other))

        def _backward():
            # self ** other derivative w.r.t self is other * self**(other-1)
            selfdiff = len(out.shape) - len(self.shape)
            selfpos = [axis for axis,dim in 
                enumerate([1 for _ in range(selfdiff)] + list(self.shape))
                if dim == 1]
            self.grad += np.sum(out.grad * other.data * self.data**(other.data-1), 
                            axis=tuple(selfpos)).reshape(self.shape)
            # self ** other derivative w.r.t other is log(self) * self**other
            otherdiff = len(out.shape) - len(other.shape)
            otherpos = [axis for axis,dim in 
                enumerate([1 for _ in range(otherdiff)] + list(other.shape))
                if dim == 1]
            other.grad += np.sum(out.grad * np.log(self.data) * out.data, 
                            axis=tuple(otherpos)).reshape(other.shape)
        out._backward = _backward

        return out

    def __truediv__(self, other):
        # broadcasting self / other using pow = -1
        return self * other**-1

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

    def max(self):
        # Only supports max over the last axis of a 2-dimensional array, the
        # specific use-case in batch classification [batch_dim, num_classes]
        out = Tensor(self.data.max(axis=-1), (self,))

        def _backward():
            mask = np.zeros_like(self.data)
            mask[np.arange(self.shape[0]), self.data.argmax(axis=-1)] = 1
            self.grad += mask * np.expand_dims(out.grad, axis=-1)
        out._backward = _backward

        return out

    def sum(self, axis=None):
        broadcast_summable = np.sum(self.data, axis=axis, keepdims=True)
        out = Tensor(np.squeeze(broadcast_summable, axis=axis), (self,))

        def _backward():
            # broadcast addition to pass gradients to input
            self.grad += out.grad.reshape(broadcast_summable.shape)
        out._backward = _backward

        return out

    def unsqueeze(self, axis=None):
        out = Tensor(np.expand_dims(self.data, axis=axis), (self,))

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.), (self,))

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,))

        def _backward():
            # derivative of exp(a) w.r.t a is exp(a)
            self.grad += out.grad * out.data
        out._backward = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,))

        def _backward():
            # derivative of log(a) w.r.t a is 1/a
            self.grad += out.grad * self.data**-1
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
        return f"Tensor(data={self.data}, grad={self.grad})"