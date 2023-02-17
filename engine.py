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
                    tensor.grad += np.sum(out.grad, axis=tuple(range(diff)))
                    # tensor position receives gradients from all out positions
                    # over which it was broadcasted
            shape_sum(self)
            shape_sum(other)
        out._backward = _backward

        return out

    def __mul__(self, other):
        # Sizes of self and other must allow matrix multiplication and broadcasting
        other = other if isinstance(other, Tensor) else Tensor(other)
        if len(self.shape) == 0 or len(other.shape) == 0:
            # np.matmul is not defined on scalars, need to use *
            out = Tensor(self.data * other.data, (self, other))
        else:
            out = Tensor(np.matmul(self.data, other.data), (self, other))

        def _backward():
            if len(self.shape) == len(other.shape) and len(other.shape) < 2:
                # valid for product of scalars or dot product of vectors
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            elif len(self.shape) == 0 and len(other.shape) > 0:
                # out[x] = self*other[x]
                # d(out[x])/d(self) = other[x]
                # d(out[x])/d(other[y]) = self if x==y else 0
                self.grad += np.sum(other.data * out.grad, axis=None)
                other.grad += self.data * out.grad
            elif len(self.shape) > 0 and len(other.shape) == 0:
                self.grad += other.data * out.grad
                other.grad += np.sum(self.data * out.grad, axis=None)
            elif len(self.shape) > 1 and len(other.shape) == 1:
                self.grad += np.expand_dims(out.grad, axis=-1) * np.expand_dims(other.data, axis=0)
                other.grad += np.sum( 
                    np.transpose(
                        np.transpose(out.grad) * np.transpose(self.data)
                        ), axis=tuple(range(len(self.shape)-1)) 
                    )
            elif len(self.shape) == 1 and len(other.shape) == 2:
                # Finish
            else: # At least two dimensions in both self and other
                # out[i][j] = self[i][k]*other[k][j]
                # d(out[i][j])/d(self[m][n]) = other[n][j] if i == m else 0
                # d(out[i][j])/d(other[m][n]) = self[i][m] if j == n else 0
                self.grad += np.matmul(out.grad, np.transpose(other.data, axes=(-1,-2)))
                other.grad += np.matmul(np.transpose(self.data, axes=(-1,-2)), out.grad)

        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.), (self,))

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self, external_grads=None):

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
        self.grad =  external_grads if external_grads is not None else np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"