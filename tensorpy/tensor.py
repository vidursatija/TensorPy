import numpy as np


class Optimizer:
    # Simple optimizer class which is like torch's Optimizer
    def __init__(self, tensors, lr):
        self.lr = lr
        self.tensors = tensors

    def step(self):
        for t in self.tensors:
            t.value -= self.lr * t.grad

    def clear(self):
        for t in self.tensors:
            t.grad[:] = 0


class Tensor:
    # Simple numpy wrapper which tracks gradients
    def __init__(
            self,
            np_array,
            compute_grad=False,
            backward_fn=None):
        self.value = np_array
        self.grad = np.zeros_like(np_array)
        self.backward_fn = backward_fn if backward_fn is not None else lambda x: np.zeros_like(
            np_array)
        self.compute_grad = compute_grad

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(
                self.value + other.value,
                backward_fn=lambda x: (
                    self.backward(x),
                    other.backward(x)))
        else:
            return Tensor(
                self.value + other,
                backward_fn=lambda x: self.backward(x))

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.value - other.value,
                          backward_fn=lambda x: (self.backward(x),
                                                 other.backward(-x)))
        else:
            return Tensor(
                self.value - other,
                backward_fn=lambda x: self.backward(x))

    def __rsub__(self, other):
        return Tensor(other - self.value,
                      backward_fn=lambda x: self.backward(-x))

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(
                self.value * other.value,
                backward_fn=lambda x: (
                    self.backward(
                        other.value * x),
                    other.backward(
                        self.value * x)))
        else:
            return Tensor(
                self.value * other,
                backward_fn=lambda x: self.backward(
                    other * x))

    def __rmul__(self, other):
        return Tensor(
            self.value * other,
            backward_fn=lambda x: self.backward(
                other * x))

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.value / other.value,
                          backward_fn=lambda x: (self.backward(x / other.value),
                                                 other.backward(-self.value * x / (other.value * other.value))))
        else:
            return Tensor(
                self.value / other,
                backward_fn=lambda x: self.backward(
                    x / other))

    def __rtruediv__(self, other):
        return Tensor(other / self.value,
                      backward_fn=lambda x: self.backward(-x * other / (self.value * self.value)))

    def matmul(self, other):
        if isinstance(other, Tensor):
            return Tensor(
                np.matmul(
                    self.value, other.value), backward_fn=lambda x: (
                    self.backward(
                        np.matmul(
                            x, other.value.T)), other.backward(
                        np.matmul(
                            self.value.T, x))))
        else:
            return Tensor(
                np.matmul(
                    self.value, other), backward_fn=lambda x: self.backward(
                    np.matmul(
                        x, other.value.T)))

    def abs(self):
        return Tensor(np.absolute(self.value), backward_fn=lambda x: self.backward(
            x * np.absolute(self.value) / (self.value + 1e-8)))

    def __pow__(self, exp):
        if isinstance(exp, Tensor):
            op = np.power(self.value, exp.value)
            return Tensor(
                op,
                backward_fn=lambda x: (
                    self.backward(
                        x *
                        exp.value *
                        np.power(
                            self.value,
                            exp.value -
                            1)),
                    other.backward(
                        x *
                        np.log(
                            self.value +
                            1e-8) *
                        op)))
        else:
            return Tensor(
                np.power(
                    self.value,
                    exp),
                backward_fn=lambda x: self.backward(
                    x *
                    exp *
                    np.power(
                        self.value,
                        exp -
                        1)))

    def __rpow__(self, exp):
        op = np.power(exp, self.value)
        return Tensor(
            op,
            backward_fn=lambda x: self.backward(
                x *
                np.log(
                    exp +
                    1e-8) *
                op))

    def exp(self):
        op = np.exp(self.value)
        return Tensor(op,
                      backward_fn=lambda x: self.backward(x * op))

    def sin(self):
        return Tensor(
            np.sin(
                self.value),
            backward_fn=lambda x: self.backward(
                x *
                np.cos(
                    self.value)))

    def cos(self):
        return Tensor(np.cos(self.value),
                      backward_fn=lambda x: self.backward(-x * np.sin(self.value)))

    def sum(self, axis=-1, keepdims=False):
        return Tensor(
            np.sum(
                self.value,
                axis=axis,
                keepdims=keepdims),
            backward_fn=lambda x: self.backward(x))

    def log(self):
        return Tensor(
            np.log(
                self.value), backward_fn=lambda x: self.backward(
                x / self.value))

    def tanh(self):
        op = np.tanh(self.value)
        return Tensor(
            op, backward_fn=lambda x: self.backward(x * (1 - op * op)))

    def max(self, other):
        if isinstance(other, Tensor):
            return Tensor(
                np.maximum(
                    self.value, other.value), backward_fn=lambda x: (
                    self.backward(
                        x * (
                            self.value >= other.value).astype(
                            x.dtype)), other.backward(
                        x * (
                            other.value >= self.value).astype(
                            x.dtype))))
        else:
            return Tensor(np.maximum(self.value, other), backward_fn=lambda x: self.backward(
                x * (self.value >= other).astype(x.dtype)))

    def min(self, other):
        if isinstance(other, Tensor):
            return Tensor(
                np.minimum(
                    self.value, other.value), backward_fn=lambda x: (
                    self.backward(
                        x * (
                            self.value <= other.value).astype(
                            x.dtype)), other.backward(
                        x * (
                            other.value <= self.value).astype(
                            x.dtype))))
        else:
            return Tensor(np.minimum(self.value, other), backward_fn=lambda x: self.backward(
                x * (self.value <= other).astype(x.dtype)))

    def backward(self, x=np.array([1.0])):
        # Gradient backward function. It calculates current gradient and passes it on to parent nodes.
        # x should have the same shape as self.grad
        if self.compute_grad is False:
            self.grad = np.zeros_like(self.value)
        else:
            if self.grad.size > x.size:
                x = np.broadcast_to(x, self.grad.shape)
            elif self.grad.size < x.size:
                start_match = len(x.shape) - len(self.grad.shape)
                sum_dims = list(range(0, start_match))
                for d in range(start_match, len(x.shape)):
                    if x.shape[d] != self.grad.shape[d - start_match]:
                        sum_dims += [d]
                # technically sum across shapes which have 1 in them
                x = x.sum(
                    axis=tuple(sum_dims),
                    keepdims=True).reshape(
                    self.grad.shape)
                # print("Backward: ",x.shape, self.grad.shape)
            else:
                x = x.reshape(self.grad.shape)
            assert x.shape == self.grad.shape
            self.grad += x
            self.backward_fn(x)
