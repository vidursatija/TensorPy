# example.py

from tensor import Tensor, Optimizer
import numpy as np

import matplotlib.pyplot as plt
import sklearn.datasets as skd

import torch
from torch import optim


def softmax(x, axis=-1):
    hmm = x.exp()
    return hmm / hmm.sum(axis=axis, keepdims=True)


def onehot(indices, output_dim):
    batch_size = len(indices)
    b = np.zeros((batch_size, output_dim))
    b[np.arange(batch_size), indices] = 1
    return b


nums = skd.load_digits()

X = nums.data
y = nums.target

partition = len(X) // 10

X_train = X[:-partition]
y_train = y[:-partition]

X_test = X[-partition:]
y_test = y[-partition:]


# y_out = matmul( (batch, 64), (64, 10) ) + (10) = (batch, 10)

W = Tensor(np.random.rand(64, 10) - 0.5)
b = Tensor(np.random.rand(10) - 0.5)

Wt = torch.tensor(W.value, requires_grad=True)
bt = torch.tensor(b.value, requires_grad=True)

batch_size = 16
learning_rate = 0.02
optimizer = Optimizer([W, b], lr=learning_rate)
optimizer_t = optim.SGD([Wt, bt], lr=learning_rate, momentum=0.0)

print("GOING TO START")

losses = []
losses_t = []

for _ in range(15):
    for i in range(0, (len(X_train) + batch_size - 1) // batch_size):
        X_batch = Tensor(X_train[i * batch_size: (i + 1)
                                 * batch_size], compute_grad=False)
        y_batch = onehot(y_train[i * batch_size: (i + 1)
                                 * batch_size], 10)  # onehot it

        b_size = len(X_batch.value)

        y_out = X_batch.matmul(W) + b
        y_out_t = torch.matmul(torch.tensor(X_batch.value), Wt) + bt

        loss = 0 - (softmax(y_out, axis=-1).log() *
                    y_batch).sum(axis=None) / b_size
        loss_t = -(torch.tensor(y_batch) *
                   torch.log(torch.softmax(y_out_t, axis=-1))).sum() / b_size

        # print(loss.value, loss_t)
        losses.append(loss.value)
        losses_t.append(loss_t)

        optimizer_t.zero_grad()
        optimizer.clear()

        loss.backward(np.ones_like(loss.value))
        loss_t.backward()

        optimizer.step()
        optimizer_t.step()
        # break
    X_batch = Tensor(X_test, compute_grad=False)
    y_batch = onehot(y_test, 10)  # onehot it

    b_size = len(X_batch.value)

    y_out = softmax(X_batch.matmul(W) + b, axis=-1)
    y_out_t = torch.softmax(
        torch.matmul(
            torch.tensor(
                X_batch.value),
            Wt) + bt,
        axis=-1)

    loss = 0 - (y_out.log() * y_batch).sum(axis=None) / b_size
    loss_t = -(torch.tensor(y_batch) * torch.log(y_out_t)).sum() / b_size
    print("Test: ", loss.value, loss_t)
    print(
        "Accuracy",
        (y_out.value.argmax(
            axis=-
            1) == y_test).astype(
            np.float32).mean())

plt.subplot(211)
plt.plot(losses)
plt.subplot(212)
plt.plot(losses_t)
plt.show()
