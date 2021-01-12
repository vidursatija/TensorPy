# example.py

from tensor import Tensor, Optimizer
import numpy as np

import matplotlib.pyplot as plt
import sklearn.datasets as skd

import torch
from torch import optim


X, y = skd.load_diabetes(return_X_y=True)

partition = len(X) // 10

X_train = X[:-partition]
y_train = y[:-partition]

X_test = X[-partition:]
y_test = y[-partition:]


# y_out = matmul( (batch, 13), (13, 1) ) + (1) = (batch, 1)

W = Tensor(np.random.rand(10, 1) - 0.5)
b = Tensor(np.random.rand(1) - 0.5)

Wt = torch.tensor(W.value, requires_grad=True)
bt = torch.tensor(b.value, requires_grad=True)

batch_size = 16
learning_rate = 0.025
optimizer = Optimizer([W, b], lr=learning_rate)
optimizer_t = optim.SGD([Wt, bt], lr=learning_rate, momentum=0.0)

print("GOING TO START")

losses = []
losses_t = []

for _ in range(15):
    for i in range(0, (len(X_train) + batch_size - 1) // batch_size):
        X_batch = Tensor(X_train[i * batch_size: (i + 1)
                                 * batch_size], compute_grad=False)
        y_batch = Tensor(y_train[i * batch_size: (i + 1)
                                 * batch_size].reshape(-1, 1), compute_grad=False)

        # Tensor(np.array([[2*len(X_batch.value)]]), compute_grad=False)
        b_size = 2 * len(X_batch.value)

        y_out = X_batch.matmul(W) + b
        y_out_t = torch.matmul(torch.tensor(X_batch.value), Wt) + bt

        l1 = y_out - y_batch
        l1_t = y_out_t - torch.tensor(y_batch.value)

        loss = (l1 * l1).sum(axis=None) / b_size
        loss_t = 0.5 * (l1_t * l1_t).mean()
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
    y_batch = Tensor(y_test.reshape(-1, 1), compute_grad=False)

    b_size = 2 * len(X_batch.value)

    y_out = X_batch.matmul(W) + b
    y_out_t = torch.matmul(torch.tensor(X_batch.value), Wt) + bt

    l1 = y_out - y_batch
    l1_t = y_out_t - torch.tensor(y_batch.value)

    loss = (l1 * l1).sum(axis=None) / b_size
    loss_t = 0.5 * (l1_t * l1_t).mean()
    print("Test: ", loss.value, loss_t)


plt.subplot(211)
plt.plot(losses)
plt.subplot(212)
plt.plot(losses_t)
plt.show()
