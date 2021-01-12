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


relu = torch.nn.ReLU()

nums = skd.load_digits()

X = nums.data
y = nums.target

partition = len(X) // 10

X_train = X[:-partition]
y_train = y[:-partition]

X_test = X[-partition:]
y_test = y[-partition:]


# y_out = matmul( (batch, 64), (64, 10) ) + (10) = (batch, 10)

W1 = Tensor(np.random.rand(64, 32) - 0.5)
b1 = Tensor(np.random.rand(32) - 0.5)
W2 = Tensor(np.random.rand(32, 10) - 0.5)
b2 = Tensor(np.random.rand(10) - 0.5)

W1t = torch.tensor(W1.value, requires_grad=True)
b1t = torch.tensor(b1.value, requires_grad=True)
W2t = torch.tensor(W2.value, requires_grad=True)
b2t = torch.tensor(b2.value, requires_grad=True)

batch_size = 16
learning_rate = 0.02
optimizer = Optimizer([W1, b1, W2, b2], lr=learning_rate)
optimizer_t = optim.SGD([W1t, b1t, W2t, b2t], lr=learning_rate, momentum=0.0)

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

        y1_out = (X_batch.matmul(W1) + b1).max(0.0)
        y1_out_t = relu(torch.matmul(torch.tensor(X_batch.value), W1t) + b1t)

        y_out = softmax(y1_out.matmul(W2) + b2, axis=-1)
        y_out_t = torch.softmax(torch.matmul(y1_out_t, W2t) + b2t, axis=-1)

        loss = 0 - (y_out.log() * y_batch).sum(axis=None) / b_size
        loss_t = -(torch.tensor(y_batch) * torch.log(y_out_t)).sum() / b_size

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

    y1_out = (X_batch.matmul(W1) + b1).max(0.0)
    y1_out_t = relu(torch.matmul(torch.tensor(X_batch.value), W1t) + b1t)

    y_out = softmax(y1_out.matmul(W2) + b2, axis=-1)
    y_out_t = torch.softmax(torch.matmul(y1_out_t, W2t) + b2t, axis=-1)

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
