"""
A demonstration of milligrad's training capability.
Code taken from PyTorch's official FashionMNIST tutorial:
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

Tutorial usage of torch.nn layers, torch.optim optimizer, and
the torch.Tensor class (autograd engine) are substituted with
milligrad's nn, optim, and engine.Tensor class.

This code can achieve 82.2% test accuracy on FashionMNIST
after 5 epochs of SGD training with a 5e-3 learning rate,
which is comparable to the outcome of the PyTorch tutorial
for the specified hyperparameters.
There is a slight slowdown in training time, indicating that
PyTorch acceleration on CPU is nontrivial but not significant
for such a small learning problem.
"""

import engine
import nn
import optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 32

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


model = nn.FeedForward([28*28, 512, 512, 10])
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=5e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        (X, y) = (engine.Tensor(X.numpy().reshape(-1,784)), 
                    engine.Tensor(y.numpy()))

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.data.item(), batch * len(X.data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = 0., 0.

    for X, y in dataloader:
        X, y = engine.Tensor(X.numpy().reshape(-1,784)), engine.Tensor(y.numpy())
        pred = model(X)
        test_loss += loss_fn(pred, y).data
        correct += (pred.data.argmax(1) == y.data).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")