"""
A demonstration of milligrad's training capability.
Code taken from PyTorch's official FashionMNIST tutorial:
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

Tutorial usage of torch.nn layers, torch.optim optimizer, and
the torch.Tensor class (autograd engine) are substituted with
milligrad's nn, optim, and engine.Tensor class.
Custom data loaders are also included.

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
import loader

image_size = 28

cons = loader.DataConstructor(image_size,
    'milligrad/data/train-images-idx3-ubyte',
    'milligrad/data/train-labels-idx1-ubyte',
    'milligrad/data/t10k-images-idx3-ubyte',
    'milligrad/data/t10k-labels-idx1-ubyte'
    )

batch_size = 32

train_dataloader = loader.DataLoader(cons.train_images, 
                    cons.train_labels, batch_size)
test_dataloader = loader.DataLoader(cons.test_images,
                    cons.test_labels, batch_size)

model = nn.FeedForward([image_size**2, 512, 512, 10])
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)
# The learning rate must be chosen carefully here.
# If it is too large, the model can jump to a point in
# parameter space where it is too sure of a wrong answer
# and the loss will be sent to infinity due to overflow, 
# breaking the autograd engine.
# For RMSprop and Adam, lr=1e-4 leads to good results in 5 epochs.
# For SGD, lr=5e-3 is effective in 5 epochs.
# All three situations lead to >80% test accuracy.
scheduler = optim.StepLR(optimizer=optimizer, step_size=1, gamma=0.1)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.data)

    for batch, (X, y) in enumerate(dataloader):
        (X, y) = (engine.Tensor(X.reshape(-1,image_size**2)), 
                    engine.Tensor(y))

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
    size = len(dataloader.data)
    num_batches = 0

    test_loss, correct = 0., 0.

    for X, y in dataloader:
        X, y = engine.Tensor(X.reshape(-1,image_size**2)), engine.Tensor(y)
        pred = model(X)
        test_loss += loss_fn(pred, y).data
        correct += (pred.data.argmax(1) == y.data).sum().item()
        num_batches += 1
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    scheduler.step()
print("Done!")