"""
A demonstration of milligrad's training capability using a
convolutional neural network.

Code template taken from:
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
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

class NeuralNetwork(nn.Module): # LeNet architecture
    def __init__(self):
        super().__init__()
        self.conv_relu_stack = nn.Sequential(
            [            
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120, activate=False),
            nn.ReLU(),
            nn.Linear(120, 84, activate=False),
            nn.ReLU(),
            nn.Linear(84,10, activate=False)
            ]
        )

    def __call__(self, x):
        logits = self.conv_relu_stack(x)
        return logits
    
    def parameters(self):
        return self.conv_relu_stack.parameters()
    
model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.data)

    for batch, (X, y) in enumerate(dataloader):
        (X, y) = (engine.Tensor(X.reshape(-1, 1, image_size, image_size)), 
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
        X, y = (engine.Tensor(X.reshape(-1, 1, image_size, image_size)), 
                  engine.Tensor(y))
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
print("Done!")