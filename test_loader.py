from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import loader
import numpy as np

def test_constructor():
    t_train = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    )

    t_test = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        )

    cons = loader.DataConstructor(28,
    'milligrad/data/train-images-idx3-ubyte',
    'milligrad/data/train-labels-idx1-ubyte',
    'milligrad/data/t10k-images-idx3-ubyte',
    'milligrad/data/t10k-labels-idx1-ubyte'
    )

    e_train_im, e_train_lab = cons.train()
    e_test_im, e_test_lab = cons.test()

    for i, e_tr in enumerate(e_train_im):
        assert(
            np.array_equal(t_train.data[i].numpy(), e_tr) and
            t_train.targets[i] == e_train_lab[i]
        )
    for i, e_te in enumerate(e_test_im):
        assert(
            np.array_equal(t_test.data[i].numpy(), e_te) and
            t_test.targets[i] == e_test_lab[i]
        )

def test_data_loader():
    t_train = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    )

    t_test = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        )

    batch_size = 32
    
    t_tr_loader = DataLoader(t_train, batch_size=batch_size)
    t_te_loader = DataLoader(t_test, batch_size=batch_size)

    cons = loader.DataConstructor(28,
    'milligrad/data/train-images-idx3-ubyte',
    'milligrad/data/train-labels-idx1-ubyte',
    'milligrad/data/t10k-images-idx3-ubyte',
    'milligrad/data/t10k-labels-idx1-ubyte'
    )

    e_train_im, e_train_lab = cons.train()
    e_test_im, e_test_lab = cons.test()

    e_tr_loader = loader.DataLoader(e_train_im, e_train_lab, batch_size=batch_size)
    e_te_loader = loader.DataLoader(e_test_im, e_test_lab, batch_size=batch_size)

    t_temp = []
    e_temp = []

    for batch, (tX,ty) in enumerate(t_tr_loader):
        t_temp.append(tX.numpy().squeeze())
        t_temp.append(ty.numpy())
        if batch > 10:
            break
    for batch, (eX,ey) in enumerate(e_tr_loader):
        e_temp.append(eX)
        e_temp.append(ey)
        if batch > 10:
            break
    for i in range(len(t_temp)):
        assert(
            np.array_equal(t_temp[i], e_temp[i])
        )
    
    t_temp = []
    e_temp = []

    for batch, (tX,ty) in enumerate(t_te_loader):
        t_temp.append(tX.numpy().squeeze())
        t_temp.append(ty.numpy())
        if batch > 10:
            break
    for batch, (eX,ey) in enumerate(e_te_loader):
        e_temp.append(eX)
        e_temp.append(ey)
        if batch > 10:
            break
    for i in range(len(t_temp)):
        assert(
            np.array_equal(t_temp[i], e_temp[i])
        )