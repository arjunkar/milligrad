import torch
import numpy as np
import engine
import optim

def test_SGD():
    a1 = np.single(np.random.random_sample(size=(2,4)))
    grad = np.single(np.random.random_sample())

    ta1 = torch.tensor(a1, requires_grad=True)
    ea1 = engine.Tensor(a1)

    tsgd = torch.optim.SGD([ta1], lr=0.1, momentum=0.9, dampening=0.2, weight_decay=0.1)
    esgd = optim.SGD([ea1], lr=0.1, momentum=0.9, dampening=0.2, weight_decay=0.1)

    (ta1.relu().sum()).backward(gradient=torch.tensor(grad))
    (ea1.relu().sum()).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

    tsgd.step()
    esgd.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )

    tsgd.step()
    esgd.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )

    tsgd2 = torch.optim.SGD([ta1], lr=0.1, momentum=0.9, dampening=0, weight_decay=0.1, nesterov=True)
    esgd2 = optim.SGD([ea1], lr=0.1, momentum=0.9, dampening=0, weight_decay=0.1, nesterov=True)

    tsgd2.step()
    esgd2.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )
    tsgd2.step()
    esgd2.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )

    tsgd3 = torch.optim.SGD([ta1], lr=0.1, momentum=0.9, dampening=0.2, weight_decay=0.1, maximize=True)
    esgd3 = optim.SGD([ea1], lr=0.1, momentum=0.9, dampening=0.2, weight_decay=0.1, maximize=True)

    tsgd3.step()
    esgd3.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )
    tsgd3.step()
    esgd3.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )

def test_RMSprop():
    a1 = np.single(np.random.random_sample(size=(2,4)))
    grad = np.single(np.random.random_sample())

    ta1 = torch.tensor(a1, requires_grad=True)
    ea1 = engine.Tensor(a1)

    tsgd = torch.optim.RMSprop([ta1], lr=0.1, momentum=0.9, alpha=0.95, weight_decay=0.1)
    esgd = optim.RMSprop([ea1], lr=0.1, momentum=0.9, alpha=0.95, weight_decay=0.1)

    (ta1.relu().sum()).backward(gradient=torch.tensor(grad))
    (ea1.relu().sum()).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

    tsgd.step()
    esgd.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )

    tsgd.step()
    esgd.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )

    tsgd2 = torch.optim.RMSprop([ta1], lr=0.1, momentum=0.9, alpha=0.95, weight_decay=0.1, centered=True)
    esgd2 = optim.RMSprop([ea1], lr=0.1, momentum=0.9, alpha=0.95, weight_decay=0.1, centered=True)

    tsgd2.step()
    esgd2.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )
    tsgd2.step()
    esgd2.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )

    tsgd3 = torch.optim.RMSprop([ta1], lr=0.1, momentum=0.9, alpha=0.95, weight_decay=0.1, maximize=True)
    esgd3 = optim.RMSprop([ea1], lr=0.1, momentum=0.9, alpha=0.95, weight_decay=0.1, maximize=True)

    tsgd3.step()
    esgd3.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )
    tsgd3.step()
    esgd3.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )

def test_Adam():
    a1 = np.single(np.random.random_sample(size=(2,4)))
    grad = np.single(np.random.random_sample())

    ta1 = torch.tensor(a1, requires_grad=True)
    ea1 = engine.Tensor(a1)

    tsgd = torch.optim.Adam([ta1], lr=0.1, weight_decay=0.1)
    esgd = optim.Adam([ea1], lr=0.1, weight_decay=0.1)

    (ta1.relu().sum()).backward(gradient=torch.tensor(grad))
    (ea1.relu().sum()).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

    tsgd.step()
    esgd.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )

    tsgd.step()
    esgd.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )

    tsgd2 = torch.optim.Adam([ta1], lr=0.1, weight_decay=0.1, amsgrad=True)
    esgd2 = optim.Adam([ea1], lr=0.1, weight_decay=0.1, amsgrad=True)

    tsgd2.step()
    esgd2.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )
    tsgd2.step()
    esgd2.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )

    tsgd3 = torch.optim.Adam([ta1], lr=0.1, weight_decay=0.1, amsgrad=True, maximize=True)
    esgd3 = optim.Adam([ea1], lr=0.1, weight_decay=0.1, amsgrad=True, maximize=True)

    tsgd3.step()
    esgd3.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )
    tsgd3.step()
    esgd3.step()
    assert(
        torch.allclose(ta1, torch.tensor(ea1.data))
    )