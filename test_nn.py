import torch
import numpy as np
import engine
import nn

def test_feed_forward():
    model = nn.FeedForward([4,3,5])
    input = engine.Tensor(
                np.single(np.random.random_sample(size=(2,4)))
            )

    assert(
        model(input).shape == (2,5)
    )

    assert(
        len(model.parameters()) == 4
    )

def test_cross_entropy():
    a1 = np.single(np.random.random_sample(size=(3,4)))
    c1 = np.array([2,1,3])
    ta1 = torch.tensor(a1, requires_grad=True)
    tc1 = torch.tensor(c1)
    ea1 = engine.Tensor(a1)
    ec1 = engine.Tensor(c1)

    t_fn = torch.nn.CrossEntropyLoss()
    t_loss = t_fn(ta1, tc1)

    e_fn = nn.CrossEntropyLoss()
    e_loss = e_fn(ea1, ec1)

    assert(
        torch.allclose(t_loss, torch.tensor(e_loss.data))
    )

    t_loss.backward()
    e_loss.backward()

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )
