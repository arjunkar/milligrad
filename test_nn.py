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

def test_pad2d():
    epadder = nn.ConstantPad2d((1,2,3,4), 2.1)
    tpadder = torch.nn.ConstantPad2d((1,2,3,4), 2.1)
    ta1 = torch.randn(size=(5,2,3,4))
    ea1 = engine.Tensor(ta1.numpy())

    assert(
        torch.allclose(tpadder(ta1), torch.tensor(epadder(ea1).data))
    )

def test_conv2d():
    def test_fwd_bwd(in_c, out_c, kern, stride, pad):
        tconv = torch.nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kern, 
                                stride=stride, padding=pad)
        econv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kern, 
                          stride=stride, padding=pad)
        for tp, ep in zip(tconv.parameters(), econv.parameters()):
            ep.data = tp.data.numpy()
        tx = torch.randn(size=(10, in_c, 2*kern*stride+3, 2*kern*stride+5), 
                         requires_grad=True)
        ex = engine.Tensor(tx.detach().numpy())
        eout = econv(ex)
        tout = tconv(tx)
        assert(
            torch.allclose(tout, torch.tensor(eout.data), atol=1e-5)
            # Due to floating point error, we do not trust this equivalence
            # beyond 1e-6 or 1e-7 tolerance for larger input sizes.
        )

        extern = torch.randn_like(tout)
        tout.backward(gradient=extern)
        eout.backward(gradient=extern.numpy())
        assert(
            torch.allclose(tx.grad, torch.tensor(ex.grad), atol=1e-5)
        )
        for tp, ep in zip(tconv.parameters(), econv.parameters()):
            assert(
                torch.allclose(tp.grad, torch.tensor(ep.grad), atol=1e-5)
            )

    test_fwd_bwd(1, 1, 2, 2, 1)
    test_fwd_bwd(2, 1, 2, 1, 2)
    test_fwd_bwd(2, 3, 3, 3, 3)
    
def test_maxpool2d():
    def test_fwd_bwd(in_c, kern, stride, pad):
        tpool = torch.nn.MaxPool2d(kernel_size=kern, stride=stride, padding=pad)
        epool = nn.MaxPool2d(kernel_size=kern, stride=stride, padding=pad)
        
        tx = torch.randn(size=(10, in_c, 2*kern*stride+3, 2*kern*stride+5), 
                         requires_grad=True)
        ex = engine.Tensor(tx.detach().numpy())
        eout = epool(ex)
        tout = tpool(tx)
        assert(
            torch.allclose(tout, torch.tensor(eout.data))
        )

        extern = torch.randn_like(tout)
        tout.backward(gradient=extern)
        eout.backward(gradient=extern.numpy())
        assert(
            torch.allclose(tx.grad, torch.tensor(ex.grad), atol=1e-5)
        )

    test_fwd_bwd(1, 2, 2, 1)
    test_fwd_bwd(2, 2, 1, 1)
    test_fwd_bwd(2, 4, 3, 2)

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
