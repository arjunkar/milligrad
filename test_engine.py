import torch
import engine
import numpy as np

np.random.seed(1234)

def test_getitem():
    a1 = np.single(np.random.random_sample(size=(2,3,4)))
    grad = np.single(np.random.random_sample(size=(3,4)))

    ta1 = torch.tensor(a1, requires_grad=True)
    (ta1[0]).backward(gradient=torch.tensor(grad))
    ea1 = engine.Tensor(a1)
    (ea1[0]).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

    grad = np.single(np.random.random_sample(size=4))

    ta1 = torch.tensor(a1, requires_grad=True)
    (ta1[0][1]).backward(gradient=torch.tensor(grad))
    ea1 = engine.Tensor(a1)
    (ea1[0][1]).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

    grad = np.single(np.random.random_sample())

    ta1 = torch.tensor(a1, requires_grad=True)
    (ta1[0][1][2]).backward(gradient=torch.tensor(grad))
    ea1 = engine.Tensor(a1)
    (ea1[0][1][2]).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

def test_add():
    a1 = np.single(np.random.random_sample(size=(2,3,4)))
    a2 = np.single(np.random.random_sample(size=(2,3,4)))
    b0 = np.single(np.random.random_sample())
    b1 = np.single(np.random.random_sample(size=4))
    b2 = np.single(np.random.random_sample(size=(3,1)))
    b3 = np.single(np.random.random_sample(size=(2,1,1)))
    b4 = np.single(np.random.random_sample(size=(2,1,4)))
    grad = np.single(np.random.random_sample(size=(2,3,4)))

    ta1 = torch.tensor(a1, requires_grad=True)
    ta2 = torch.tensor(a2, requires_grad=True)
    tb0 = torch.tensor(b0, requires_grad=True)
    tb1 = torch.tensor(b1, requires_grad=True)
    tb2 = torch.tensor(b2, requires_grad=True)
    tb3 = torch.tensor(b3, requires_grad=True)
    tb4 = torch.tensor(b4, requires_grad=True)

    (ta1+ta2).backward(gradient=torch.tensor(grad))
    (ta1+tb0).backward(gradient=torch.tensor(grad))
    (ta1+tb1).backward(gradient=torch.tensor(grad))
    (ta1+tb2).backward(gradient=torch.tensor(grad))
    (ta1+tb3).backward(gradient=torch.tensor(grad))
    (ta1+tb4).backward(gradient=torch.tensor(grad))

    ea1 = engine.Tensor(a1)
    ea2 = engine.Tensor(a2)
    eb0 = engine.Tensor(b0)
    eb1 = engine.Tensor(b1)
    eb2 = engine.Tensor(b2)
    eb3 = engine.Tensor(b3)
    eb4 = engine.Tensor(b4)

    (ea1+ea2).backward(gradient=grad)
    (ea1+eb0).backward(gradient=grad)
    (ea1+eb1).backward(gradient=grad)
    (ea1+eb2).backward(gradient=grad)
    (ea1+eb3).backward(gradient=grad)
    (ea1+eb4).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )
    assert(
        torch.allclose(ta2.grad, torch.tensor(ea2.grad))
    )
    assert(
        torch.allclose(tb0.grad, torch.tensor(eb0.grad))
    )
    assert(
        torch.allclose(tb1.grad, torch.tensor(eb1.grad))
    )
    assert(
        torch.allclose(tb2.grad, torch.tensor(eb2.grad))
    )
    assert(
        torch.allclose(tb3.grad, torch.tensor(eb3.grad))
    )
    assert(
        torch.allclose(tb4.grad, torch.tensor(eb4.grad))
    )

def test_sub():
    a1 = np.single(np.random.random_sample(size=(2,3,4)))
    a2 = np.single(np.random.random_sample(size=(2,3,4)))
    b0 = np.single(np.random.random_sample())
    b1 = np.single(np.random.random_sample(size=4))
    b2 = np.single(np.random.random_sample(size=(3,1)))
    b3 = np.single(np.random.random_sample(size=(2,1,1)))
    b4 = np.single(np.random.random_sample(size=(2,1,4)))
    grad = np.single(np.random.random_sample(size=(2,3,4)))

    ta1 = torch.tensor(a1, requires_grad=True)
    ta2 = torch.tensor(a2, requires_grad=True)
    tb0 = torch.tensor(b0, requires_grad=True)
    tb1 = torch.tensor(b1, requires_grad=True)
    tb2 = torch.tensor(b2, requires_grad=True)
    tb3 = torch.tensor(b3, requires_grad=True)
    tb4 = torch.tensor(b4, requires_grad=True)

    (ta1-ta2).backward(gradient=torch.tensor(grad))
    (ta1-tb0).backward(gradient=torch.tensor(grad))
    (ta1-tb1).backward(gradient=torch.tensor(grad))
    (ta1-tb2).backward(gradient=torch.tensor(grad))
    (ta1-tb3).backward(gradient=torch.tensor(grad))
    (ta1-tb4).backward(gradient=torch.tensor(grad))

    ea1 = engine.Tensor(a1)
    ea2 = engine.Tensor(a2)
    eb0 = engine.Tensor(b0)
    eb1 = engine.Tensor(b1)
    eb2 = engine.Tensor(b2)
    eb3 = engine.Tensor(b3)
    eb4 = engine.Tensor(b4)

    (ea1-ea2).backward(gradient=grad)
    (ea1-eb0).backward(gradient=grad)
    (ea1-eb1).backward(gradient=grad)
    (ea1-eb2).backward(gradient=grad)
    (ea1-eb3).backward(gradient=grad)
    (ea1-eb4).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )
    assert(
        torch.allclose(ta2.grad, torch.tensor(ea2.grad))
    )
    assert(
        torch.allclose(tb0.grad, torch.tensor(eb0.grad))
    )
    assert(
        torch.allclose(tb1.grad, torch.tensor(eb1.grad))
    )
    assert(
        torch.allclose(tb2.grad, torch.tensor(eb2.grad))
    )
    assert(
        torch.allclose(tb3.grad, torch.tensor(eb3.grad))
    )
    assert(
        torch.allclose(tb4.grad, torch.tensor(eb4.grad))
    )

def test_mul():
    a1 = np.single(np.random.random_sample(size=(2,3,4)))
    a2 = np.single(np.random.random_sample(size=(2,3,4)))
    b0 = np.single(np.random.random_sample())
    b1 = np.single(np.random.random_sample(size=4))
    b2 = np.single(np.random.random_sample(size=(3,1)))
    b3 = np.single(np.random.random_sample(size=(2,1,1)))
    b4 = np.single(np.random.random_sample(size=(2,1,4)))
    grad = np.single(np.random.random_sample(size=(2,3,4)))

    ta1 = torch.tensor(a1, requires_grad=True)
    ta2 = torch.tensor(a2, requires_grad=True)
    tb0 = torch.tensor(b0, requires_grad=True)
    tb1 = torch.tensor(b1, requires_grad=True)
    tb2 = torch.tensor(b2, requires_grad=True)
    tb3 = torch.tensor(b3, requires_grad=True)
    tb4 = torch.tensor(b4, requires_grad=True)

    (ta1*ta2).backward(gradient=torch.tensor(grad))
    (ta1*tb0).backward(gradient=torch.tensor(grad))
    (ta1*tb1).backward(gradient=torch.tensor(grad))
    (ta1*tb2).backward(gradient=torch.tensor(grad))
    (ta1*tb3).backward(gradient=torch.tensor(grad))
    (ta1*tb4).backward(gradient=torch.tensor(grad))

    ea1 = engine.Tensor(a1)
    ea2 = engine.Tensor(a2)
    eb0 = engine.Tensor(b0)
    eb1 = engine.Tensor(b1)
    eb2 = engine.Tensor(b2)
    eb3 = engine.Tensor(b3)
    eb4 = engine.Tensor(b4)

    (ea1*ea2).backward(gradient=grad)
    (ea1*eb0).backward(gradient=grad)
    (ea1*eb1).backward(gradient=grad)
    (ea1*eb2).backward(gradient=grad)
    (ea1*eb3).backward(gradient=grad)
    (ea1*eb4).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )
    assert(
        torch.allclose(ta2.grad, torch.tensor(ea2.grad))
    )
    assert(
        torch.allclose(tb0.grad, torch.tensor(eb0.grad))
    )
    assert(
        torch.allclose(tb1.grad, torch.tensor(eb1.grad))
    )
    assert(
        torch.allclose(tb2.grad, torch.tensor(eb2.grad))
    )
    assert(
        torch.allclose(tb3.grad, torch.tensor(eb3.grad))
    )
    assert(
        torch.allclose(tb4.grad, torch.tensor(eb4.grad))
    )

def test_div():
    a1 = np.single(np.random.random_sample(size=(2,3,4)))
    a2 = np.single(np.random.random_sample(size=(2,3,4)))
    b0 = np.single(np.random.random_sample())
    b1 = np.single(np.random.random_sample(size=4))
    b2 = np.single(np.random.random_sample(size=(3,1)))
    b3 = np.single(np.random.random_sample(size=(2,1,1)))
    b4 = np.single(np.random.random_sample(size=(2,1,4)))
    grad = np.single(np.random.random_sample(size=(2,3,4)))

    ta1 = torch.tensor(a1, requires_grad=True)
    ta2 = torch.tensor(a2, requires_grad=True)
    tb0 = torch.tensor(b0, requires_grad=True)
    tb1 = torch.tensor(b1, requires_grad=True)
    tb2 = torch.tensor(b2, requires_grad=True)
    tb3 = torch.tensor(b3, requires_grad=True)
    tb4 = torch.tensor(b4, requires_grad=True)

    (ta1/ta2).backward(gradient=torch.tensor(grad))
    (ta1/tb0).backward(gradient=torch.tensor(grad))
    (ta1/tb1).backward(gradient=torch.tensor(grad))
    (ta1/tb2).backward(gradient=torch.tensor(grad))
    (ta1/tb3).backward(gradient=torch.tensor(grad))
    (ta1/tb4).backward(gradient=torch.tensor(grad))

    ea1 = engine.Tensor(a1)
    ea2 = engine.Tensor(a2)
    eb0 = engine.Tensor(b0)
    eb1 = engine.Tensor(b1)
    eb2 = engine.Tensor(b2)
    eb3 = engine.Tensor(b3)
    eb4 = engine.Tensor(b4)

    (ea1/ea2).backward(gradient=grad)
    (ea1/eb0).backward(gradient=grad)
    (ea1/eb1).backward(gradient=grad)
    (ea1/eb2).backward(gradient=grad)
    (ea1/eb3).backward(gradient=grad)
    (ea1/eb4).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )
    assert(
        torch.allclose(ta2.grad, torch.tensor(ea2.grad))
    )
    assert(
        torch.allclose(tb0.grad, torch.tensor(eb0.grad))
    )
    assert(
        torch.allclose(tb1.grad, torch.tensor(eb1.grad))
    )
    assert(
        torch.allclose(tb2.grad, torch.tensor(eb2.grad))
    )
    assert(
        torch.allclose(tb3.grad, torch.tensor(eb3.grad))
    )
    assert(
        torch.allclose(tb4.grad, torch.tensor(eb4.grad))
    )

def test_pow():
    a1 = np.single(np.random.random_sample(size=(2,3,4)))
    a2 = np.single(np.random.random_sample(size=(2,3,4)))
    b0 = np.single(np.random.random_sample())
    b1 = np.single(np.random.random_sample(size=4))
    b2 = np.single(np.random.random_sample(size=(3,1)))
    b3 = np.single(np.random.random_sample(size=(2,1,1)))
    b4 = np.single(np.random.random_sample(size=(2,1,4)))
    grad = np.single(np.random.random_sample(size=(2,3,4)))

    ta1 = torch.tensor(a1, requires_grad=True)
    ta2 = torch.tensor(a2, requires_grad=True)
    tb0 = torch.tensor(b0, requires_grad=True)
    tb1 = torch.tensor(b1, requires_grad=True)
    tb2 = torch.tensor(b2, requires_grad=True)
    tb3 = torch.tensor(b3, requires_grad=True)
    tb4 = torch.tensor(b4, requires_grad=True)

    (ta1**ta2).backward(gradient=torch.tensor(grad))
    (ta1**tb0).backward(gradient=torch.tensor(grad))
    (ta1**tb1).backward(gradient=torch.tensor(grad))
    (ta1**tb2).backward(gradient=torch.tensor(grad))
    (ta1**tb3).backward(gradient=torch.tensor(grad))
    (ta1**tb4).backward(gradient=torch.tensor(grad))

    ea1 = engine.Tensor(a1)
    ea2 = engine.Tensor(a2)
    eb0 = engine.Tensor(b0)
    eb1 = engine.Tensor(b1)
    eb2 = engine.Tensor(b2)
    eb3 = engine.Tensor(b3)
    eb4 = engine.Tensor(b4)

    (ea1**ea2).backward(gradient=grad)
    (ea1**eb0).backward(gradient=grad)
    (ea1**eb1).backward(gradient=grad)
    (ea1**eb2).backward(gradient=grad)
    (ea1**eb3).backward(gradient=grad)
    (ea1**eb4).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )
    assert(
        torch.allclose(ta2.grad, torch.tensor(ea2.grad))
    )
    assert(
        torch.allclose(tb0.grad, torch.tensor(eb0.grad))
    )
    assert(
        torch.allclose(tb1.grad, torch.tensor(eb1.grad))
    )
    assert(
        torch.allclose(tb2.grad, torch.tensor(eb2.grad))
    )
    assert(
        torch.allclose(tb3.grad, torch.tensor(eb3.grad))
    )
    assert(
        torch.allclose(tb4.grad, torch.tensor(eb4.grad))
    )

def test_matmul():
    dot1 = np.single(np.random.random_sample(4))
    dot2 = np.single(np.random.random_sample(4))
    dotgrad = np.single(np.random.random_sample())

    tdot1 = torch.tensor(dot1, requires_grad=True)
    tdot2 = torch.tensor(dot2, requires_grad=True)

    (tdot1.matmul(tdot2)).backward(gradient=torch.tensor(dotgrad))

    edot1 = engine.Tensor(dot1)
    edot2 = engine.Tensor(dot2)

    (edot1.matmul(edot2)).backward(gradient=dotgrad)

    assert(
        torch.allclose(tdot1.grad, torch.tensor(edot1.grad))
    )
    assert(
        torch.allclose(tdot1.grad, torch.tensor(edot1.grad))
    )

    mat1 = np.single(np.random.random_sample(size=(2,3,4)))
    matdot1 = np.single(np.random.random_sample(size=(2,3)))

    tmat1 = torch.tensor(mat1, requires_grad=True)

    (tmat1.matmul(tdot1)).backward(gradient=torch.tensor(matdot1))

    emat1 = engine.Tensor(mat1)

    (emat1.matmul(edot1)).backward(gradient=matdot1)

    assert(
        torch.allclose(tmat1.grad, torch.tensor(emat1.grad))
    )
    assert(
        torch.allclose(tdot1.grad, torch.tensor(edot1.grad))
    )

    dot3 = np.single(np.random.random_sample(3))
    matdot2 = np.single(np.random.random_sample(size=(2,4)))

    tdot3 = torch.tensor(dot3, requires_grad=True)

    (tdot3.matmul(tmat1)).backward(gradient=torch.tensor(matdot2))

    edot3 = engine.Tensor(dot3)

    (edot3.matmul(emat1)).backward(gradient=matdot2)

    assert(
        torch.allclose(tmat1.grad, torch.tensor(emat1.grad))
    )
    assert(
        torch.allclose(tdot3.grad, torch.tensor(edot3.grad))
    )

    mat2 = np.single(np.random.random_sample(size=(2,4,5)))
    matprod = np.single(np.random.random_sample(size=(2,3,5)))

    tmat2 = torch.tensor(mat2, requires_grad=True)

    (tmat1.matmul(tmat2)).backward(gradient=torch.tensor(matprod))

    emat2 = engine.Tensor(mat2)

    (emat1.matmul(emat2)).backward(gradient=matprod)

    assert(
        torch.allclose(tmat1.grad, torch.tensor(emat1.grad))
    )
    assert(
        torch.allclose(tmat2.grad, torch.tensor(emat2.grad))
    )

def test_max():
    a1 = np.single(np.random.random_sample(size=(2,4)))
    grad = np.single(np.random.random_sample(size=()))

    ta1 = torch.tensor(a1, requires_grad=True)
    (ta1.amax(axis=-1).sum()).backward(gradient=torch.tensor(grad))

    ea1 = engine.Tensor(a1)
    (ea1.max().sum()).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

def test_sum():
    a1 = np.single(np.random.random_sample(size=(2,3,4)))
    grad = np.single(np.random.random_sample(size=()))

    ta1 = torch.tensor(a1, requires_grad=True)
    (ta1.sum()).backward(gradient=torch.tensor(grad))

    ea1 = engine.Tensor(a1)
    (ea1.sum()).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

    grad = np.single(np.random.random_sample(size=(2,3)))
    (ta1.sum(axis=-1)).backward(gradient=torch.tensor(grad))
    (ea1.sum(axis=-1)).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

    grad = np.single(np.random.random_sample(size=(4)))
    (ta1.sum(axis=(0,1))).backward(gradient=torch.tensor(grad))
    (ea1.sum(axis=(0,1))).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

def test_unsqueeze():
    a1 = np.single(np.random.random_sample(size=(2,3,4)))
    grad = np.single(np.random.random_sample(size=(2,3,4,1)))

    ta1 = torch.tensor(a1, requires_grad=True)
    ea1 = engine.Tensor(a1)

    (ta1.unsqueeze(-1)).backward(gradient=torch.tensor(grad))
    (ea1.unsqueeze(axis=-1)).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

def test_relu():
    a1 = np.single(np.random.random_sample(size=(2,3,4)))
    grad = np.single(np.random.random_sample(size=(2,3,4)))

    ta1 = torch.tensor(a1, requires_grad=True)
    (ta1.relu()).backward(gradient=torch.tensor(grad))

    ea1 = engine.Tensor(a1)
    (ea1.relu()).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

def test_exp():
    a1 = np.single(np.random.random_sample(size=(2,3,4)))
    grad = np.single(np.random.random_sample(size=(2,3,4)))

    ta1 = torch.tensor(a1, requires_grad=True)
    (ta1.exp()).backward(gradient=torch.tensor(grad))

    ea1 = engine.Tensor(a1)
    (ea1.exp()).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

def test_log():
    a1 = np.single(np.random.random_sample(size=(2,3,4)))
    grad = np.single(np.random.random_sample(size=(2,3,4)))

    ta1 = torch.tensor(a1, requires_grad=True)
    (ta1.log()).backward(gradient=torch.tensor(grad))

    ea1 = engine.Tensor(a1)
    (ea1.log()).backward(gradient=grad)

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

def test_layer():
    x = np.single(np.random.random_sample(size=(5,6)))
    W1 = np.single(np.random.random_sample(size=(6,4)))
    b1 = np.single(np.random.random_sample(size=(4)))
    grad = np.single(np.random.random_sample(size=(5,4)))

    tx = torch.tensor(x, requires_grad=True)
    tW1 = torch.tensor(W1, requires_grad=True)
    tb1 = torch.tensor(b1, requires_grad=True)

    ((tx.matmul(tW1) + tb1).relu()).backward(gradient=torch.tensor(grad))

    ex = engine.Tensor(x)
    eW1 = engine.Tensor(W1)
    eb1 = engine.Tensor(b1)

    ((ex.matmul(eW1) + eb1).relu()).backward(gradient=grad)

    assert(
        torch.allclose(tx.grad, torch.tensor(ex.grad))
    )
    assert(
        torch.allclose(tW1.grad, torch.tensor(eW1.grad))
    )
    assert(
        torch.allclose(tb1.grad, torch.tensor(eb1.grad))
    )
