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
    def test_shapes(shape1, shape2):
        tm1 = torch.randn(size=shape1, requires_grad=True)
        tm2 = torch.randn(size=shape2, requires_grad=True)

        em1 = engine.Tensor(tm1.detach().numpy())
        em2 = engine.Tensor(tm2.detach().numpy())

        extern = torch.randn_like(tm1.matmul(tm2))

        tout = tm1.matmul(tm2)
        eout = em1.matmul(em2)

        assert(
            torch.allclose(tout, torch.tensor(eout.data))
        )

        (tm1.matmul(tm2)).backward(gradient=extern)
        (em1.matmul(em2)).backward(gradient=extern.numpy())

        assert(
            torch.allclose(tm1.grad, torch.tensor(em1.grad))
        )
        assert(
            torch.allclose(tm2.grad, torch.tensor(em2.grad))
        )
    test_shapes( (4,) , (4,) )
    test_shapes( (3,4) , (4,) )
    test_shapes( (3,) , (3,4) )
    test_shapes( (2,3) , (3,2) )
    test_shapes( (3,2,3) , (3,2) )
    test_shapes( (3,2) , (2,2,4) )
    test_shapes( (2,3,2) , (2,2,4) )
    test_shapes( (3, 2, 1, 4), (3, 1, 2, 4, 3) )

def test_max():
    def test_shape_axis(shape, axis):
        ta1 = torch.randn(size=shape, requires_grad=True)
        ea1 = engine.Tensor(ta1.detach().numpy())
        extern = torch.randn_like(ta1.max(dim=axis).values)

        (ta1.max(dim=axis).values).backward(gradient=extern)
        (ea1.max(axis=axis)).backward(gradient=extern.numpy())

        assert(
            torch.allclose(ta1.grad, torch.tensor(ea1.grad))
        )
    # torch cannot apply .max over multiple axes at once
    test_shape_axis( (2,3,4) , 0 )
    test_shape_axis( (2,3,4) , 1 )
    test_shape_axis( (2,3,4) , 2 )

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

def test_as_strided():
    # Because torch.as_strided does not work the same way as
    # np.as_strided, we will have to test the engine manually without
    # using torch's autograd engine.
    extern = torch.randn(size=(5,4))
    ea1 = engine.Tensor(torch.randn(size=(3,3)).numpy())
    # ea1.strides = (12, 4) since np.float32 is 4 bytes
    (ea1.as_strided(shape=(5,4), strides=(4,4))).backward(gradient=extern.numpy())

    expected = torch.tensor(
        [
        [extern[0,0], extern[0,1] + extern[1,0], extern[0,2] + extern[1,1] + extern[2,0]],
        [extern[0,3] + extern[1,2] + extern[2,1] + extern[3,0], 
         extern[1,3] + extern[2,2] + extern[3,1] + extern[4,0], 
         extern[2,3] + extern[3,2] + extern[4,1]],
        [extern[3,3] + extern[4,2], extern[4,3], 0.]
        ]
    )

    assert(
        torch.allclose(expected, torch.tensor(ea1.grad))
    )

def test_cat():
    ta1 = torch.randn(size=(2,3,4), requires_grad=True)
    ta2 = torch.randn(size=(2,3,2), requires_grad=True)
    ea1 = engine.Tensor(ta1.detach().numpy())
    ea2 = engine.Tensor(ta2.detach().numpy())
    extern = torch.randn(size=(2,3,6))

    (torch.cat((ta1, ta2), dim=-1)).backward(gradient=extern)
    (ea1.cat(ea2, axis=-1)).backward(gradient=extern.numpy())

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )
    assert(
        torch.allclose(ta2.grad, torch.tensor(ea2.grad))
    )

def test_reshape():
    # *arg test
    ta1 = torch.randn(size=(2,3,4), requires_grad=True)
    ea1 = engine.Tensor(ta1.detach().numpy())
    extern = torch.randn(size=(8,3))

    (ta1.reshape(8,3)).backward(gradient=extern)
    (ea1.reshape(8,3)).backward(gradient=extern.numpy())

    assert(
        torch.allclose(ta1.grad, torch.tensor(ea1.grad))
    )

    def test_sizes(shape1, shape2):
        ta1 = torch.randn(size=shape1, requires_grad=True)
        ea1 = engine.Tensor(ta1.detach().numpy())
        extern = torch.randn(size=shape2)

        (ta1.reshape(shape2)).backward(gradient=extern)
        (ea1.reshape(shape2)).backward(gradient=extern.numpy())

        assert(
            torch.allclose(ta1.grad, torch.tensor(ea1.grad))
        )
    test_sizes((2,3), (1,6))
    test_sizes((2,8,4,5), (20,16))

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
