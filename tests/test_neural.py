
import numpy as np
from tensorlogic import Program, nt

def test_mlp_perceptron_step():
    P = Program()
    P.set_tensor("W", nt(np.array([[2.0, -1.0],[0.3, 0.7]]), ["i","j"]))
    P.set_tensor("X", nt(np.array([1.0, 3.0]), ["j"]))
    P.equation("Y[i] = step(W[i,j] * X[j])")
    Y = P.eval("Y[i]")
    y = Y.numpy()
    assert y.shape == (2,)
    assert (y == np.array([0.0, 1.0])).all()

def test_attention_head_shapes():
    P = Program()
    X = nt(np.random.randn(4, 3), ["p","d"])
    P.set_tensor("X", X)
    P.set_tensor("WQ", nt(np.eye(3), ["dk","d"]))
    P.set_tensor("WK", nt(np.eye(3), ["dk","d"]))
    P.set_tensor("WV", nt(np.eye(3), ["dv","d"]))
    P.equation("Query[p,dk] = WQ[dk,d] * X[p,d]")
    P.equation("Key[p,dk] = WK[dk,d] * X[p,d]")
    P.equation("Val[p,dv] = WV[dv,d] * X[p,d]")
    P.equation("Comp[p,p2] = softmax(Query[p,dk] * Key[p2,dk], axis=p2)")
    P.equation("Attn[p,dv] = Comp[p,p2] * Val[p2,dv]")
    A = P.eval("Attn[p,dv]")
    assert A.numpy().shape == (4,3)
