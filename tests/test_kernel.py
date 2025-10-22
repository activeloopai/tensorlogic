
import numpy as np
from tensorlogic import Program, nt

def test_polynomial_kernel():
    P = Program()
    X = nt(np.array([[1.0,2.0],[3.0,4.0]]), ["i","j"])
    P.set_tensor("X", X)
    P.equation("K[i,i2] = (X[i,j] * X[i2,j]) ^ 2")
    K = P.eval("K[i,i2]").numpy()
    # compute manually: (x dot y)^2
    Phi = X.numpy()
    want = (Phi @ Phi.T)**2
    assert np.allclose(K, want)
