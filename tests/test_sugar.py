
import numpy as np
from tensorlogic import Program, nt

def test_sugar_kernel_equals_dsl():
    P = Program()
    K, X = P.vars("K","X")
    P.set_tensor("X", nt(np.array([[1.0,2.0],[3.0,4.0]]), ["i","j"]))

    # Sugar assignment
    K["i","i2"] = (X["i","j"] * X["i2","j"]) ** 2
    Ks = P.eval("K[i,i2]").numpy()

    # DSL assignment equivalent
    P2 = Program()
    P2.set_tensor("X", nt(np.array([[1.0,2.0],[3.0,4.0]]), ["i","j"]))
    P2.equation("K[i,i2] = (X[i,j] * X[i2,j]) ^ 2")
    Kd = P2.eval("K[i,i2]").numpy()

    assert np.allclose(Ks, Kd)
