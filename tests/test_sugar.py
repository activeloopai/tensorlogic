
import numpy as np
from tensorlogic import Tensor, clear_global_program

def test_sugar_kernel():
    clear_global_program()
    X = Tensor(np.array([[1.0,2.0],[3.0,4.0]]), ["i","j"], name="X")
    K = Tensor(np.zeros((2,2)), ["i","i2"], name="K")

    # Sugar assignment
    K["i","i2"] = (X["i","j"] * X["i2","j"]) ** 2
    result = K["i","i2"].eval().numpy()
    
    # Expected: (x dot y)^2
    Phi = X.numpy()
    expected = (Phi @ Phi.T)**2
    assert np.allclose(result, expected)
