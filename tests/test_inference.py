
import numpy as np
from tensorlogic import Tensor, clear_global_program

def test_forward_backward_eval_equivalence():
    clear_global_program()
    A = Tensor(np.array([[1,0],[0,1]], dtype=float), ["i","j"], name="A")
    B = Tensor(np.array([[1,1],[0,1]], dtype=float), ["j","k"], name="B")
    C = Tensor(np.zeros((2,2)), ["i","k"], name="C")
    C["i","k"] = A["i","j"] * B["j","k"]
    result = C["i","k"].eval()
    # manual
    want = np.einsum("ij,jk->ik", A.data, B.data)
    assert np.allclose(result.numpy(), want)
