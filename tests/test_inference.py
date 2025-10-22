
import numpy as np
from tensorlogic import Program, nt

def test_forward_backward_eval_equivalence():
    P = Program()
    P.set_tensor("A", nt(np.array([[1,0],[0,1]], dtype=float), ["i","j"]))
    P.set_tensor("B", nt(np.array([[1,1],[0,1]], dtype=float), ["j","k"]))
    P.equation("C[i,k] = A[i,j] * B[j,k]")
    C = P.eval("C[i,k]")
    # manual
    want = np.einsum("ij,jk->ik", P.tensors["A"].data, P.tensors["B"].data)
    assert np.allclose(C.numpy(), want)
