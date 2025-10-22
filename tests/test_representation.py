
import numpy as np
from tensorlogic import Program, nt

def test_join_projection_step():
    P = Program(backend="numpy")
    P.set_tensor("Sister", nt(np.array([[0,1,0],
                                        [0,0,0],
                                        [0,0,0]], dtype=float), ["x","y"]))
    P.set_tensor("Parent", nt(np.array([[0,0,0],
                                        [0,0,1],
                                        [0,0,0]], dtype=float), ["y","z"]))
    P.equation("Aunt[x,z] = step(Sister[x,y] * Parent[y,z])")
    A = P.eval("Aunt[x,z]")
    assert A.numpy()[0,2] == 1.0
    assert A.numpy().sum() == 1.0
