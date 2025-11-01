
import numpy as np
from tensorlogic import Tensor, clear_global_program

def test_join_projection_step():
    clear_global_program()
    Sister = Tensor(np.array([[0,1,0],
                              [0,0,0],
                              [0,0,0]], dtype=float), ["x","y"], name="Sister")
    Parent = Tensor(np.array([[0,0,0],
                              [0,0,1],
                              [0,0,0]], dtype=float), ["y","z"], name="Parent")
    # Create Aunt as a named tensor for syntactic sugar
    Aunt = Tensor(np.zeros((3,3)), ["x","z"], name="Aunt")
    Aunt["x","z"] = (Sister["x","y"] * Parent["y","z"]).step()
    A = Aunt["x","z"].eval()
    assert A.numpy()[0,2] == 1.0
    assert A.numpy().sum() == 1.0
