
import os, numpy as np, pytest

pytestmark = pytest.mark.skipif("torch" not in os.environ.get("TENSORLOGIC_TEST_BACKENDS","") and __import__("importlib").util.find_spec("torch") is None, reason="torch not installed")

from tensorlogic import Tensor, clear_global_program, mark_learnable, backward, sgd_step, value
import tensorlogic

def test_torch_linear_regression():
    try:
        import torch  # noqa
    except Exception:
        pytest.skip("torch not available")
    
    clear_global_program()
    # Set backend to torch for the global program
    tensorlogic._global_program = tensorlogic.program.Program(backend="torch")
    
    # Data: y = 2*x
    X = Tensor(np.array([[1.0],[2.0],[3.0]]), ["e","j"], name="X")
    Y = Tensor(np.array([2.0,4.0,6.0]), ["e"], name="Y")
    W = Tensor(np.array([0.0]), ["j"], name="W")  # parameter
    
    mark_learnable("W")
    
    # Pred[e] = X[e,j] * W[j]
    Pred = Tensor(np.zeros(3), ["e"], name="Pred")
    Pred["e"] = X["e","j"] * W["j"]
    
    # Loss[] = (Pred[e] - Y[e]) * (Pred[e] - Y[e])
    Loss = Tensor(np.array(0.0), [], name="Loss")
    Loss[""] = (Pred["e"] - Y["e"]) * (Pred["e"] - Y["e"])
    
    # One gradient step with learning rate that allows convergence
    grads = backward("Loss[]")
    sgd_step(lr=0.01)  # Lower learning rate for stable convergence
    
    # Loss should go down after step
    l1 = value("Loss[]").item()
    grads2 = backward("Loss[]")
    sgd_step(lr=0.01)  # Lower learning rate for stable convergence
    l2 = value("Loss[]").item()
    assert l2 <= l1
