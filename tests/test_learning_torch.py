
import os, numpy as np, pytest

pytestmark = pytest.mark.skipif("torch" not in os.environ.get("TENSORLOGIC_TEST_BACKENDS","") and __import__("importlib").util.find_spec("torch") is None, reason="torch not installed")

from tensorlogic import Program, nt

def test_torch_linear_regression():
    try:
        import torch  # noqa
    except Exception:
        pytest.skip("torch not available")
    P = Program(backend="torch")
    # Data: y = 2*x
    X = nt(np.array([[1.0],[2.0],[3.0]]), ["e","j"])
    Y = nt(np.array([2.0,4.0,6.0]), ["e"])
    P.set_tensor("X", X)
    P.set_tensor("Y", Y)
    P.set_tensor("W", nt(np.array([0.0]), ["j"]))  # parameter
    P.mark_learnable("W")
    P.equation("Pred[e] = X[e,j] * W[j]")
    P.equation("Loss[] = (Pred[e] - Y[e]) * (Pred[e] - Y[e])")
    # One gradient step
    grads = P.backward("Loss[]")
    P.sgd_step(lr=0.1)
    # Loss should go down after step
    import torch
    l1 = P.value("Loss[]").item()
    grads2 = P.backward("Loss[]")
    P.sgd_step(lr=0.1)
    l2 = P.value("Loss[]").item()
    assert l2 <= l1
