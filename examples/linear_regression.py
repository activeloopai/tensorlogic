
from tensorlogic import Program, nt
import numpy as np

try:
    import torch
except Exception:
    raise SystemExit("This example requires PyTorch.")

P = Program(backend="torch")
X = nt(np.array([[1.0],[2.0],[3.0]]), ["e","j"])
Y = nt(np.array([2.0,4.0,6.0]), ["e"])

P.set_tensor("X", X)
P.set_tensor("Y", Y)
P.set_tensor("W", nt(np.array([0.0]), ["j"]))
P.mark_learnable("W")

P.equation("Pred[e] = X[e,j] * W[j]")
P.equation("Loss[] = (Pred[e] - Y[e]) * (Pred[e] - Y[e])")

for step in range(10):
    grads = P.backward("Loss[]")
    P.sgd_step(lr=0.1)
    print(f"step={step:02d} loss={P.value('Loss[]').item():.6f} W={P.tensors['W'].data.detach().cpu().numpy()}")
