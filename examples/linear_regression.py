
from tensorlogic import Tensor, mark_learnable, backward, sgd_step, value
import tensorlogic
import numpy as np

try:
    import torch
    tensorlogic.set_backend("torch")

except Exception:
    raise SystemExit("This example requires PyTorch.")


X = Tensor(np.array([[1.0],[2.0],[3.0]]), ["e","j"], name="X")
Y = Tensor(np.array([2.0,4.0,6.0]), ["e"], name="Y")
W = Tensor(np.array([0.0]), ["j"], name="W")


Pred = Tensor(np.zeros(3), ["e"], name="Pred")
Pred["e"] = X["e","j"] * W["j"]

Loss = Tensor(np.array(0.0), [], name="Loss")
Loss[""] = (Pred["e"] - Y["e"]) * (Pred["e"] - Y["e"])

mark_learnable("W")

for step in range(10):
    grads = backward("Loss[]")
    sgd_step(lr=0.01)
    print(f"step={step:02d} loss={value('Loss[]').item():.6f} W={tensorlogic._global_program.tensors['W'].data.detach().cpu().numpy()}")
