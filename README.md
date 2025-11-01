# Tensor Logic

Vibe coded implementation based on [Tensor Logic](https://arxiv.org/pdf/2510.12269) authored by Pedro Domingos

**Tensor Logic**: a named-index tensor language that unifies neural and symbolic AI in a single, tiny core:

```
pip install tensorlogic
```

> *A program is a set of tensor equations. RHS = joins (implicit `einsum`) + projection (sum over indices not in the LHS) + optional nonlinearity.*

This repository provides a lightweight Python framework with **swappable backends**
(Numpy / optional PyTorch / optional JAX) through a thin `einsum`-driven abstraction.

## Highlights

- 🧮 **Named indices**: write equations with symbolic indices instead of raw axis numbers.
- ➕ **Joins & projection**: implicit `einsum` to multiply tensors on shared indices and sum the rest.
- 🧠 **Neuro + Symbolic**: includes helper utilities for relations (Datalog-like facts), attention, kernels, and small graphical models.
- 🔁 **Forward chaining** (fixpoint) and **backward evaluation** of queries.
- 🔌 **Backends**: `numpy` built-in; `torch` and `jax` if installed.
- 🧪 **Tests**: cover each section of the paper with compact, didactic examples.

> Learning / gradients are supported when the backend has autograd (Torch/JAX).
> With Numpy backend, you can evaluate programs but not differentiate them.

## Quick peek

```python
from tensorlogic import Tensor

# Minimal tensor logic - just like writing math equations!
W = Tensor([[2., -1.],[0.3, 0.7]], ["i","j"], name="W")  # 2x2 weights
X = Tensor([1., 3.], ["j"], name="X")         # 2 inputs
Y = Tensor([0., 0.], ["i"], name="Y")         # output

Y["i"] = (W["i","j"] * X["j"]).step()      # einsum 'ij,j->i' + step
result = Y["i"].eval()                      # evaluate eagerly

print(result.indices, result.data)         # ('i',)  [0. 1.]
```

## Write equations just like math

```python
from tensorlogic import Tensor
import numpy as np

# Kernel computation (squared dot kernel)
X = Tensor([[1.,2.],[3.,4.]], ["i","j"], name="X")
K = Tensor(np.zeros((2,2)), ["i","i2"], name="K")
K["i","i2"] = (X["i","j"] * X["i2","j"]) ** 2
print("Kernel:", K["i","i2"].eval().numpy())

# Attention mechanism (now using a single pass, no intermediate computation)
X = Tensor(np.array([[0.1, 0.2],[0.3, 0.4],[0.1,0.8]]), ["p","d"], name="X")
WQ = Tensor(np.eye(2), ["dk","d"], name="WQ")
WK = Tensor(np.eye(2), ["dk","d"], name="WK")
WV = Tensor(np.eye(2), ["dv","d"], name="WV")

Query = Tensor(np.zeros((3,2)), ["p","dk"], name="Query")
Key = Tensor(np.zeros((3,2)), ["p","dk"], name="Key")
Val = Tensor(np.zeros((3,2)), ["p","dv"], name="Val")
Comp = Tensor(np.zeros((3,3)), ["p","p2"], name="Comp")
Attn = Tensor(np.zeros((3,2)), ["p","dv"], name="Attn")

Query["p","dk"] = WQ["dk","d"] * X["p","d"]
Key["p","dk"]   = WK["dk","d"] * X["p","d"]
Val["p","dv"]   = WV["dv","d"] * X["p","d"]

# Compute raw attention scores using deferred evaluation (no intermediate numpy operations)
Comp["p","p2"] = Query["p","dk"] * Key["p2","dk"]  # einsum over shared 'dk'
scores = Comp["p","p2"].eval().numpy()
print("Raw scores:", scores)
```

This compiles to efficient backend `einsum` on NumPy / PyTorch / JAX.

- **Native symbolic/Datalog style** via `Relation`:
  ```python
  People = Domain(["Alice","Bob","Charlie"])
  Parent = Relation("Parent", People, People)
  Sister = Relation("Sister", People, People)
  Aunt   = Relation("Aunt",   People, People)

  Parent["Bob","Charlie"] = 1     # facts
  Sister["Alice","Bob"]    = 1

  Aunt["x","z"] = (Sister["x","y"] * Parent["y","z"]).step()   # rule
  ```
  Facts are stored in the program as Boolean tensors; rules are equations (join + projection + step),
  and the final relation is the OR of facts and rules.

- **Learnable parameters through `Tensor`**:
  ```python
  # Data tensor
  X = Tensor(np.random.randn(3, 5), ["i","j"], name="X")

  # Learnable parameter (Xavier init), marked learnable by default when no init is provided
  W = Tensor(idxs=["o","i"], sizes=[8, 5], name="W")

  # Non-learnable parameter (explicit init)
  C = Tensor(idxs=["i","j"], sizes=[3, 5], name="C", init="zeros", learnable=False)
  ```

- **Attention correctness** in examples: scaled dot-product (`1/sqrt(dk)`), normalized
  along the comparison axis (`softmax(..., axis="p2")`).

- **Examples** updated: `examples/attention.py`, `examples/symbolic_aunt.py`.


# Development
Repository is under development
```
uv run pytest
```

