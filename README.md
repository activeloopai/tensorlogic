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
from tensorlogic import Program, nt

P = Program()                             # numpy backend by default
P.set_tensor("W", nt([[2., -1.]], ["i","j"]))  # 1x2
P.set_tensor("X", nt([1., 3.], ["j"]))         # 2

P.equation("Y[i] = step(W[i,j] * X[j])")  # einsum 'ij,j->i' + step
Y = P.eval("Y[i]")                         # returns NamedTensor

print(Y.indices, Y.data)  # ('i',)  array([1., 0.])
```

See `examples/` for more!


## Pythonic sugar (write equations directly in Python)

```python
from tensorlogic import Program, nt, softmax

P = Program()
K, X = P.vars("K","X")
P.set_tensor("X", nt([[1.,2.],
                      [3.,4.]], ["i","j"]))

# K[i,i2] = (X[i,j] * X[i2,j])^2
K["i","i2"] = (X["i","j"] * X["i2","j"]) ** 2

# Attention (single head)
Query, Key, Val, Comp, Attn = P.vars("Query","Key","Val","Comp","Attn")
Query["p","dk"] = WQ["dk","d"] * X["p","d"]
Key["p","dk"]   = WK["dk","d"] * X["p","d"]
Val["p","dv"]   = WV["dv","d"] * X["p","d"]
Comp["p","p2"]  = softmax(Query["p","dk"] * Key["p2","dk"], axis="p2").ast
Attn["p","dv"]  = Comp["p","p2"] * Val["p2","dv"]
```

This compiles to efficient backend `einsum` on NumPy / PyTorch / JAX.
