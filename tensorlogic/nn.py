
from __future__ import annotations
from typing import Sequence, Tuple
import math
from .program import Var, Expr, softmax
from .namedtensor import nt
from . import _global_program

def param(name: str, indices: Sequence[str], shape: Sequence[int], init: str = "randn", scale: float = 0.02) -> Var:
    """Create a parameter tensor with random init and return Var."""
    b = _global_program.backend
    if init == "eye":
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("eye init requires square 2D shape")
        data = b.eye(shape[0])
    elif init == "zeros":
        data = b.zeros(shape)
    elif init == "ones":
        data = b.ones(shape)
    else:
        data = b.randn(shape) * scale
    _global_program.set_tensor(name, nt(data, indices, b))
    return _global_program[name]

def single_head_self_attention(prefix: str, stream: Var, p: str = "p", d: str = "d", dk: str = "dk", dv: str = "dv"):
    """Construct equations for one self-attention head using sugar. Returns Attn Var for convenience."""
    WQ = param(f"{prefix}.WQ", [dk, d], [1,1], init="randn")  # shapes should be set by caller via set_tensor override or replaced
    WK = param(f"{prefix}.WK", [dk, d], [1,1], init="randn")
    WV = param(f"{prefix}.WV", [dv, d], [1,1], init="randn")
    Query = _global_program[f"{prefix}.Query"]; Key = _global_program[f"{prefix}.Key"]; Val = _global_program[f"{prefix}.Val"]
    Comp = _global_program[f"{prefix}.Comp"]; Attn = _global_program[f"{prefix}.Attn"]

    Query[p,dk] = WQ[dk,d] * stream[p,d]
    Key[p,dk]   = WK[dk,d] * stream[p,d]
    Val[p,dv]   = WV[dv,d] * stream[p,d]

    Comp[p,"p2"]  = softmax((Query[p,dk] * Key["p2",dk]), axis="p2").ast
    Attn[p,dv]    = Comp[p,"p2"] * Val["p2",dv]
    return Attn

def mlp(prefix: str, X: Var, p: str = "p", d: str = "d", dh: str = "dh"):
    W1 = param(f"{prefix}.W1", [d, d], [1,1], init="randn")
    W2 = param(f"{prefix}.W2", [d, d], [1,1], init="randn")
    H = _global_program[f"{prefix}.H"]; Y = _global_program[f"{prefix}.Y"]
    H[p,d] = (W1[d,d] * X[p,d]).gelu().ast
    Y[p,d] = W2[d,d] * H[p,d]
    return Y
