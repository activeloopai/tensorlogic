
from __future__ import annotations
import re
from typing import Dict, List, Tuple, Optional, Any, Sequence, Union
from dataclasses import dataclass

from .backend import get_backend, Backend
from .namedtensor import NamedTensor, nt

# ---- Expression AST ----

@dataclass
class TensorRef:
    name: str
    indices: Tuple[str, ...]

@dataclass
class Number:
    value: float

@dataclass
class Call:
    func: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

@dataclass
class BinOp:
    op: str   # '+', '-', '*', '/', '^'
    left: Any
    right: Any

@dataclass
class Equation:
    lhs_name: str
    lhs_indices: Tuple[str, ...]
    rhs: Any  # Expression AST




# ---- Pythonic Sugar: Var / Expr with named index __getitem__ and assignment ----

class Expr:
    """A thin wrapper carrying (Program, AST). Supports arithmetic to build AST."""
    __slots__ = ("prog", "ast")
    def __init__(self, prog: "Program", ast):
        self.prog = prog
        self.ast = ast

    # elementwise binary ops -> BinOp AST
    def _bin(self, other, op: str):
        b = to_expr(other, self.prog)
        return Expr(self.prog, BinOp(op, self.ast, b.ast))

    def __add__(self, other): return self._bin(other, "+")
    def __sub__(self, other): return self._bin(other, "-")
    def __mul__(self, other): return self._bin(other, "*")
    def __truediv__(self, other): return self._bin(other, "/")
    def __pow__(self, p):
        if isinstance(p, (int,float)):
            return Expr(self.prog, BinOp("^", self.ast, Number(float(p))))
        elif isinstance(p, Expr) and isinstance(p.ast, Number):
            return Expr(self.prog, BinOp("^", self.ast, p.ast))
        raise ValueError("Exponent must be a number or Expr(Number).")

    # call-like sugars
    def relu(self): return Expr(self.prog, Call("relu", (self.ast,), {}))
    def sigmoid(self): return Expr(self.prog, Call("sigmoid", (self.ast,), {}))
    def step(self): return Expr(self.prog, Call("step", (self.ast,), {}))
    def lnorm(self): return Expr(self.prog, Call("lnorm", (self.ast,), {}))
    def gelu(self): return Expr(self.prog, Call("gelu", (self.ast,), {}))

def softmax(x: "Expr", axis: str) -> "Expr":
    return Expr(x.prog, Call("softmax", (to_expr(x, x.prog).ast,), {"axis": TensorRef(axis, tuple())}))

def to_expr(x, prog: "Program") -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int,float)):
        return Expr(prog, Number(float(x)))
    if isinstance(x, TensorRef) or isinstance(x, Call) or isinstance(x, BinOp) or isinstance(x, Number):
        return Expr(prog, x)
    raise TypeError(f"Cannot convert {type(x)} to Expr.")

class Var:
    """Program-bound variable reference. Enables:
         X['i','j']   -> Expr(TensorRef('X', ('i','j')))
         K['i','k'] = ...  -> creates Equation(LHS=K[i,k], RHS=...)
    """
    __slots__ = ("prog", "name")
    def __init__(self, prog: "Program", name: str):
        self.prog = prog
        self.name = name

    def __getitem__(self, indices):
        if isinstance(indices, tuple):
            idx = indices
        else:
            idx = (indices,)
        return Expr(self.prog, TensorRef(self.name, tuple(idx)))

    def __setitem__(self, indices, rhs):
        # Handle K["i","i2"] = <Expr or numeric or AST>
        if isinstance(indices, tuple):
            idx = indices
        else:
            idx = (indices,)
        expr = to_expr(rhs, self.prog)
        eq = Equation(self.name, tuple(idx), expr.ast)
        self.prog.equation(eq)
    lhs_name: str
    lhs_indices: Tuple[str, ...]
    rhs: Any  # Expression AST

# ---- Parser ----

_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<NUMBER>\d+(?:\.\d+)?)|"
    r"(?P<IDENT>[A-Za-z_]\w*)|"
    r"(?P<LBRACK>\[)|(?P<RBRACK>\])|"
    r"(?P<LPAREN>\()|(?P<RPAREN>\))|"
    r"(?P<COMMA>,)|(?P<EQUAL>=)|"
    r"(?P<STAR>\*)|(?P<PLUS>\+)|(?P<MINUS>-)|"
    r"(?P<SLASH>/)|(?P<CARET>\^)"
    r")"
)

class Token:
    def __init__(self, typ, val):
        self.typ = typ
        self.val = val
    def __repr__(self):
        return f"Token({self.typ},{self.val})"

def tokenize(s: str):
    pos = 0
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m:
            raise SyntaxError(f"Unexpected char at {pos}: {s[pos:pos+20]}")
        pos = m.end()
        for typ, val in m.groupdict().items():
            if val is not None:
                yield Token(typ, val)
                break
    yield Token("EOF", "")

class Parser:
    def __init__(self, s: str):
        self.tokens = list(tokenize(s))
        self.i = 0

    def peek(self):
        return self.tokens[self.i]

    def pop(self, typ=None):
        t = self.tokens[self.i]
        if typ and t.typ != typ:
            raise SyntaxError(f"Expected {typ} got {t.typ}")
        self.i += 1
        return t

    def parse_equation(self) -> Equation:
        lhs_name, lhs_indices = self.parse_tensor_lhs()
        self.pop("EQUAL")
        rhs = self.parse_expr()
        self.pop("EOF")
        return Equation(lhs_name, tuple(lhs_indices), rhs)

    def parse_tensor_lhs(self):
        name = self.pop("IDENT").val
        idx = []
        if self.peek().typ == "LBRACK":
            self.pop("LBRACK")
            idx = self.parse_index_list()
            self.pop("RBRACK")
        return name, idx

    def parse_tensor_ref(self):
        name = self.pop("IDENT").val
        idx = []
        if self.peek().typ == "LBRACK":
            self.pop("LBRACK")
            idx = self.parse_index_list()
            self.pop("RBRACK")
        return TensorRef(name, tuple(idx))

    def parse_index_list(self):
        idx = []
        if self.peek().typ in ("IDENT",):
            idx.append(self.pop("IDENT").val)
            while self.peek().typ == "COMMA":
                self.pop("COMMA")
                idx.append(self.pop("IDENT").val)
        return idx

    def parse_expr(self):
        return self.parse_add()

    def parse_add(self):
        node = self.parse_mul()
        while self.peek().typ in ("PLUS", "MINUS"):
            op = self.pop().typ
            right = self.parse_mul()
            node = BinOp("+" if op == "PLUS" else "-", node, right)
        return node

    def parse_mul(self):
        node = self.parse_pow()
        while self.peek().typ in ("STAR", "SLASH"):
            op = self.pop().typ
            right = self.parse_pow()
            node = BinOp("*" if op == "STAR" else "/", node, right)
        return node

    def parse_pow(self):
        node = self.parse_primary()
        if self.peek().typ == "CARET":
            self.pop("CARET")
            right = self.parse_pow()
            node = BinOp("^", node, right)
        return node

    def parse_primary(self):
        t = self.peek()
        if t.typ == "NUMBER":
            self.pop()
            return Number(float(t.val))
        if t.typ == "IDENT":
            # Could be function call or tensor ref
            # Lookahead:
            if self.tokens[self.i+1].typ == "LPAREN":
                return self.parse_call()
            else:
                return self.parse_tensor_ref()
        if t.typ == "LPAREN":
            self.pop()
            node = self.parse_expr()
            self.pop("RPAREN")
            return node
        raise SyntaxError(f"Unexpected token {t}")

    def parse_call(self):
        func = self.pop("IDENT").val
        self.pop("LPAREN")
        args = []
        kwargs = {}
        if self.peek().typ != "RPAREN":
            # parse args/kwargs separated by commas
            while True:
                # kw? arg: IDENT '=' expr
                if self.peek().typ == "IDENT" and self.tokens[self.i+1].typ == "EQUAL":
                    key = self.pop("IDENT").val
                    self.pop("EQUAL")
                    val = self.parse_expr()
                    kwargs[key] = val
                else:
                    args.append(self.parse_expr())
                if self.peek().typ != "COMMA":
                    break
                self.pop("COMMA")
        self.pop("RPAREN")
        return Call(func, tuple(args), kwargs)

# ---- Program ----

class Program:
    def __init__(self, backend: str="numpy"):
        self.backend: Backend = get_backend(backend)
        self.tensors: Dict[str, NamedTensor] = {}
        self.equations: List[Equation] = []
        self._vars_cache: dict[str, Var] = {}


        # registry of learnables (backend-specific flags). For torch, use requires_grad on tensors.
        self.learnable_names: set[str] = set()

    # --- Learning (autodiff) ---
    def mark_learnable(self, name: str):
        """Mark a tensor as learnable (if backend supports it). For Torch, sets requires_grad=True."""
        t = self.tensors.get(name)
        if t is None:
            raise KeyError(f"Cannot mark unknown tensor '{name}' learnable.")
        if self.backend.name == "torch":
            import torch
            if isinstance(t.data, torch.Tensor):
                t.data.requires_grad_(True)
        self.learnable_names.add(name)

    def value(self, query: str) -> Any:
        """Return backend-native array for a query (drop indices)."""
        return self.eval(query).data

    def backward(self, loss_query: str = "Loss") -> Dict[str, Any]:
        """Compute gradients of loss w.r.t learnable tensors (Torch/JAX only)."""
        if not self.backend.is_autodiff():
            raise RuntimeError("Autodiff is only available with Torch or JAX backends.")
        loss_nt = self.eval(loss_query)
        loss = loss_nt.data
        if self.backend.name == "torch":
            if loss.dim() != 0:
                # sum to scalar
                loss = loss.sum()
            loss.backward()
            grads = {}
            import torch
            for name in self.learnable_names:
                t = self.tensors.get(name)
                if t is None:
                    continue
                if isinstance(t.data, torch.Tensor) and t.data.grad is not None:
                    grads[name] = t.data.grad.clone()
            return grads
        else:
            # JAX path: return loss value; gradient via jax.grad would require functional style.
            raise NotImplementedError("JAX autodiff API not implemented in this minimal version.")

    def sgd_step(self, lr: float = 1e-2):
        if self.backend.name != "torch":
            raise RuntimeError("sgd_step implemented for torch only in this minimal version.")
        import torch
        for name in self.learnable_names:
            t = self.tensors.get(name)
            if t is None: 
                continue
            if isinstance(t.data, torch.Tensor) and t.data.grad is not None:
                with torch.no_grad():
                    t.data -= lr * t.data.grad
                    t.data.grad.zero_()

    # --- Pythonic sugar API ---
    def __getitem__(self, name: str) -> Var:
        """P['X'] returns a program-bound variable to use with index sugar."""
        if name not in self._vars_cache:
            self._vars_cache[name] = Var(self, name)
        return self._vars_cache[name]

    def vars(self, *names: str):
        """Convenience: K, X, Y = P.vars('K','X','Y')"""
        return tuple(self[name] for name in names)

    def const(self, value: float) -> Expr:
        return Expr(self, Number(float(value)))

    # --- User API

    def set_tensor(self, name: str, tensor: NamedTensor):
        if tensor.backend.name != self.backend.name:
            # convert numerically
            tensor = NamedTensor(self.backend.asarray(tensor.numpy()), tensor.indices, self.backend)
        self.tensors[name] = tensor

    def rand_tensor(self, name: str, indices: Sequence[str], sizes: Sequence[int]):
        shape = tuple(sizes)
        data = self.backend.randn(shape)
        self.tensors[name] = NamedTensor(data, tuple(indices), self.backend)

    def equation(self, eq: str | Equation):
        if isinstance(eq, str):
            eq = Parser(eq).parse_equation()
        self.equations.append(eq)

    def clear_equations(self):
        self.equations.clear()

    # --- Evaluation

    def eval(self, query: str) -> NamedTensor:
        """Backward evaluation of a single query like 'Y[i]'."""
        # parse a LHS-like query
        parser = Parser(query+"=0")
        lhs_name, lhs_idx = parser.parse_tensor_lhs()
        # Evaluate recursively
        return self._eval_tensor(lhs_name, tuple(lhs_idx), set())

    def _eval_tensor(self, name: str, want_idx: Tuple[str, ...], seen: set) -> NamedTensor:
        if name in self.tensors:
            t = self.tensors[name]
            if set(want_idx) - set(t.indices):
                # Missing indices: cannot fabricate
                # We can try to broadcast by inserting size-1 dims; otherwise raise.
                raise KeyError(f"Tensor '{name}' does not have indices {want_idx}. Has {t.indices}.")
            return t.reorder(want_idx)
        # look for equation(s) defining this LHS
        if name in seen:
            # to avoid infinite loops, return zeros of guessed shape
            # Guess shape from first equation RHS
            eqs = [e for e in self.equations if e.lhs_name == name]
            if not eqs:
                raise KeyError(f"No tensor or equation defines '{name}'.")
            out = self._compute_eq(eqs[0], want_idx)
            return out
        seen.add(name)
        eqs = [e for e in self.equations if e.lhs_name == name]
        if not eqs:
            raise KeyError(f"No tensor or equation defines '{name}'.")
        # Sum contributions of all equations with same LHS (implicit sum)
        parts = []
        for e in eqs:
            parts.append(self._compute_eq(e, want_idx))
        # Sum (projection aligned by want_idx)
        out = parts[0]
        for p in parts[1:]:
            out = NamedTensor(self.backend.add(out.data, p.data), out.indices, self.backend)
        return out



    def _eval_tensor_natural(self, name: str, seen: set) -> NamedTensor:
        if name in self.tensors:
            return self.tensors[name]
        if name in seen:
            eqs = [e for e in self.equations if e.lhs_name == name]
            if not eqs:
                raise KeyError(f"No tensor or equation defines '{name}'.")
            return self._compute_eq(eqs[0], eqs[0].lhs_indices)
        seen.add(name)
        eqs = [e for e in self.equations if e.lhs_name == name]
        if not eqs:
            raise KeyError(f"No tensor or equation defines '{name}'.")
        parts = []
        for e in eqs:
            parts.append(self._compute_eq(e, e.lhs_indices))
        # Sum aligned to the first part's indices
        out = parts[0]
        for p in parts[1:]:
            p2 = p.reorder(out.indices)
            out = NamedTensor(self.backend.add(out.data, p2.data), out.indices, self.backend)
        return out

    def _compute_eq(self, e: Equation, want_idx: Tuple[str, ...]) -> NamedTensor:
        lhs_idx = e.lhs_indices
        out = self._eval_expr(e.rhs, lhs_idx)
        # Project (sum) over any indices not present in LHS
        extra = [idx for idx in out.indices if idx not in lhs_idx]
        if extra:
            # sum over each extra axis
            for idx in extra:
                axis = out.indices.index(idx)
                out = NamedTensor(self.backend.sum(out.data, axis=axis), tuple([i for i in out.indices if i != idx]), self.backend)
        # Reorder to the query's desired order
        return out.reorder(want_idx)

    # ---- Expr evaluation


    
    def _eval_expr(self, node, out_indices: Tuple[str, ...]) -> NamedTensor:
        if isinstance(node, Number):
            x = self.backend.asarray(node.value)
            return NamedTensor(x, tuple(), self.backend)

        if isinstance(node, TensorRef):
            # fetch or compute recursively
            t = self.tensors.get(node.name)
            if t is None:
                t = self._eval_tensor_natural(node.name, set())
            # If user referenced the tensor with different index names (aliases), rename
            if tuple(node.indices) != t.indices:
                if len(node.indices) != len(t.indices):
                    raise KeyError(f"Tensor '{node.name}' arity mismatch: referenced {node.indices}, defined as {t.indices}")
                mapping = {old:new for old,new in zip(t.indices, node.indices)}
                t = t.rename(mapping)
            return t.reorder(node.indices)

        if isinstance(node, BinOp):
            if node.op in ("+", "-"):
                a = self._eval_expr(node.left, out_indices)
                b = self._eval_expr(node.right, out_indices)
                if node.op == "+":
                    return a + b
                else:
                    return a - b
            if node.op in ("*", "/", "^"):
                if node.op == "*":
                    return self._einsum_product(node.left, node.right, out_indices)
                elif node.op == "/":
                    a = self._eval_expr(node.left, out_indices)
                    b = self._eval_expr(node.right, out_indices)
                    return NamedTensor(self.backend.truediv(a.data, b.data), out_indices, self.backend)
                elif node.op == "^":
                    a = self._eval_expr(node.left, out_indices)
                    if isinstance(node.right, Number):
                        return NamedTensor(self.backend.pow(a.data, node.right.value), a.indices, self.backend)
                    else:
                        raise ValueError("Exponent must be a number in DSL for now.")

        if isinstance(node, Call):
            fn = node.func.lower()
            if fn in ("relu","sig","sigmoid","exp","log","sqrt","sin","cos","step"):
                x = self._eval_expr(node.args[0], out_indices)
                if fn == "sig": fn = "sigmoid"
                return x.apply_elemwise(fn)

            if fn == "softmax":
                x = self._eval_expr(node.args[0], out_indices)
                axis_name = None
                if "axis" in node.kwargs and isinstance(node.kwargs["axis"], TensorRef):
                    axis_name = node.kwargs["axis"].name
                if axis_name is None:
                    ax = -1
                else:
                    ax = x.indices.index(axis_name)
                out = self.backend.softmax(x.data, axis=ax)
                return NamedTensor(out, x.indices, self.backend)

            if fn == "concat":
                if "axis" not in node.kwargs:
                    raise ValueError("concat requires axis='<index>' kwarg")
                axis_node = node.kwargs["axis"]
                if isinstance(axis_node, TensorRef):
                    new_axis = axis_node.name
                else:
                    raise ValueError("concat axis must be an index name")
                parts = [self._eval_expr(a, out_indices) for a in node.args]
                # append new axis at end
                b = parts[0].backend
                if b.name == "numpy":
                    import numpy as np
                    out = np.stack([p.data for p in parts], axis=len(parts[0].indices))
                elif b.name == "torch":
                    import torch
                    out = torch.stack([p.data for p in parts], dim=len(parts[0].indices))
                else:
                    import numpy as np
                    out = np.stack([p.data for p in parts], axis=len(parts[0].indices))
                new_indices = tuple(list(parts[0].indices) + [new_axis])
                return NamedTensor(out, new_indices, b)

            if fn == "gelu":
                x = self._eval_expr(node.args[0], out_indices)
                b = self.backend
                c = 0.7978845608028654  # sqrt(2/pi)
                x_data = x.data
                x3 = b.pow(x_data, 3.0)
                inner = b.mul(c, b.add(x_data, b.mul(0.044715, x3)))
                t = b.tanh(inner)
                out = b.mul(0.5, b.mul(x_data, b.add(1.0, t)))
                return NamedTensor(out, x.indices, self.backend)

            if fn == "lnorm" or fn == "layernorm":
                x = self._eval_expr(node.args[0], out_indices)
                ax = len(x.indices) - 1
                out = self.backend.layer_norm(x.data, axis=ax)
                return NamedTensor(out, x.indices, self.backend)

            raise KeyError(f"Unknown function '{node.func}'")

        raise TypeError(f"Unknown node {node}")

    def _collect_indices(self, expr) -> List[str]:
        """Collect all index names mentioned by tensors in an expression (for heuristics)."""
        if isinstance(expr, TensorRef):
            return list(expr.indices)
        if isinstance(expr, Number):
            return []
        if isinstance(expr, Call):
            idx = []
            for a in expr.args:
                idx += self._collect_indices(a)
            for a in expr.kwargs.values():
                idx += self._collect_indices(a)
            return idx
        if isinstance(expr, BinOp):
            return self._collect_indices(expr.left) + self._collect_indices(expr.right)
        return []

    def _einsum_product(self, left, right, out_indices: Tuple[str, ...]) -> NamedTensor:
        """Multiply two expressions and sum over indices not in out_indices."""
        # Evaluate factors into possibly intermediate outputs: we need to ensure
        # both sides are produced as NamedTensors with their native indices.
        a = self._eval_to_factor(left)
        b = self._eval_to_factor(right)

        # If both are scalars: simple multiply
        if len(a.indices) == 0 and len(b.indices) == 0:
            return NamedTensor(self.backend.mul(a.data, b.data), tuple(), self.backend)

        # Build global index -> letter map
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        all_idx = list(dict.fromkeys(list(a.indices) + list(b.indices)))
        if len(all_idx) > len(letters):
            raise ValueError("Too many distinct indices for einsum.")
        idx2ch = {idx: letters[i] for i, idx in enumerate(all_idx)}

        a_sub = "".join(idx2ch[i] for i in a.indices)
        b_sub = "".join(idx2ch[i] for i in b.indices)
        out_sub = "".join(idx2ch[i] for i in out_indices)

        subs = f"{a_sub},{b_sub}->{out_sub}" if out_sub else f"{a_sub},{b_sub}->"
        out_data = self.backend.einsum(subs, a.data, b.data)
        return NamedTensor(out_data, out_indices, self.backend)

    def _eval_to_factor(self, expr) -> NamedTensor:
        """Evaluate an expression to a factor (NamedTensor). If elementwise sum/diff,
        first materialize to required indices union."""
        if isinstance(expr, (TensorRef, Number, Call)):
            # Evaluate to some indices; for elementwise later alignment we'll align
            res = self._eval_expr(expr, tuple(self._collect_indices(expr)))
            return res
        if isinstance(expr, BinOp):
            if expr.op in ("*", "/", "^"):
                # recursively collapse
                if expr.op == "*":
                    # choose output indices as union of children indices
                    idx = list(dict.fromkeys(self._collect_indices(expr.left) + self._collect_indices(expr.right)))
                    return self._einsum_product(expr.left, expr.right, tuple(idx))
                elif expr.op == "/":
                    left = self._eval_to_factor(expr.left)
                    right = self._eval_to_factor(expr.right)
                    # For division, ensure same shape
                    right_aligned = right.reorder(left.indices)
                    return NamedTensor(self.backend.truediv(left.data, right_aligned.data), left.indices, self.backend)
                elif expr.op == "^":
                    base = self._eval_to_factor(expr.left)
                    if isinstance(expr.right, Number):
                        return NamedTensor(self.backend.pow(base.data, expr.right.value), base.indices, self.backend)
                    else:
                        raise ValueError("Exponent must be number")
            if expr.op in ("+", "-"):
                # Align both children to union of indices
                idx = list(dict.fromkeys(self._collect_indices(expr.left) + self._collect_indices(expr.right)))
                a = self._eval_expr(expr.left, tuple(idx))
                b = self._eval_expr(expr.right, tuple(idx))
                if expr.op == "+":
                    return NamedTensor(self.backend.add(a.data, b.data), tuple(idx), self.backend)
                else:
                    return NamedTensor(self.backend.sub(a.data, b.data), tuple(idx), self.backend)
        raise TypeError(f"Cannot collapse expression of type {type(expr)}")


# Public helper to build a NamedTensor quickly
def ntensor(data, indices: Sequence[str], backend: str = "numpy") -> NamedTensor:
    from .namedtensor import nt
    return nt(data, indices, get_backend(backend))