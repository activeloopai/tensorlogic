
from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, List, Optional, Any
from .backend import Backend, get_backend

Index = str

@dataclass
class NamedTensor:
    data: Any
    indices: Tuple[Index, ...]
    backend: Backend
    name: Optional[str] = None

    def __init__(self, data, indices: Sequence[Index], backend: Backend = None, name: str = None):
        if backend is None:
            backend = get_backend(None)
        t = backend.asarray(data)
        self.data = t
        self.indices = tuple(indices)
        self.backend = backend
        self.name = name

        # Register with global program if name provided
        if self.name:
            from . import _global_program
            _global_program.set_tensor(self.name, self)


    def __getitem__(self, indices):
        """Enable X['i','j'] syntax like Var."""
        if not self.name:
            raise ValueError("Tensor must have a name to use indexing syntax. Provide name parameter or assign to a variable.")
        from .program import Expr, TensorRef
        from . import _global_program
        if isinstance(indices, tuple):
            idx = indices
        else:
            idx = (indices,)
        return Expr(_global_program, TensorRef(self.name, tuple(idx)))

    def __setitem__(self, indices, rhs):
        """Enable X['i','j'] = ... syntax for equations."""
        if not self.name:
            raise ValueError("Tensor must have a name to use assignment syntax. Provide name parameter or assign to a variable.")
        from .program import to_expr, Equation
        from . import _global_program
        if isinstance(indices, tuple):
            idx = indices
        else:
            idx = (indices,)
        expr = to_expr(rhs, _global_program)
        eq = Equation(self.name, tuple(idx), expr.ast)
        _global_program.equation(eq)

    def _auto_detect_name(self) -> Optional[str]:
        """Try to detect the variable name from the call stack."""
        try:
            # Get the caller's frame
            frame = inspect.currentframe().f_back
            print(f"Frame: {frame}, name: {frame.f_code.co_name if frame else None}")
            if frame and frame.f_code.co_name == '<module>':
                # We're called from global scope
                # Get the source code of the calling line
                import linecache
                filename = frame.f_code.co_filename
                print(f"Filename: {filename}, lineno: {frame.f_lineno}")
                line_content = linecache.getline(filename, frame.f_lineno).strip()
                print(f"Line content: {repr(line_content)}")
                # Look for pattern like "X = Tensor("
                import re
                match = re.match(r'(\w+)\s*=\s*Tensor\s*\(', line_content)
                if match:
                    return match.group(1)
        except Exception as e:
            print(f"Name detection failed: {e}")
            pass
        return None

    def reorder(self, new_indices: Sequence[Index]) -> "NamedTensor":
        """Reorder axes to match new_indices (subset must match)"""
        # Compute permutation to transform current order into new order.
        if tuple(new_indices) == self.indices:
            return self
        pos = []
        for idx in new_indices:
            try:
                pos.append(self.indices.index(idx))
            except ValueError:
                raise KeyError(f"Index '{idx}' not found in tensor with indices {self.indices}")
        transposed = self.backend.transpose(self.data, axes=pos)
        return NamedTensor(transposed, tuple(new_indices), self.backend)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    def numpy(self):
        return self.backend.to_numpy(self.data)

    def rename(self, mapping: Dict[Index, Index]) -> "NamedTensor":
        new_idx = tuple(mapping.get(i, i) for i in self.indices)
        return NamedTensor(self.data, new_idx, self.backend)

    def __add__(self, other: "NamedTensor") -> "NamedTensor":
        other_aligned = align_binary(self, other)
        return NamedTensor(self.backend.add(self.data, other_aligned.data), self.indices, self.backend)

    def __sub__(self, other: "NamedTensor") -> "NamedTensor":
        other_aligned = align_binary(self, other)
        return NamedTensor(self.backend.sub(self.data, other_aligned.data), self.indices, self.backend)

    def __mul__(self, other):
        if isinstance(other, NamedTensor):
            raise ValueError("Use einsum operations (via Tensor indexing) to multiply named tensors with joins.")
        else:
            return NamedTensor(self.backend.mul(self.data, other), self.indices, self.backend)

    def __truediv__(self, other):
        return NamedTensor(self.backend.truediv(self.data, other), self.indices, self.backend)

    def apply_elemwise(self, fn_name: str, **kwargs) -> "NamedTensor":
        fn = getattr(self.backend, fn_name)
        out = fn(self.data, **kwargs) if kwargs else fn(self.data)
        return NamedTensor(out, self.indices, self.backend)


def nt(data, indices: Sequence[Index], backend: Backend=None) -> NamedTensor:
    """Convenience to build a NamedTensor from raw data and index names."""
    return NamedTensor(data, indices, backend)


def align_binary(a: NamedTensor, b: NamedTensor) -> NamedTensor:
    """Broadcast `b` to `a`'s index order for elementwise ops."""
    # We will attempt to permute and/or unsqueeze `b` to match `a`'s indices.
    backend = a.backend
    # First, permute b to the order of indices they share
    axes = []
    for idx in a.indices:
        if idx in b.indices:
            axes.append(b.indices.index(idx))
    # Now transpose b to these axes, then expand missing axes
    bb = b.data
    if axes:
        bb = backend.transpose(bb, axes=axes) if len(axes) != len(b.indices) else bb
    # Insert singleton dims for indices in a that b doesn't have
    # This is easier by reshaping; compute target shape
    b_shape_map = dict(zip(b.indices, b.data.shape))
    target_shape = []
    for idx in a.indices:
        dim = b_shape_map.get(idx, 1)
        target_shape.append(dim)
    # Try reshape/broadcast
    bb = backend.reshape(bb, target_shape) if tuple(target_shape) != tuple(getattr(bb, "shape", [])) else bb
    return NamedTensor(bb, a.indices, backend)
