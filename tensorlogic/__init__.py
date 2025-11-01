
import os
from .backend import get_backend, NumpyBackend, TorchBackend, JaxBackend
from .program import Program, ntensor
from .relations import Domain, relation_from_facts
from .utils import text_to_boolean_matrix

from .program import Expr, Var, softmax

# Global program instance for minimal syntax
_global_program = Program()

# Import after global program is created to avoid circular import
from .namedtensor import NamedTensor, nt as nt

# Tensor is an alias for NamedTensor for PyTorch-like API

# Tensor: factory that returns a NamedTensor.
# Usage:
#   Tensor(np_array, ["i","j"], name="X")                   # data
#   Tensor(idxs=["dk","d"], sizes=[dk,d], name="WQ")        # learnable param (Xavier)
#   Tensor(idxs=["dk","d"], sizes=[dk,d], name="WQ", init="zeros", learnable=False)  # data param
def Tensor(data=None, indices=None, name=None, idxs=None, sizes=None, init=None, learnable=None, scale: float = 0.02):
    from .namedtensor import nt as _nt
    b = _global_program.backend
    if data is not None:
        # Data path
        t = NamedTensor(data, indices, b, name=name)
        return t
    # Param-like path
    if idxs is None or sizes is None or name is None:
        raise ValueError("Tensor param creation requires idxs=[...], sizes=[...], and name=...")
    if init is None:
        # Xavier-ish
        if len(sizes) == 2:
            fan_in = sizes[1]; fan_out = sizes[0]
            sc = (2.0/(fan_in+fan_out+1e-8))**0.5
            data = b.randn(sizes) * sc
        else:
            data = b.randn(sizes) * scale
        if learnable is None:
            learnable = True
    else:
        if init == "eye":
            if len(sizes) != 2 or sizes[0] != sizes[1]:
                raise ValueError("eye init requires square 2D shape")
            data = b.eye(sizes[0])
        elif init == "zeros":
            data = b.zeros(sizes)
        elif init == "ones":
            data = b.ones(sizes)
        else:
            data = b.randn(sizes) * scale
        if learnable is None:
            learnable = False
    # Register tensor
    _global_program.set_tensor(name, _nt(data, idxs, b))
    if learnable:
        _global_program.mark_learnable(name)
    # Return a NamedTensor handle
    return _global_program[name]

# Keep NamedTensor export for advanced users
TensorClass = NamedTensor

# Global eval function for the global program
def eval(query: str):
    """Evaluate a query on the global program."""
    return _global_program.eval(query)

def clear_global_program():
    """Clear the global program (for testing)."""
    global _global_program
    # Preserve the current backend setting
    backend_name = _global_program.backend.name if _global_program else None
    if backend_name is None:
        backend_name = os.environ.get("TENSORLOGIC_BACKEND", "numpy")
    _global_program = Program(backend_name)

def mark_learnable(name: str):
    """Mark a tensor as learnable (if backend supports it)."""
    _global_program.mark_learnable(name)

def backward(loss_query: str = "Loss"):
    """Compute gradients of loss w.r.t learnable tensors."""
    return _global_program.backward(loss_query)

def sgd_step(lr: float = 1e-2):
    """Perform one SGD step on learnable tensors."""
    _global_program.sgd_step(lr)

def value(query: str):
    """Return backend-native array for a query."""
    return _global_program.value(query)

def set_tensor(name: str, tensor):
    """Set a tensor in the global program."""
    _global_program.set_tensor(name, tensor)

def equation(eq):
    """Add an equation to the global program."""
    _global_program.equation(eq)

def set_backend(name: str):
    """Set the global backend for tensor operations.
    
    Args:
        name: Backend name, one of: "numpy", "torch", "jax"
    
    Raises:
        ValueError: If backend name is invalid
        RuntimeError: If backend is not available (e.g., PyTorch/JAX not installed)
    """
    global _global_program
    # Set environment variable so get_backend(None) uses the new backend
    os.environ["TENSORLOGIC_BACKEND"] = name.lower()
    # Update the global program's backend
    _global_program.backend = get_backend(name.lower())

__all__ = [
    "get_backend",
    "NumpyBackend","TorchBackend","JaxBackend",
    "NamedTensor","Tensor","nt",
    "ntensor","Expr","Var","softmax",
    "Domain","relation_from_facts",
    "text_to_boolean_matrix",
    "eval","clear_global_program",
    "mark_learnable","backward","sgd_step","value","set_tensor","equation",
    "set_backend",
]

from .nn import param
__all__.append('param')


def softmax_on(scores_expr, lhs_indices):
    """Softmax normalized along dotted axis provided in lhs_indices, e.g., ("p","p2.")."""
    if not isinstance(lhs_indices, (list, tuple)):
        lhs_indices = (lhs_indices,)
    axis = None
    clean = []
    for s in lhs_indices:
        if isinstance(s, str) and s.endswith("."):
            if axis is not None:
                raise ValueError("Only one dotted axis allowed")
            axis = s[:-1]
            clean.append(axis)
        else:
            clean.append(s)
    if axis is None:
        raise ValueError("Provide a dotted axis like 'p2.' in lhs_indices")
    return softmax(scores_expr, axis=axis)

# Expose native Relation
from .relations import Relation
__all__.append("Relation")
__all__.append("softmax_on")
