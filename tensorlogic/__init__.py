
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
Tensor = NamedTensor

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
