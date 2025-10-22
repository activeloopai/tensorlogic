
from __future__ import annotations
import math
import os
from typing import Any, Tuple, Sequence, Optional

# The backend protocol (duck-typed)
class Backend:
    name: str = "base"

    def tensor(self, data, dtype=None):
        raise NotImplementedError

    def asarray(self, data):
        raise NotImplementedError

    def to_numpy(self, x):
        raise NotImplementedError

    def zeros(self, shape, dtype=None):
        raise NotImplementedError

    def ones(self, shape, dtype=None):
        raise NotImplementedError

    def rand(self, shape, seed: Optional[int]=None):
        raise NotImplementedError

    def randn(self, shape, seed: Optional[int]=None):
        raise NotImplementedError

    def eye(self, n, dtype=None):
        raise NotImplementedError

    def einsum(self, subscripts: str, *operands):
        raise NotImplementedError

    def sum(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def max(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def exp(self, x):
        raise NotImplementedError

    def log(self, x):
        raise NotImplementedError

    def sqrt(self, x):
        raise NotImplementedError

    def sin(self, x):
        raise NotImplementedError

    def cos(self, x):
        raise NotImplementedError

    def tanh(self, x):
        raise NotImplementedError

    def where(self, cond, a, b):
        raise NotImplementedError

    def greater(self, a, b):
        raise NotImplementedError

    def maximum(self, a, b):
        raise NotImplementedError

    def minimum(self, a, b):
        raise NotImplementedError

    def clip(self, x, a_min=None, a_max=None):
        raise NotImplementedError

    def transpose(self, x, axes=None):
        raise NotImplementedError

    def reshape(self, x, shape):
        raise NotImplementedError

    def arange(self, n):
        raise NotImplementedError

    # arithmetic
    def add(self, a, b): raise NotImplementedError
    def sub(self, a, b): raise NotImplementedError
    def mul(self, a, b): raise NotImplementedError
    def truediv(self, a, b): raise NotImplementedError
    def pow(self, a, p): raise NotImplementedError

    # nn utils
    def sigmoid(self, x):
        return self.truediv(1.0, self.add(1.0, self.exp(self.mul(-1.0, x))))

    def relu(self, x):
        return self.maximum(0.0, x)

    def softmax(self, x, axis=-1):
        # numerically stable
        m = self.max(x, axis=axis, keepdims=True)
        e = self.exp(self.sub(x, m))
        s = self.sum(e, axis=axis, keepdims=True)
        return self.truediv(e, s)

    def layer_norm(self, x, axis=-1, eps: float = 1e-5):
        mean = self.sum(x, axis=axis, keepdims=True) / x.shape[axis]
        var = self.sum(self.pow(self.sub(x, mean), 2.0), axis=axis, keepdims=True) / x.shape[axis]
        return self.truediv(self.sub(x, mean), self.sqrt(self.add(var, eps)))

    def step(self, x):
        # Heaviside: 1 if x>0 else 0
        return self.where(self.greater(x, 0.0), 1.0, 0.0)

    # autograd introspection
    def requires_grad(self, x) -> bool:
        return False

    def is_autodiff(self) -> bool:
        return False


class NumpyBackend(Backend):
    name = "numpy"
    def __init__(self):
        import numpy as np
        self.np = np

    def tensor(self, data, dtype=None):
        return self.np.array(data, dtype=dtype)

    def asarray(self, data):
        return self.np.asarray(data)

    def to_numpy(self, x):
        return self.np.asarray(x)

    def zeros(self, shape, dtype=None):
        return self.np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return self.np.ones(shape, dtype=dtype)

    def rand(self, shape, seed: Optional[int]=None):
        rng = self.np.random.default_rng(seed)
        return rng.random(shape)

    def randn(self, shape, seed: Optional[int]=None):
        rng = self.np.random.default_rng(seed)
        return rng.standard_normal(shape)

    def eye(self, n, dtype=None):
        return self.np.eye(n, dtype=dtype)

    def einsum(self, subscripts: str, *operands):
        return self.np.einsum(subscripts, *operands)

    def sum(self, x, axis=None, keepdims=False):
        return self.np.sum(x, axis=axis, keepdims=keepdims)

    def max(self, x, axis=None, keepdims=False):
        return self.np.max(x, axis=axis, keepdims=keepdims)

    def exp(self, x): return self.np.exp(x)
    def log(self, x): return self.np.log(x)
    def sqrt(self, x): return self.np.sqrt(x)
    def sin(self, x): return self.np.sin(x)
    def cos(self, x): return self.np.cos(x)
    def tanh(self, x): return self.np.tanh(x)
    def where(self, cond, a, b): return self.np.where(cond, a, b)
    def greater(self, a, b): return self.np.greater(a, b)
    def maximum(self, a, b): return self.np.maximum(a, b)
    def minimum(self, a, b): return self.np.minimum(a, b)
    def clip(self, x, a_min=None, a_max=None): return self.np.clip(x, a_min, a_max)
    def transpose(self, x, axes=None): return self.np.transpose(x, axes)
    def reshape(self, x, shape): return self.np.reshape(x, shape)
    def arange(self, n): return self.np.arange(n)

    def add(self, a, b): return self.asarray(a) + self.asarray(b)
    def sub(self, a, b): return self.asarray(a) - self.asarray(b)
    def mul(self, a, b): return self.asarray(a) * self.asarray(b)
    def truediv(self, a, b): return self.asarray(a) / self.asarray(b)
    def pow(self, a, p): return self.asarray(a) ** p


class TorchBackend(Backend):
    name = "torch"
    def __init__(self):
        import torch
        self.torch = torch

    def tensor(self, data, dtype=None):
        if dtype is None:
            return self.torch.as_tensor(data)
        return self.torch.as_tensor(data, dtype=dtype)

    def asarray(self, data):
        return self.torch.as_tensor(data)

    def to_numpy(self, x):
        if isinstance(x, self.torch.Tensor):
            return x.detach().cpu().numpy()
        return self.torch.as_tensor(x).detach().cpu().numpy()

    def zeros(self, shape, dtype=None):
        return self.torch.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return self.torch.ones(shape, dtype=dtype)

    def rand(self, shape, seed: Optional[int]=None):
        if seed is not None:
            self.torch.manual_seed(seed)
        return self.torch.rand(shape)

    def randn(self, shape, seed: Optional[int]=None):
        if seed is not None:
            self.torch.manual_seed(seed)
        return self.torch.randn(shape)

    def eye(self, n, dtype=None):
        return self.torch.eye(n, dtype=dtype)

    def einsum(self, subscripts: str, *operands):
        return self.torch.einsum(subscripts, *operands)

    def sum(self, x, axis=None, keepdims=False):
        return self.torch.sum(x, dim=axis, keepdim=keepdims)

    def max(self, x, axis=None, keepdims=False):
        if axis is None:
            return self.torch.max(x)
        m, _ = self.torch.max(x, dim=axis, keepdim=keepdims)
        return m

    def exp(self, x): return self.torch.exp(x)
    def log(self, x): return self.torch.log(x)
    def sqrt(self, x): return self.torch.sqrt(x)
    def sin(self, x): return self.torch.sin(x)
    def cos(self, x): return self.torch.cos(x)
    def tanh(self, x): return self.torch.tanh(x)
    def where(self, cond, a, b): return self.torch.where(cond, a, b)
    def greater(self, a, b): return self.torch.gt(a, b)
    def maximum(self, a, b): return self.torch.maximum(self.asarray(a), self.asarray(b))
    def minimum(self, a, b): return self.torch.minimum(self.asarray(a), self.asarray(b))
    def clip(self, x, a_min=None, a_max=None): return self.torch.clamp(x, min=a_min, max=a_max)
    def transpose(self, x, axes=None): return x.permute(axes) if axes is not None else x.t()
    def reshape(self, x, shape): return x.reshape(shape)
    def arange(self, n): return self.torch.arange(n)

    def add(self, a, b): return self.asarray(a) + self.asarray(b)
    def sub(self, a, b): return self.asarray(a) - self.asarray(b)
    def mul(self, a, b): return self.asarray(a) * self.asarray(b)
    def truediv(self, a, b): return self.asarray(a) / self.asarray(b)
    def pow(self, a, p): return self.asarray(a) ** p

    def requires_grad(self, x) -> bool:
        return isinstance(x, self.torch.Tensor) and bool(getattr(x, "requires_grad", False))

    def is_autodiff(self) -> bool:
        return True


class JaxBackend(Backend):
    name = "jax"
    def __init__(self):
        import jax
        import jax.numpy as jnp
        self.jax = jax
        self.jnp = jnp

    def tensor(self, data, dtype=None):
        return self.jnp.array(data, dtype=dtype)

    def asarray(self, data):
        return self.jnp.asarray(data)

    def to_numpy(self, x):
        return self.jnp.asarray(x)

    def zeros(self, shape, dtype=None):
        return self.jnp.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return self.jnp.ones(shape, dtype=dtype)

    def rand(self, shape, seed: Optional[int]=None):
        key = self.jax.random.PRNGKey(0 if seed is None else seed)
        return self.jax.random.uniform(key, shape)

    def randn(self, shape, seed: Optional[int]=None):
        key = self.jax.random.PRNGKey(0 if seed is None else seed)
        return self.jax.random.normal(key, shape)

    def eye(self, n, dtype=None):
        return self.jnp.eye(n, dtype=dtype)

    def einsum(self, subscripts: str, *operands):
        return self.jnp.einsum(subscripts, *operands)

    def sum(self, x, axis=None, keepdims=False):
        return self.jnp.sum(x, axis=axis, keepdims=keepdims)

    def max(self, x, axis=None, keepdims=False):
        return self.jnp.max(x, axis=axis, keepdims=keepdims)

    def exp(self, x): return self.jnp.exp(x)
    def log(self, x): return self.jnp.log(x)
    def sqrt(self, x): return self.jnp.sqrt(x)
    def sin(self, x): return self.jnp.sin(x)
    def cos(self, x): return self.jnp.cos(x)
    def tanh(self, x): return self.jnp.tanh(x)
    def where(self, cond, a, b): return self.jnp.where(cond, a, b)
    def greater(self, a, b): return self.jnp.greater(a, b)
    def maximum(self, a, b): return self.jnp.maximum(self.asarray(a), self.asarray(b))
    def minimum(self, a, b): return self.jnp.minimum(self.asarray(a), self.asarray(b))
    def clip(self, x, a_min=None, a_max=None): return self.jnp.clip(x, a_min, a_max)
    def transpose(self, x, axes=None): return self.jnp.transpose(x, axes)
    def reshape(self, x, shape): return self.jnp.reshape(x, shape)
    def arange(self, n): return self.jnp.arange(n)

    def add(self, a, b): return self.asarray(a) + self.asarray(b)
    def sub(self, a, b): return self.asarray(a) - self.asarray(b)
    def mul(self, a, b): return self.asarray(a) * self.asarray(b)
    def truediv(self, a, b): return self.asarray(a) / self.asarray(b)
    def pow(self, a, p): return self.asarray(a) ** p

    def is_autodiff(self) -> bool:
        return True


def get_backend(name: Optional[str] = None) -> Backend:
    if name is None:
        name = os.environ.get("TENSORLOGIC_BACKEND", "numpy")
    name = name.lower()
    if name == "numpy":
        return NumpyBackend()
    if name == "torch":
        try:
            return TorchBackend()
        except Exception as e:
            raise RuntimeError("PyTorch backend requested but not available.") from e
    if name == "jax":
        try:
            return JaxBackend()
        except Exception as e:
            raise RuntimeError("JAX backend requested but not available.") from e
    raise ValueError(f"Unknown backend '{name}'. Choose from: numpy, torch, jax.")
