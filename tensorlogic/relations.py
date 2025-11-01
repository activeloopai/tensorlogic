
from __future__ import annotations
from typing import Dict, List, Tuple, Sequence, Any, Optional, Union
from .backend import get_backend, Backend
from .namedtensor import NamedTensor

class Domain:
    """Finite domain of symbols -> integer ids."""
    def __init__(self, symbols: Sequence[str]):
        self.sym2id: Dict[str,int] = {s:i for i,s in enumerate(symbols)}
        self.id2sym: List[str] = list(symbols)

    def id(self, sym: str) -> int:
        return self.sym2id[sym]

    def __len__(self):
        return len(self.id2sym)

def relation_from_facts(name: str,
                        indices: Sequence[str],
                        facts: Sequence[Tuple[str, ...]],
                        domains: Dict[str, Domain],
                        backend: Union[Backend, str, None] = None) -> NamedTensor:
    """Build a Boolean (0/1 float) tensor from a list of facts (tuples of symbols)."""
    if backend is None or isinstance(backend, str):
        backend = get_backend(backend)
    shape = [ len(domains[idx]) for idx in indices ]
    data = backend.zeros(shape)
    # mark ones
    for fact in facts:
        coords = tuple(domains[idx].id(sym) for idx, sym in zip(indices, fact))
        # set 1.0
        # For numpy / jax: fancy indexing; for torch, use indexing assignment style
        if backend.name == "torch":
            import torch
            # work around by converting to numpy then back (small sizes assumed)
            tmp = data.detach().cpu().numpy()
            tmp[coords] = 1.0
            data = backend.asarray(tmp)
        else:
            arr = data
            arr[coords] = 1.0
    return NamedTensor(data, tuple(indices), backend)


class Relation:
    """
    Native relation with Python assignments.
      Facts: R["Alice","Bob"] = 1
      Rules: H["x","z"] = (R["x","y"] * S["y","z"]).step()
    Uses a NamedTensor under the hood (registered with the global program).
    """
    def __init__(self, name: str, *domains: Domain):
        if not domains:
            raise ValueError("Provide at least one Domain")
        self.name = name
        self.domains = domains
        # default indices names: x,y,z,u,v,w as needed
        base = list("xyzuvw")
        if len(domains) <= len(base):
            self.indices = tuple(base[:len(domains)])
        else:
            self.indices = tuple([f"v{k}" for k in range(len(domains))])
        # create a zeros tensor in the program (data tensor by default)
        shape = [len(d) for d in domains]
        b = get_backend(None)
        data = b.zeros(shape)
        self.tensor = NamedTensor(data, self.indices, b, name=name)

    def __getitem__(self, idxs):
        if not isinstance(idxs, (list,tuple)):
            idxs = (idxs,)
        if len(idxs) != len(self.indices):
            raise ValueError("Wrong arity in relation indexing")
        # return expression handle for rules: delegate to NamedTensor
        return self.tensor[idxs]

    def __setitem__(self, idxs, rhs):
        if not isinstance(idxs, (list,tuple)):
            idxs = (idxs,)
        if len(idxs) != len(self.indices):
            raise ValueError("Wrong arity in relation indexing")
        # detect fact: all entries are constants existing in respective domains
        is_const = []
        for pos, key in enumerate(idxs):
            is_const.append(key in self.domains[pos].sym2id)
        if all(is_const):
            # set 1.0 in underlying data and update program tensor
            coords = tuple(self.domains[pos].id(idxs[pos]) for pos in range(len(idxs)))
            arr = self.tensor.data
            # backend-agnostic set
            if hasattr(arr, "__setitem__"):
                arr[coords] = 1.0
            else:
                # fallback: convert to numpy, set, wrap back (rare)
                import numpy as _np
                tmp = arr if isinstance(arr, _np.ndarray) else arr.numpy()
                tmp[coords] = 1.0
                arr = tmp
            # re-register the tensor to update global program storage
            from . import _global_program
            self.tensor = NamedTensor(arr, self.indices, _global_program.backend, name=self.name)
            _global_program.set_tensor(self.name, self.tensor)
            return
        # Else: rule; delegate to NamedTensor equation path
        self.tensor[idxs] = rhs


    def value(self, *names):
        if len(names) != len(self.domains):
            raise ValueError("Arity mismatch in value(...)")
        coords = tuple(self.domains[i].id(names[i]) for i in range(len(names)))
        # Evaluate the full relation (facts + rules)
        evaluated = self.tensor[self.indices].eval().numpy()
        return float(evaluated[coords])
