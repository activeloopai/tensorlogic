
from __future__ import annotations
from typing import Dict, List, Tuple, Sequence, Any, Optional
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
                        backend: Backend=None) -> NamedTensor:
    """Build a Boolean (0/1 float) tensor from a list of facts (tuples of symbols)."""
    if backend is None:
        backend = get_backend(None)
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
