
from __future__ import annotations
import re
from typing import List, Dict, Sequence, Tuple, Optional
from .backend import get_backend, Backend
from .namedtensor import NamedTensor

def text_to_boolean_matrix(text: str, vocab: Optional[Sequence[str]]=None, backend: Backend=None):
    """Return (matrix, positions, vocab) where matrix[pos, token] = 1 if token occurs at pos."""
    if backend is None:
        backend = get_backend(None)
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    if vocab is None:
        vocab = sorted(set(tokens))
    tok2id = {t:i for i,t in enumerate(vocab)}
    mat = backend.zeros((len(tokens), len(vocab)))
    # set ones
    if backend.name == "torch":
        tmp = mat.detach().cpu().numpy()
        for i, tok in enumerate(tokens):
            j = tok2id.get(tok)
            if j is not None:
                tmp[i, j] = 1.0
        import torch
        mat = torch.as_tensor(tmp, dtype=torch.float32)
    else:
        for i, tok in enumerate(tokens):
            j = tok2id.get(tok)
            if j is not None:
                mat[i, j] = 1.0
    return NamedTensor(mat, ("pos","tok"), backend), tokens, list(vocab)
