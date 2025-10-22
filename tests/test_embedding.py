
import numpy as np
from tensorlogic import Program, nt

def test_embedding_membership():
    rng = np.random.default_rng(0)
    D = 128
    backend = "numpy"
    P = Program(backend=backend)

    # Emb[x,i] random unit vectors for 5 objects
    E = rng.standard_normal((5,D)); E = (E / np.linalg.norm(E, axis=1, keepdims=True))
    P.set_tensor("Emb", nt(E, ["x","i"]))

    # Make set V = {0,2,3}
    V = np.zeros((5,), dtype=float); V[[0,2,3]] = 1.0
    P.set_tensor("V", nt(V, ["x"]))

    # Superposition S[i] = V[x] * Emb[x,i]
    P.equation("S[i] = V[x] * Emb[x,i]")
    # Dot with each A: Dots[x] = S[i] * Emb[x,i]
    P.equation("Dots[x] = S[i] * Emb[x,i]")
    dots = P.eval("Dots[x]").numpy()
    assert dots[0] > 0.5 and dots[1] < 0.5
