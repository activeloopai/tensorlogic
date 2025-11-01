
import numpy as np
from tensorlogic import Tensor, clear_global_program

def test_embedding_membership():
    clear_global_program()
    rng = np.random.default_rng(0)
    D = 128

    # Emb[x,i] random unit vectors for 5 objects
    E = rng.standard_normal((5,D)); E = (E / np.linalg.norm(E, axis=1, keepdims=True))
    Emb = Tensor(E, ["x","i"], name="Emb")

    # Make set V = {0,2,3}
    V = np.zeros((5,), dtype=float); V[[0,2,3]] = 1.0
    V_tensor = Tensor(V, ["x"], name="V")

    # Superposition S[i] = V[x] * Emb[x,i]
    S = Tensor(np.zeros(D), ["i"], name="S")
    S["i"] = V_tensor["x"] * Emb["x","i"]
    
    # Dot with each A: Dots[x] = S[i] * Emb[x,i]
    Dots = Tensor(np.zeros(5), ["x"], name="Dots")
    Dots["x"] = S["i"] * Emb["x","i"]
    
    dots = Dots["x"].eval().numpy()
    assert dots[0] > 0.5 and dots[1] < 0.5
