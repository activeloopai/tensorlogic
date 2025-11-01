
import numpy as np
from tensorlogic import Tensor, clear_global_program

def normalize(p, axis=-1):
    s = p.sum(axis=axis, keepdims=True)
    return p/s

def test_simple_bayes_net():
    clear_global_program()
    # Rain -> Sprinkler, Rain -> Wet
    # P(Rain): [r]
    PRain = Tensor(np.array([0.2, 0.8]), ["r"], name="PRain")  # r=0/1
    # P(Sprinkler | Rain): CPT[s, r]
    CPT_S = Tensor(np.array([[0.8, 0.2],
                             [0.2, 0.8]]), ["s","r"], name="CPT_S")
    # P(Wet | Rain): CPT[w, r]
    CPT_W = Tensor(np.array([[0.9, 0.3],
                             [0.1, 0.7]]), ["w","r"], name="CPT_W")
    # P(S) = CPT_S[s,r] * PRain[r]
    PS = Tensor(np.zeros(2), ["s"], name="PS")
    PS["s"] = CPT_S["s","r"] * PRain["r"]
    # P(W) = CPT_W[w,r] * PRain[r]
    PW = Tensor(np.zeros(2), ["w"], name="PW")
    PW["w"] = CPT_W["w","r"] * PRain["r"]
    # Joint over r,s,w (unnormalized for clarity): J[r,s,w] = PRain[r]*CPT_S[s,r]*CPT_W[w,r]
    J_tensor = Tensor(np.zeros((2,2,2)), ["r","s","w"], name="J")
    J_tensor["r","s","w"] = PRain["r"] * CPT_S["s","r"] * CPT_W["w","r"]
    J = J_tensor["r","s","w"].eval().numpy()
    # Marginal P(W): sum over r,s
    PW_marginal = J.sum(axis=(0,1))
    # Compare to direct
    PW2 = PW["w"].eval().numpy()
    assert np.allclose(PW_marginal, PW2)
