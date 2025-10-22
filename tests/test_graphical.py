
import numpy as np
from tensorlogic import Program, nt

def normalize(p, axis=-1):
    s = p.sum(axis=axis, keepdims=True)
    return p/s

def test_simple_bayes_net():
    # Rain -> Sprinkler, Rain -> Wet
    P = Program()
    # P(Rain): [r]
    P.set_tensor("PRain", nt(np.array([0.2, 0.8]), ["r"]))  # r=0/1
    # P(Sprinkler | Rain): CPT[s, r]
    P.set_tensor("CPT_S", nt(np.array([[0.8, 0.2],
                                       [0.2, 0.8]]), ["s","r"]))
    # P(Wet | Rain): CPT[w, r]
    P.set_tensor("CPT_W", nt(np.array([[0.9, 0.3],
                                       [0.1, 0.7]]), ["w","r"]))
    # P(S) = CPT_S[s,r] * PRain[r]
    P.equation("PS[s] = CPT_S[s,r] * PRain[r]")
    # P(W) = CPT_W[w,r] * PRain[r]
    P.equation("PW[w] = CPT_W[w,r] * PRain[r]")
    # Joint over r,s,w (unnormalized for clarity): J[r,s,w] = PRain[r]*CPT_S[s,r]*CPT_W[w,r]
    P.equation("J[r,s,w] = PRain[r] * CPT_S[s,r] * CPT_W[w,r]")
    J = P.eval("J[r,s,w]").numpy()
    # Marginal P(W): sum over r,s
    PW = J.sum(axis=(0,1))
    # Compare to direct
    PW2 = P.eval("PW[w]").numpy()
    assert np.allclose(PW, PW2)
