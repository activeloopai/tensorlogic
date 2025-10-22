
from tensorlogic import Program, nt, softmax
import numpy as np

P = Program()
X, WQ, WK, WV, Query, Key, Val, Comp, Attn = P.vars("X","WQ","WK","WV","Query","Key","Val","Comp","Attn")

# tiny shapes: p=3, d=2, dk=2, dv=2
P.set_tensor("X",  nt(np.array([[0.1, 0.2],[0.3, 0.4],[0.1,0.8]]), ["p","d"]))
P.set_tensor("WQ", nt(np.eye(2), ["dk","d"]))
P.set_tensor("WK", nt(np.eye(2), ["dk","d"]))
P.set_tensor("WV", nt(np.eye(2), ["dv","d"]))

Query["p","dk"] = WQ["dk","d"] * X["p","d"]
Key["p","dk"]   = WK["dk","d"] * X["p","d"]
Val["p","dv"]   = WV["dv","d"] * X["p","d"]

# Comp[p,p2] = softmax(Query[p,dk]*Key[p2,dk], axis=p2)
Comp["p","p2"]  = softmax(Query["p","dk"] * Key["p2","dk"], axis="p2").ast
# Attn[p,dv] = Comp[p,p2] * Val[p2,dv]
Attn["p","dv"]  = Comp["p","p2"] * Val["p2","dv"]

print("Attn shape:", P.eval("Attn[p,dv]").numpy().shape)
