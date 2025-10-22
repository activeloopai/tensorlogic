from tensorlogic import Program, nt
import numpy as np

P = Program()
# X[p,d]
P.set_tensor("X", nt(np.array([[0.1, 0.2],[0.3, 0.4],[0.1,0.8]]), ["p","d"]))
# WQ[dk,d], WK[dk,d], WV[dv,d]; use tiny sizes dk=dv=d=2
P.set_tensor("WQ", nt(np.eye(2), ["dk","d"]))
P.set_tensor("WK", nt(np.eye(2), ["dk","d"]))
P.set_tensor("WV", nt(np.eye(2), ["dv","d"]))

P.equation("Query[p,dk] = WQ[dk,d] * X[p,d]")
P.equation("Key[p,dk]   = WK[dk,d] * X[p,d]")
P.equation("Val[p,dv]   = WV[dv,d] * X[p,d]")
# Compute comparisons then values: Comp[p,p2] = softmax(Query[p,dk] * Key[p2,dk], axis='p2')
P.equation("Comp[p,p2]  = softmax(Query[p,dk] * Key[p2,dk], axis=p2)")
P.equation("Attn[p,dv]  = Comp[p,p2] * Val[p2,dv]")

A = P.eval("Attn[p,dv]")
print("Attn:", A.numpy())
