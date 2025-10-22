
from tensorlogic import Program, nt, softmax
from tensorlogic.nn import param
import numpy as np

P = Program()
# Stream[p,d] = Embeddings; we seed with some toy data (p=4, d=3)
Stream = P["Stream"]
P.set_tensor("Stream", nt(np.random.randn(4,3), ["p","d"]))

# Self-attention parameters (manually dims to match Stream: dk=dv=d=3 here)
WQ = param(P, "WQ", ["dk","d"], [3,3], init="eye")
WK = param(P, "WK", ["dk","d"], [3,3], init="eye")
WV = param(P, "WV", ["dv","d"], [3,3], init="eye")

Query = P["Query"]; Key = P["Key"]; Val = P["Val"]; Comp = P["Comp"]; Attn = P["Attn"]
Query["p","dk"] = WQ["dk","d"] * Stream["p","d"]
Key["p","dk"]   = WK["dk","d"] * Stream["p","d"]
Val["p","dv"]   = WV["dv","d"] * Stream["p","d"]

Comp["p","p2"]  = softmax(Query["p","dk"] * Key["p2","dk"], axis="p2").ast
Attn["p","dv"]  = Comp["p","p2"] * Val["p2","dv"]

# Residual + LayerNorm + tiny MLP
WS = param(P, "WS", ["d","d"], [3,3], init="randn")
MLP1 = param(P, "MLP1", ["d","d"], [3,3], init="randn")
MLP2 = param(P, "MLP2", ["d","d"], [3,3], init="randn")

Merged = P["Merged"]; NewStream = P["NewStream"]; Hidden = P["Hidden"]; NewStream2 = P["NewStream2"]
Merged["p","d"] = WS["d","d"] * Attn["p","dv"] + Stream["p","d"]
NewStream["p","d"] = Merged["p","d"].lnorm().ast

Hidden["p","d"] = (MLP1["d","d"] * NewStream["p","d"]).gelu().ast
NewStream2["p","d"] = (MLP2["d","d"] * Hidden["p","d"]) + NewStream["p","d"]

print("Block out shape:", P.eval("NewStream2[p,d]").numpy().shape)
