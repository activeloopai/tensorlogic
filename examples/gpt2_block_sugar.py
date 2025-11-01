
from tensorlogic import Tensor, softmax
from tensorlogic.nn import param
import numpy as np

# Stream[p,d] = Embeddings; we seed with some toy data (p=4, d=3)
Stream = Tensor(np.random.randn(4,3), ["p","d"], name="Stream")

# Self-attention parameters (manually dims to match Stream: dk=dv=d=3 here)
WQ = param("WQ", ["dk","d"], [3,3], init="eye")
WK = param("WK", ["dk","d"], [3,3], init="eye")
WV = param("WV", ["dv","d"], [3,3], init="eye")

Query = Tensor(np.zeros((4,3)), ["p","dk"], name="Query")
Key = Tensor(np.zeros((4,3)), ["p","dk"], name="Key")
Val = Tensor(np.zeros((4,3)), ["p","dv"], name="Val")
Comp = Tensor(np.zeros((4,4)), ["p","p2"], name="Comp")
Attn = Tensor(np.zeros((4,3)), ["p","dv"], name="Attn")

Query["p","dk"] = WQ["dk","d"] * Stream["p","d"]
Key["p","dk"]   = WK["dk","d"] * Stream["p","d"]
Val["p","dv"]   = WV["dv","d"] * Stream["p","d"]

Comp["p","p2"]  = softmax(Query["p","dk"] * Key["p2","dk"], axis="p2").ast
Attn["p","dv"]  = Comp["p","p2"] * Val["p2","dv"]

# Residual + LayerNorm + tiny MLP
WS = param("WS", ["d","d"], [3,3], init="randn")
MLP1 = param("MLP1", ["d","d"], [3,3], init="randn")
MLP2 = param("MLP2", ["d","d"], [3,3], init="randn")

Merged = Tensor(np.zeros((4,3)), ["p","d"], name="Merged")
NewStream = Tensor(np.zeros((4,3)), ["p","d"], name="NewStream")
Hidden = Tensor(np.zeros((4,3)), ["p","d"], name="Hidden")
NewStream2 = Tensor(np.zeros((4,3)), ["p","d"], name="NewStream2")

Merged["p","d"] = WS["d","d"] * Attn["p","dv"] + Stream["p","d"]
NewStream["p","d"] = Merged["p","d"].lnorm().ast

Hidden["p","d"] = (MLP1["d","d"] * NewStream["p","d"]).gelu().ast
NewStream2["p","d"] = (MLP2["d","d"] * Hidden["p","d"]) + NewStream["p","d"]

print("Block out shape:", NewStream2["p","d"].eval().numpy().shape)
