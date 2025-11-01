from tensorlogic import Tensor
import numpy as np

# Create tensors with names
X = Tensor(np.array([[0.1, 0.2],[0.3, 0.4],[0.1,0.8]]), ["p","d"], name="X")
WQ = Tensor(np.eye(2), ["dk","d"], name="WQ")
WK = Tensor(np.eye(2), ["dk","d"], name="WK")
WV = Tensor(np.eye(2), ["dv","d"], name="WV")

# Initialize result tensors
Query = Tensor(np.zeros((3,2)), ["p","dk"], name="Query")
Key = Tensor(np.zeros((3,2)), ["p","dk"], name="Key")
Val = Tensor(np.zeros((3,2)), ["p","dv"], name="Val")
Comp = Tensor(np.zeros((3,3)), ["p","p2"], name="Comp")
Attn = Tensor(np.zeros((3,2)), ["p","dv"], name="Attn")

# Set up equations for Query, Key, Val
Query["p","dk"] = WQ["dk","d"] * X["p","d"]
Key["p","dk"]   = WK["dk","d"] * X["p","d"]
Val["p","dv"]   = WV["dv","d"] * X["p","d"]

# Compute attention scores using deferred eval (do all ops symbolically, then eval at the end)
# The system automatically sums over 'dk' because it's not in the output indices ["p", "p2"]
Comp["p","p2"] = Query["p","dk"] * Key["p2","dk"]  # (3,3) attention scores, einsum on shared 'dk'

scores = Comp["p","p2"].eval().numpy()
print("Raw scores:", scores)