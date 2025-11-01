from tensorlogic import Tensor, softmax
import numpy as np
from math import sqrt

# Create tensors with names
X = Tensor(np.array([[0.1, 0.2],[0.3, 0.4],[0.1,0.8]]), ["p","d"], name="X")
WQ = Tensor(np.eye(2), ["dk","d"], name="WQ")
WK = Tensor(np.eye(2), ["dk","d"], name="WK")
WV = Tensor(np.eye(2), ["dv","d"], name="WV")

# Initialize result tensors
Query = Tensor(np.zeros((3,2)), ["p","dk"], name="Query")
Key   = Tensor(np.zeros((3,2)), ["p","dk"], name="Key")
Val   = Tensor(np.zeros((3,2)), ["p","dv"], name="Val")
Comp  = Tensor(np.zeros((3,3)), ["p","p2"], name="Comp")
Attn  = Tensor(np.zeros((3,2)), ["p","dv"], name="Attn")

# Equations
Query["p","dk"] = WQ["dk","d"] * X["p","d"]
Key["p","dk"]   = WK["dk","d"] * X["p","d"]
Val["p","dv"]   = WV["dv","d"] * X["p","d"]

# Scaled dot-product + softmax along p2
scale = 1.0 / sqrt(2.0)
Scores = Tensor(np.zeros((3,3)), ["p","p2"], name="Scores")
Scores["p","p2"] = scale * (Query["p","dk"] * Key["p2","dk"])
Comp["p","p2"]   = softmax(Scores["p","p2"], axis="p2").ast
Attn["p","dv"]   = Comp["p","p2"] * Val["p2","dv"]

print("Comp row-sum≈1:", (Comp["p","p2"].eval().numpy().sum(axis=1)))
print("Attn shape:", Attn["p","dv"].eval().numpy().shape)
