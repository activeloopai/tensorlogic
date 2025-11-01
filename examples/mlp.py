from tensorlogic import Tensor

# Simple MLP: Y = step(W[i,j] * X[j])
W = Tensor([[2.0, -1.0],[0.3,0.7]], ["i","j"], name="W")
X = Tensor([1.0, 3.0], ["j"], name="X")
Y = Tensor([0.0, 0.0], ["i"], name="Y")

Y["i"] = (W["i","j"] * X["j"]).step()
result = Y["i"].eval()
print("Y:", result.numpy(), "indices:", result.indices)
