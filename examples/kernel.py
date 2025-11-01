
from tensorlogic import Tensor

X = Tensor([[1.,2.],
            [3.,4.]], ["i","j"], name="X")
K = Tensor([[0.,0.],[0.,0.]], ["i","i2"], name="K")

# K[i,i2] = (X[i,j] * X[i2,j])^2   using Pythonic sugar
K["i","i2"] = (X["i","j"] * X["i2","j"]) ** 2

out = K["i","i2"].eval().numpy()
print("K =", out)
