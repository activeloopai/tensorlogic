
from tensorlogic import Program, nt

P = Program()
K, X = P.vars("K","X")
P.set_tensor("X", nt([[1.,2.],
                      [3.,4.]], ["i","j"]))

# K[i,i2] = (X[i,j] * X[i2,j])^2   using Pythonic sugar
K["i","i2"] = (X["i","j"] * X["i2","j"]) ** 2

out = P.eval("K[i,i2]").numpy()
print("K =", out)
