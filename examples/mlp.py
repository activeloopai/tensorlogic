from tensorlogic import Program, nt

# Simple MLP: Y = step(W[i,j] * X[j])
P = Program(backend="numpy")
P.set_tensor("W", nt([[2.0, -1.0],[0.3,0.7]], ["i","j"]))
P.set_tensor("X", nt([1.0, 3.0], ["j"]))

P.equation("Y[i] = step(W[i,j] * X[j])")
Y = P.eval("Y[i]")
print("Y:", Y.numpy(), "indices:", Y.indices)
