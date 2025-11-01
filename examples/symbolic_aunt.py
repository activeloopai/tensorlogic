from tensorlogic import Tensor, Domain, Relation
import numpy as np

People = Domain(["Alice","Bob","Charlie","Dora"])

Parent = Relation("Parent", People, People)
Sister = Relation("Sister", People, People)
Aunt   = Relation("Aunt",   People, People)

# Facts
Parent["Bob","Charlie"] = 1
Parent["Dora","Alice"]  = 1
Sister["Alice","Bob"]   = 1
Sister["Dora","Bob"]    = 1

# Rule
Aunt["x","z"] = (Sister["x","y"] * Parent["y","z"]).step()

# Evaluate
print("Aunt(Alice,Charlie)=", Aunt.value("Alice","Charlie"))
