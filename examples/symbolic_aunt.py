from tensorlogic import Tensor, Domain, relation_from_facts
import numpy as np
People = Domain(["Alice","Bob","Charlie"])

Parent_data = relation_from_facts("Parent", ["x","y"],
    [("Alice","Bob"), ("Bob","Charlie")], {"x": People, "y": People}, "numpy")
Sister_data = relation_from_facts("Sister", ["x","y"],
    [("Alice","Bob")], {"x": People, "y": People}, "numpy")

Parent = Tensor(Parent_data.data, Parent_data.indices, name="Parent")
Sister = Tensor(Sister_data.data, Sister_data.indices, name="Sister")

# Aunt(x,z) <- Sister(x,y), Parent(y,z)
Aunt = Tensor(np.zeros((3,3)), ["x","z"], name="Aunt")
Aunt["x","z"] = (Sister["x","y"] * Parent["y","z"]).step()

q = Aunt["x","z"].eval()
print(q.indices, q.numpy())
