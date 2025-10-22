from tensorlogic import Program, Domain, relation_from_facts

People = Domain(["Alice","Bob","Charlie"])
P = Program()

Parent = relation_from_facts("Parent", ["x","y"],
    [("Alice","Bob"), ("Bob","Charlie")], {"x": People, "y": People}, P.backend)
Sister = relation_from_facts("Sister", ["x","y"],
    [("Alice","Bob")], {"x": People, "y": People}, P.backend)

P.set_tensor("Parent", Parent)
P.set_tensor("Sister", Sister)

# Aunt(x,z) <- Sister(x,y), Parent(y,z)
P.equation("Aunt[x,z] = step(Sister[x,y] * Parent[y,z])")

q = P.eval("Aunt[x,z]")
print(q.indices, q.numpy())
