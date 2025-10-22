
import numpy as np
from tensorlogic import Program, Domain, relation_from_facts

def test_symbolic_aunt_rule():
    People = Domain(["Alice","Bob","Charlie"])
    P = Program()

    Parent = relation_from_facts("Parent", ["x","y"],
        [("Alice","Bob"), ("Bob","Charlie")], {"x": People, "y": People}, P.backend)
    Sister = relation_from_facts("Sister", ["x","y"],
        [("Alice","Bob")], {"x": People, "y": People}, P.backend)

    P.set_tensor("Parent", Parent)
    P.set_tensor("Sister", Sister)

    P.equation("Aunt[x,z] = step(Sister[x,y] * Parent[y,z])")
    A = P.eval("Aunt[x,z]")
    arr = A.numpy()
    assert arr[0,2] == 1.0
    assert arr.sum() == 1.0
