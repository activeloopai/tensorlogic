
from tensorlogic import Domain, Relation

def test_relation_facts_and_rule():
    P = Domain(["Alice","Bob","Charlie"])
    Parent = Relation("Parent", P, P)
    Sister = Relation("Sister", P, P)
    Aunt   = Relation("Aunt",   P, P)

    Parent["Bob","Charlie"] = 1
    Sister["Alice","Bob"] = 1

    Aunt["x","z"] = (Sister["x","y"] * Parent["y","z"]).step()
    assert Aunt.value("Alice","Charlie") == 1.0
