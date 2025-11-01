
import numpy as np
from tensorlogic import Tensor, softmax, clear_global_program

def test_mlp_perceptron_step():
    clear_global_program()
    W = Tensor(np.array([[2.0, -1.0],[0.3, 0.7]]), ["i","j"], name="W")
    X = Tensor(np.array([1.0, 3.0]), ["j"], name="X")
    Y = Tensor(np.zeros(2), ["i"], name="Y")
    Y["i"] = (W["i","j"] * X["j"]).step()
    result = Y["i"].eval()
    y = result.numpy()
    assert y.shape == (2,)
    assert (y == np.array([0.0, 1.0])).all()

def test_attention_head_shapes():
    clear_global_program()
    # Use deterministic input for reproducible testing
    X = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.3, 0.1]]), ["p","d"], name="X")
    WQ = Tensor(np.eye(3), ["dk","d"], name="WQ")
    WK = Tensor(np.eye(3), ["dk","d"], name="WK")
    WV = Tensor(np.eye(3), ["dv","d"], name="WV")
    Query = Tensor(np.zeros((4,3)), ["p","dk"], name="Query")
    Key = Tensor(np.zeros((4,3)), ["p","dk"], name="Key")
    Val = Tensor(np.zeros((4,3)), ["p","dv"], name="Val")
    Comp = Tensor(np.zeros((4,4)), ["p","p2"], name="Comp")
    Attn = Tensor(np.zeros((4,3)), ["p","dv"], name="Attn")
    Query["p","dk"] = WQ["dk","d"] * X["p","d"]
    Key["p","dk"] = WK["dk","d"] * X["p","d"]
    Val["p","dv"] = WV["dv","d"] * X["p","d"]
    Comp["p","p2"] = softmax(Query["p","dk"] * Key["p2","dk"], axis="p2").ast
    Attn["p","dv"] = Comp["p","p2"] * Val["p2","dv"]
    
    # Evaluate results
    A = Attn["p","dv"].eval()
    attention_scores = Comp["p","p2"].eval()
    
    # Test 1: Shape correctness
    assert A.numpy().shape == (4, 3), f"Expected attention output shape (4, 3), got {A.numpy().shape}"
    assert attention_scores.numpy().shape == (4, 4), f"Expected attention scores shape (4, 4), got {attention_scores.numpy().shape}"
    
    # Test 2: Softmax normalization - attention scores should sum to 1 along p2 axis
    scores_np = attention_scores.numpy()
    row_sums = np.sum(scores_np, axis=1)  # Sum over p2 dimension
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"Attention scores don't sum to 1: {row_sums}"
    
    # Test 3: Attention scores should be non-negative (softmax output)
    assert np.all(scores_np >= 0), "Attention scores should be non-negative"
    assert np.all(scores_np <= 1), "Attention scores should be <= 1 (probability distribution)"
    
    # Test 4: Attention output is a valid weighted combination
    # Verify that attention output has correct shape and contains valid values
    A_np = A.numpy()
    assert not np.any(np.isnan(A_np)), "Attention output should not contain NaN values"
    assert not np.any(np.isinf(A_np)), "Attention output should not contain Inf values"
    
    # Test 5: Attention preserves dimensionality correctly
    assert A_np.shape[0] == X.numpy().shape[0], "Number of positions should match input"
    assert A_np.shape[1] == Val.numpy().shape[1], "Output dimension should match value dimension"
    
    # Test 6: Attention output has expected properties for weighted combination
    # Each output position should be a convex combination of value vectors
    # (weighted by attention scores)
    Query_np = Query["p","dk"].eval().numpy()
    Key_np = Key["p","dk"].eval().numpy()
    Val_np = Val["p","dv"].eval().numpy()
    
    # Verify Query, Key, Val are computed correctly (with identity matrices, they equal X)
    assert np.allclose(Query_np, X.numpy(), atol=1e-6), "Query should equal X with identity weight matrix"
    assert np.allclose(Key_np, X.numpy(), atol=1e-6), "Key should equal X with identity weight matrix"
    assert np.allclose(Val_np, X.numpy(), atol=1e-6), "Val should equal X with identity weight matrix"
    
    # Test 7: Attention mechanism produces output (not all zeros)
    assert np.any(A_np != 0), "Attention output should be non-zero"
    
    # Test 8: Attention scores have correct structure (each row is a probability distribution)
    # Each row should sum to 1 (already tested) and be non-negative (already tested)
    # Additionally, scores should be structured appropriately for attention
    assert scores_np.shape == (4, 4), "Attention scores should be (num_positions, num_positions)"
    
    # Test 9: Verify the mathematical correctness of attention computation
    # Attention output should be: sum over p2 of (Comp[p,p2] * Val[p2,dv])
    # This is equivalent to: scores @ Val
    expected_attention = scores_np @ Val_np
    # Verify that the einsum computation matches the manual computation
    assert expected_attention.shape == (4, 3), "Expected attention computation should have shape (4, 3)"
    assert np.allclose(A_np, expected_attention, atol=1e-6), \
        "Attention output should match the expected weighted combination of values"
