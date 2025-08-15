def analyze_polynomial(poly_def):
    """
    Analyze a polynomial definition to extract:
    1. Number of unique entries
    2. Number of entries that are reused
    
    Args:
        poly_def: List of lists representing polynomial terms
        
    Returns:
        tuple: (unique_count, reused_count)
    """
    # Flatten all entries from all terms
    all_entries = []
    for term in poly_def:
        all_entries.extend(term)
    
    # Count occurrences of each entry
    entry_counts = {}
    for entry in all_entries:
        entry_counts[entry] = entry_counts.get(entry, 0) + 1
    
    # Count unique entries (total distinct entries)
    unique_count = len(entry_counts)
    
    # Count reused entries (entries that appear more than once)
    reused_count = sum(1 for count in entry_counts.values() if count > 1)
    
    return unique_count, reused_count

def count_operations(poly_def):
    """
    Count the number of additions and multiplications in a polynomial definition.
    
    Args:
        poly_def: List of lists representing polynomial terms
        
    Returns:
        tuple: (num_adds, num_products)
        
    Rules:
    - Number of adds = number of sublists - 1 (terms are added together)
    - Number of products within a sublist = number of entries in that sublist - 1
    """
    # Number of additions: sublists - 1 (adding terms together)
    num_adds = len(poly_def) - 1 if len(poly_def) > 0 else 0
    
    # Number of multiplications: sum of (entries_in_sublist - 1) for each sublist
    num_products = 0
    for term in poly_def:
        if len(term) > 1:
            num_products += len(term) - 1
        # If a term has only 1 entry, it contributes 0 multiplications
    
    return num_adds, num_products

def analyze_polynomial_detailed(poly_def):
    """
    Detailed analysis showing which entries are reused and their counts
    """
    # Flatten all entries from all terms
    all_entries = []
    for term in poly_def:
        all_entries.extend(term)
    
    # Count occurrences of each entry
    entry_counts = {}
    for entry in all_entries:
        entry_counts[entry] = entry_counts.get(entry, 0) + 1
    
    # Separate unique and reused entries
    unique_entries = set(entry_counts.keys())
    reused_entries = {entry: count for entry, count in entry_counts.items() if count > 1}
    
    print(f"All entries: {sorted(unique_entries)}")
    print(f"Reused entries: {reused_entries}")
    print(f"Unique count: {len(unique_entries)}")
    print(f"Reused count: {len(reused_entries)}")
    
    return len(unique_entries), len(reused_entries)

# Test with the vanilla_gate example
if __name__ == "__main__":
    vanilla_gate = [
        ["q1", "w1"],
        ["q2", "w2"],
        ["q3", "w3"],
        ["qM", "w1", "w2"],
        ["qc"],
    ]
    
    print("=== Vanilla Gate Analysis ===")
    unique, reused = analyze_polynomial_detailed(vanilla_gate)
    adds, products = count_operations(vanilla_gate)
    print(f"\nResult: {unique} unique entries, {reused} reused entries")
    print(f"Operations: {adds} additions, {products} multiplications")
    
    # Test with jellyfish_gate
    jellyfish_gate = [
        ["q1", "w1"],
        ["q2", "w2"],
        ["q3", "w3"],
        ["q4", "w4"],
        ["q5", "w5"],
        ["qM1", "w1", "w2"],
        ["qM2", "w3", "w4"],
        ["qH1", "w1", "w1", "w1", "w1", "w1"],
        ["qH2", "w2", "w2", "w2", "w2", "w2"],
        ["qH3", "w3", "w3", "w3", "w3", "w3"],
        ["qH4", "w4", "w4", "w4", "w4", "w4"],
        ["qECC", "w1", "w2", "w3", "w4"],
        ["qc"],
    ]
    
    print("\n=== Jellyfish Gate Analysis ===")
    unique, reused = analyze_polynomial_detailed(jellyfish_gate)
    adds, products = count_operations(jellyfish_gate)
    print(f"\nResult: {unique} unique entries, {reused} reused entries")
    print(f"Operations: {adds} additions, {products} multiplications")
    
    # Test operation counting with a simple example
    print("\n=== Operation Counting Examples ===")
    
    # Example 1: ["q1", "w1"] -> 1 multiplication (q1 * w1), 0 additions (single term)
    simple_1 = [["q1", "w1"]]
    adds, products = count_operations(simple_1)
    print(f"Simple 1 term: {simple_1} -> {adds} adds, {products} products")
    
    # Example 2: ["q1", "w1"], ["q2"] -> 1 addition (term1 + term2), 1 multiplication (q1 * w1)
    simple_2 = [["q1", "w1"], ["q2"]]
    adds, products = count_operations(simple_2)
    print(f"Two terms: {simple_2} -> {adds} adds, {products} products")
    
    # Example 3: ["q1", "w1", "w2"] -> 2 multiplications (q1 * w1 * w2), 0 additions
    simple_3 = [["q1", "w1", "w2"]]
    adds, products = count_operations(simple_3)
    print(f"Triple product: {simple_3} -> {adds} adds, {products} products")
