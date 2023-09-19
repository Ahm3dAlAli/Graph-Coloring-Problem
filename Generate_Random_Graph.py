import numpy as np

def random_graph(vertices, sparsity):
    # Initialize empty graph as a dictionary
    G = {i: [] for i in range(1, vertices+1)}
    
    # Loop over all possible edges and add them with probability '1-sparsity'
    for i in range(1, vertices+1):
        for j in range(1, vertices+1):
            if i != j:  # No self-loops
                if np.random.rand() < 1-sparsity:
                    G[i].append(j)
                    
    return G
