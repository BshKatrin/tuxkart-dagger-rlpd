import numpy as np

def compute_curvature(nodes):
    """
    Compute the curvature of a segment of track using centerline nodes.
    Curvature is estimated based on angle differences between consecutive nodes.
    """
    if nodes is None or len(nodes) < 2:
        print("Not enough nodes to compute curvature!") 
        return 0  # Not enough points to compute curvature
    
    #print("Hello from compute_curvature function")
    
    nodes = np.asarray(nodes)

    direction_changes = []
    for i in range(len(nodes) - 1):
        dx = nodes[i + 1][0] - nodes[i][0]  # Extract X
        dz = nodes[i + 1][2] - nodes[i][2]  # Extract Z
        angle = np.arctan2(dz, dx)
        direction_changes.append(angle)

    if len(direction_changes) > 1:
        curvature = np.mean(np.diff(direction_changes)) * 10  # Increase sensitivity by scaling
    else:
        curvature = 0

    #print(f"Computed Curvature: {curvature}")  
    return curvature
