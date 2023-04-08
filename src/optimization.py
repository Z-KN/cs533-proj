
import gurobipy as gp
import numpy as np
# Define the mapping matrix
# Rows correspond to nodes (A, B, C)
# Columns correspond to processing elements (PE1, PE2, PE3, PE4)
graph = np.array([[0, 1, 1],  # Distance between PE1 and all other PEs
                  [0, 0, 1],  # Distance between PE2 and all other PEs
                  [0, 0, 0]]) # Distance between PE4 and all other PEs

# mapping_matrix = np.array([0,1,2]),  # Node A mapped to PE1
mapping_matrix = np.array([[1, 0, 0, 0],  # Node A mapped to PE1
                           [0, 1, 0, 0],  # Node B mapped to PE2
                           [0, 0, 1, 0]]) # Node C mapped to PE3
# Define the distances between processing elements
distances = np.array([[0, 1, 1, 2],  # Distance between PE1 and all other PEs
                      [1, 0, 2, 1],  # Distance between PE2 and all other PEs
                      [1, 2, 0, 1],  # Distance between PE3 and all other PEs
                      [2, 1, 1, 0]]) # Distance between PE4 and all other PEs

# Compute the number of hops between each pair of nodes
hops = np.array([[0, 1, 2],  # Hops from node A to all other nodes
                 [1, 0, 1],  # Hops from node B to all other nodes
                 [2, 1, 0]]) # Hops from node C to all other nodes

# Compute the total cost
# print(distances[mapping_matrix])
print(graph@mapping_matrix * (mapping_matrix@distances))
# print(mapping_matrix.T@graph * (distances@mapping_matrix.T))
# print(distances[mapping_matrix].T@graph)
total_cost = np.sum(graph@mapping_matrix * (mapping_matrix@distances))
print(f"total_cost={total_cost}")
# Create a new model
m = gp.Model()

# mapping = np.stack((A_mapping,B_mapping,C_mapping),axis=0)
mapping = m.addMVar((3,4),vtype=gp.GRB.BINARY)
m.addMConstr(np.full((1, 4), 1), mapping[0].T, '=', np.full(1, 1))
m.addMConstr(np.full((1, 4), 1), mapping[1].T, '=', np.full(1, 1))
m.addMConstr(np.full((1, 4), 1), mapping[2].T, '=', np.full(1, 1))


val=(graph@mapping) * (mapping@distances)
m.setObjective(val.sum(), gp.GRB.MINIMIZE)
m.optimize()

print(f"Optimal objective value: {m.objVal}")
print(f"Solution values: mapping={mapping.X}")
