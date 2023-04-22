
import numpy as np
import gurobipy as gp

# Define the mapping matrix
# Rows correspond to nodes (A, B, C)
# Columns correspond to processing elements (PE1, PE2, PE3, PE4)
# graph3 = np.array([[0, 1, 1],  # Distance between PE1 and all other PEs
#                   [0, 0, 1],  # Distance between PE2 and all other PEs
#                   [0, 0, 0]]) # Distance between PE4 and all other PEs

graph = np.array([[0, 1, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])

# graph = np.loadtxt('resnet_connection_matrix.txt',delimiter=',')
# graph = np.loadtxt('transformer_connection_matrix.txt',delimiter=',')

# mapping_matrix3 = np.array([[1, 0, 0, 0],  # Node A mapped to PE1
#                            [0, 1, 0, 0],   # Node B mapped to PE2
#                            [0, 0, 0, 1],])

# mapping_matrix = np.array([[0, 0, 1, 0],  # Node A mapped to PE1
#                            [0, 0, 0, 1],  # Node B mapped to PE2
#                            [0, 1, 0, 0],
#                            [1, 0, 0, 0],]) # Node C mapped to PE3


# Define the distances between processing elements
distances = np.array([[0, 1, 1, 2],  # Distance between PE1 and all other PEs
                      [1, 0, 2, 1],  # Distance between PE2 and all other PEs
                      [1, 2, 0, 1],  # Distance between PE3 and all other PEs
                      [2, 1, 1, 0]]) # Distance between PE4 and all other PEs

def comm_matrix(graph,distances,mapping_space):
    '''
    Core function
    modeling the communication latency spent on NoC
    given graph and distances that are predefined
    mapping_space is the target variable
    '''
    return graph * (mapping_space@distances@mapping_space.T)

def find_parallel_nodes(graph):
    '''find if there are parallel nodes to execute simultaneous'''
    parallel_nodes = []
    n = len(graph)
    for i in range(n):
        for j in range(n):
                # find 1s in an array, use where()
                if np.where(graph[i,:] & graph[:,j])[0].size > 0:
                    parallel_nodes.append(np.where(graph[i,:] & graph[:,j])[0])
    return parallel_nodes

parallel_nodes_list = find_parallel_nodes(graph)

# Create a new model
m = gp.Model()

num_PE = 4
num_var = graph.shape[0]
mapping_space = m.addMVar((num_var,num_PE),vtype=gp.GRB.BINARY)
Q = np.eye(num_PE)
for i in range(num_var):
    m.addMConstr(np.full((1, num_PE), 1), mapping_space[i].T, '=', np.full(1, 1))
    for j in range(num_var):
        if i!=j:
            # different nodes cannot map to the same PE temporarily
            m.addMQConstr(Q, None, '=', 0, mapping_space[i], mapping_space[j])

comp_lat_per_node=np.array([i+1 for i in range(num_var)])
# comp_lat_per_node=m.addMVar((num_var),vtype=gp.GRB.INTEGER)
# for i in range(num_var):
#     m.addConstr(comp_lat_per_node[i]==i+1)

mapping_time=m.addMVar((num_var),vtype=gp.GRB.INTEGER)
# temporarily treat the first one as beginning
for i in range(num_var):
    if not graph[:,i].any():
        m.addConstr(mapping_time[i]==0)
    else:
        # have dependency
        # find previous node
        # fine nonzero index of graph[:i]:
        dep_id_list=np.where(graph[:,i])[0]
        for dep_id in dep_id_list:
            m.addConstr(mapping_time[i]>=mapping_time[dep_id]+comp_lat_per_node[dep_id]+
                        comm_matrix(graph,distances,mapping_space)[dep_id,i])

comm_lat=m.addVar(vtype=gp.GRB.INTEGER)
comm=comm_matrix(graph,distances,mapping_space).sum()
m.addConstr(comm_lat==comm)

comp_end=m.addMVar((num_var),vtype=gp.GRB.INTEGER)
end_time=mapping_time+comp_lat_per_node
for i in range(num_var):
    m.addConstr(comp_end[i]==end_time[i])

# m.Params.NonConvex = 2
comp_lat=m.addVar(vtype=gp.GRB.INTEGER)
m.addConstr(comp_lat==gp.max_([comp_end[i] for i in range(num_var)],constant=0))
m.setObjective((0+1)*(comp_lat), gp.GRB.MINIMIZE)
m.optimize()

print(f"Optimal objective value: {m.objVal}")
print(f"Solution values: mapping_space=\n{mapping_space.X}")
print(f"Solution values: mapping_time=\n{mapping_time.X}")
print(comm_matrix(graph,distances,mapping_space.X))