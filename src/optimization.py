
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

graph = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0]])

# graph = np.loadtxt('resnet_adj_mat.txt',delimiter=',').astype(int)
# graph = np.loadtxt('transformer_adj_mat.txt',delimiter=',').astype(int)

# mapping_matrix3 = np.array([[1, 0, 0, 0],  # Node A mapped to PE1
#                            [0, 1, 0, 0],   # Node B mapped to PE2
#                            [0, 0, 0, 1],])

# mapping_matrix = np.array([[0, 0, 1, 0],  # Node A mapped to PE1
#                            [0, 0, 0, 1],  # Node B mapped to PE2
#                            [0, 1, 0, 0],
#                            [1, 0, 0, 0],]) # Node C mapped to PE3
# mapping_matrix =np.array([[-0, -0, 1,  0, -0, -0,],
#                         [-0, -0, -0, 1,  0, -0,],
#                         [-0, -0, -0, -0, 1,  0,],
#                         [-0, -0,  0, -0, -0, 1,]])

# print(graph)
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
    # both return are ok in terms of sum(),
    # but slightly different meaning
    # print(mapping_space@distances@mapping_space.T)
    return graph * (mapping_space@distances@mapping_space.T)
    # print(mapping_space.T@graph@mapping_space)
    # return mapping_space.T@graph@mapping_space * (distances)

def find_parallel_nodes(graph):
    '''find if there are parallel nodes to execute simultaneous'''
    parallel_nodes = []
    n = len(graph)
    for i in range(n):
        for j in range(n):
                # find 1s in an array, use where()
                if np.where(graph[i,:] & graph[:,j])[0].size > 1:
                    parallel_nodes.append(np.where(graph[i,:] & graph[:,j])[0])
    return parallel_nodes


# parallel_nodes_list = find_parallel_nodes(graph)
# print(parallel_nodes_list)

def gen_dis(row_num, col_num):
    # create a matrix of zeros with shape (i*j, i*j)
    matrix = np.zeros((row_num*col_num, row_num*col_num),dtype=np.int32)
    # loop through each row and column
    for i1 in range(row_num):
        for j1 in range(col_num):
            # calculate the index of the current position in the flattened matrix
            index1 = i1 * col_num + j1
            # loop through each other row and column
            for i2 in range(row_num):
                for j2 in range(col_num):
                    # calculate the index of the other position in the flattened matrix
                    index2 = i2 * col_num + j2
                    # calculate the Manhattan distance between the two positions
                    distance = abs(i1 - i2) + abs(j1 - j2)
                    # store the distance in the matrix
                    matrix[index1][index2] = distance

    return matrix

distances = gen_dis(6, 6)
print(distances)

from collections import deque

def bfs(adj_matrix):
    # Initialize the appended array and the queue
    num_nodes = len(adj_matrix)
    appended = np.zeros(num_nodes,dtype=np.int32)
    last_appended = np.zeros(num_nodes,dtype=np.int32)

    visited = np.zeros(num_nodes,dtype=np.int32)
    queue = deque()
    
    subgraph=[]
    # Start the BFS from node 0
    queue.append(0)
    appended[0] = 1
    depth_level = [0] * num_nodes
    nodes_at_level = [[0]]
    
    # Traverse the graph using BFS
    while len(queue) > 0:
        cur_node = queue.popleft()
        # visited[cur_node] = 1
        cur_depth = depth_level[cur_node]
        
        all_deps_appended = True  # Flag for checking if all dependencies are appended
                        
        for neighbor in np.where(adj_matrix[cur_node,:] == 1)[0]:
            if appended[neighbor] == 0:
                queue.append(neighbor)
                appended[neighbor] = 1
                depth_level[neighbor] = cur_depth + 1
                
                # all_deps_appended = True
                if len(nodes_at_level) <= depth_level[neighbor]:
                    nodes_at_level.append([neighbor])
                else:
                    nodes_at_level[depth_level[neighbor]].append(neighbor)
                # print(nodes_at_level)
                for dep in np.where(adj_matrix[:,neighbor] == 1)[0]:
                    if dep not in appended:
                        all_deps_appended = False
                        break
        # print(all_deps_appended)
        if all_deps_appended:
            new_subgraph = appended.copy() - last_appended
            if(new_subgraph.any()):
                # avoid all zero, occuring at the end
                subgraph.append(new_subgraph)
            last_appended = appended.copy()
            # print("SUBG",subgraph)
        # todo: remove the augementation part
    return depth_level, nodes_at_level, subgraph

print(bfs(graph))


# end_node_id=np.argwhere(np.array([not (graph[i].any()) for i in range(graph.shape[0])]))
# print(end_node_id)
# print(comm_matrix(graph,distances,mapping_matrix))
# Create a new model
# m = gp.Model()

# num_PE = distances.shape[0]
# print(num_PE)
# num_var = graph.shape[0]
# mapping_space = m.addMVar((num_var,num_PE),vtype=gp.GRB.BINARY)
# Q = np.eye(num_PE)
# for i in range(num_var):
#     print(i)
#     m.addMConstr(np.full((1, num_PE), 1), mapping_space[i].T, '=', np.full(1, 1))
#     for j in range(num_var):
#         if i!=j:
#             # different nodes cannot map to the same PE temporarily
#             m.addMQConstr(Q, None, '=', 0, mapping_space[i], mapping_space[j])

# comp_lat_per_node=np.array([i+1 for i in range(num_var)])
# # comp_lat_per_node=m.addMVar((num_var),vtype=gp.GRB.INTEGER)
# # for i in range(num_var):
# #     m.addConstr(comp_lat_per_node[i]==i+1)

# print("HERE2")
# mapping_time=m.addMVar((num_var),vtype=gp.GRB.INTEGER)
# # temporarily treat the first one as beginning
# for i in range(num_var):
#     print(i)
#     if not graph[:,i].any():
#         m.addConstr(mapping_time[i]==0)
#     else:
#         # have dependency
#         # find previous node
#         # fine nonzero index of graph[:i]:
#         dep_id_list=np.where(graph[:,i])[0]
#         for dep_id in dep_id_list:
#             m.addConstr(mapping_time[i]>=mapping_time[dep_id]+comp_lat_per_node[dep_id]+
#                         comm_matrix(graph,distances,mapping_space)[dep_id,i])

# comm_lat=m.addVar(vtype=gp.GRB.INTEGER)
# comm=comm_matrix(graph,distances,mapping_space).sum()
# m.addConstr(comm_lat==comm)

# comp_end=m.addMVar((num_var),vtype=gp.GRB.INTEGER)
# end_time=mapping_time+comp_lat_per_node
# for i in range(num_var):
#     m.addConstr(comp_end[i]==end_time[i])

# print("HERE1!")
# # m.Params.NonConvex = 2
# comp_lat=m.addVar(vtype=gp.GRB.INTEGER)
# m.addConstr(comp_lat==gp.max_([comp_end[i] for i in range(num_var)],constant=0))
# # m.addConstr(comp_lat==comp_end[end_node_id])
# m.setObjective((0+1)*(comp_lat), gp.GRB.MINIMIZE)
# m.optimize()

# print(f"Optimal objective value: {m.objVal}")
# print(f"Solution values: mapping_space=\n{mapping_space.X}")
# print(f"Solution values: mapping_time=\n{mapping_time.X}")
# print(comm_matrix(graph,distances,mapping_space.X))