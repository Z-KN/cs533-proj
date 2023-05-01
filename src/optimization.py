
import numpy as np
import gurobipy as gp
import json
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
# graph = np.array([[0, 0, 1, 0],
#                   [0, 0, 0, 1],
#                   [0, 0, 0, 1],
#                   [0, 0, 0, 0]])

# graph = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 1, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 1, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 1, 1],
#                   [0, 0, 0, 0, 0, 0, 0, 1],
#                   [0, 0, 0, 0, 0, 0, 0, 0]])

graph = np.array([[0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0]])

# graph = np.loadtxt('resnet_adj_mat.txt',delimiter=',').astype(int)
graph = np.loadtxt('transformer_adj_mat.txt',delimiter=',').astype(int)

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

# with open('resnet_simba_timespace.json') as f:
with open('transformer_simba_timespace.json') as f:
    comp_lat_per_node = np.array(json.load(f))
    # print(comp_lat_per_node)

a=comp_lat_per_node.copy()
row_PE = 5
col_PE = 5
num_PE = row_PE * col_PE
distances = gen_dis(row_PE, col_PE)
# print(distances)

from collections import deque

def bfs_partition(adj_matrix):
    # Initialize the visited array and the queue
    num_nodes = len(adj_matrix)
    visited = np.zeros(num_nodes,dtype=np.int32)
    last_visited = np.zeros(num_nodes,dtype=np.int32)

    queue = deque()
    
    partition=[]
    starting_nodes=[i for i in range(num_nodes) if not adj_matrix[:,i].any()]
    for i in starting_nodes:
        queue.append(i)
    visited[starting_nodes] = 1
    depth_level = [0] * num_nodes
    nodes_at_depth_levels = [starting_nodes]
    
    # Traverse the graph using BFS
    while len(queue) > 0:
        cur_node = queue.popleft()
        cur_depth = depth_level[cur_node]
        all_deps_visited = True  # Flag for checking if all dependencies are visited
        
        neighbors = np.where(adj_matrix[cur_node,:] == 1)[0]
        for neighbor in neighbors:
            if visited[neighbor] == 0:
                queue.append(neighbor)
                visited[neighbor] = 1
                depth_level[neighbor] = cur_depth + 1
                
                # all_deps_visited = True
                if len(nodes_at_depth_levels) <= depth_level[neighbor]:
                    nodes_at_depth_levels.append([neighbor])
                else:
                    nodes_at_depth_levels[depth_level[neighbor]].append(neighbor)
                # print(nodes_at_depth_levels)
        
        for neighbor in neighbors:
            for dep in np.where(adj_matrix[:,neighbor] == 1)[0]:
                if not visited[dep]:
                    all_deps_visited = False
                    break
        # print(all_deps_visited)
        if all_deps_visited:
            new_partition = visited.copy() - last_visited
            if(new_partition.any()):
                # avoid all zero, occuring at the end
                partition.append(new_partition)
            last_visited = visited.copy()
            # print("SUBG",partition)
        # todo: remove the augementation part
    return depth_level, nodes_at_depth_levels, partition

# print(bfs(graph))
_,_,partition=bfs_partition(graph)
# print("PARTITION",partition)
num_nodes_each_subgraph=np.array([i.sum() for i in partition])
print("num_nodes_each_subgraph", num_nodes_each_subgraph)
# temporarily make sure this 
assert (num_PE>=max(num_nodes_each_subgraph))
# print scan of num_nodes_each_subgraph

def get_partition_id(num_nodes_each_subgraph,num_PE):
    start_count = 0
    end_count = 0
    partition_id=np.zeros(num_nodes_each_subgraph.shape,dtype=np.int32)
    cum_id = 0
    while end_count < len(num_nodes_each_subgraph):
        end_count += 1
        if num_nodes_each_subgraph[start_count:end_count+1].sum() > num_PE\
            or end_count >= len(num_nodes_each_subgraph):
            partition_id[start_count:end_count] = cum_id
            start_count = end_count
            cum_id += 1
    return partition_id
partition_id=get_partition_id(num_nodes_each_subgraph,num_PE)
    # print("SUBG COOR", subgraph_coor_range)
print("ID",partition_id)

subgraphs=[]
subgraph_comp_lat_per_node = []
for i in range(max(partition_id)+1):
    subgraph_coor_range=np.arange(0)
    for j in np.where(partition_id==i)[0]:
        subgraph_coor_range = np.concatenate((subgraph_coor_range, np.where(partition[j])[0]))
    subgraph = graph[subgraph_coor_range,:][:,subgraph_coor_range]
    print(subgraph_coor_range)
    subgraphs.append(subgraph)
    subgraph_comp_lat_per_node.append(np.array([comp_lat_per_node[i] for i in subgraph_coor_range]))
# print(subgraphs)
print(subgraph_comp_lat_per_node)

# comp_lat_per_node = subgraph_comp_lat_per_node
# subgraph_comp_lat_per_node = []
# for i in range(max(partition_id)):
#     subgraph_comp_lat_per_node.append(comp_lat_per_node[i])
# end_node_id=np.argwhere(np.array([not (graph[i].any()) for i in range(graph.shape[0])]))
# print(end_node_id)
# print(comm_matrix(graph,distances,mapping_matrix))
# Create a new model
obj_val_list=[]
mapping_space_list = []
mapping_time_list = []
for i in range(len(subgraphs)):
    graph = subgraphs[i]
    print("GGG", graph)
    comp_lat_per_node = subgraph_comp_lat_per_node[i]
    print("LAT", comp_lat_per_node)
    m = gp.Model()
    num_var = graph.shape[0]
    print(num_var)
    mapping_space = m.addMVar((num_var,num_PE),vtype=gp.GRB.BINARY)
    Q = np.eye(num_PE)
    for i in range(num_var):
        print(i)
        m.addMConstr(np.full((1, num_PE), 1), mapping_space[i].T, '=', np.full(1, 1))
        for j in range(num_var):
            if i!=j:
                # different nodes cannot map to the same PE temporarily
                m.addMQConstr(Q, None, '=', 0, mapping_space[i], mapping_space[j])

    # comp_lat_per_node=np.array([i+1 for i in range(num_var)])
    # comp_lat_per_node = 
    # comp_lat_per_node=m.addMVar((num_var),vtype=gp.GRB.INTEGER)
    # for i in range(num_var):
    #     m.addConstr(comp_lat_per_node[i]==i+1)

    print("HERE2")
    mapping_time=m.addMVar((num_var),vtype=gp.GRB.INTEGER)
    # temporarily treat the first one as beginning
    for i in range(num_var):
        print(i)
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

    print("HERE1!")
    # m.Params.NonConvex = 2
    comp_lat=m.addVar(vtype=gp.GRB.INTEGER)
    m.addConstr(comp_lat==gp.max_([comp_end[i] for i in range(num_var)],constant=0))
    # m.addConstr(comp_lat==comp_end[end_node_id])
    m.setObjective((0+1)*(comp_lat), gp.GRB.MINIMIZE)
    m.optimize()

    print(f"Optimal objective value: {m.objVal}")
    print(f"Solution values: mapping_space=\n{mapping_space.X}")
    print(f"Solution values: mapping_time=\n{mapping_time.X}")
    # print(comm_matrix(graph,distances,mapping_space.X))
    obj_val_list.append(m.objVal)
    mapping_space_list.append(mapping_space.X)
    mapping_time_list.append(mapping_time.X)

print(obj_val_list)
print(mapping_space_list)
print(mapping_time_list)
print(sum(obj_val_list))
print("BASELINE TIME", sum(a))