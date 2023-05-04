import numpy as np
import gurobipy as gp
from collections import deque
import json
# add argparse to get the command argument
import argparse
from math import ceil
import pdb


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
    return graph * (mapping_space@distances@mapping_space.T) / 64
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

def bfs_partition(adj_matrix):
    # Initialize the visited array and the queue
    num_nodes = len(adj_matrix)
    visited = np.zeros(num_nodes,dtype=np.int32)
    last_visited = np.zeros(num_nodes,dtype=np.int32)

    queue = deque()
    
    partition=[]
    starting_nodes=[i for i in range(num_nodes) if not adj_matrix[:,i].any()]
    # print(adj_matrix)
    print(starting_nodes)
    for i in starting_nodes:
        queue.append(i)
    visited[starting_nodes] = 1
    depth_level = [0] * num_nodes
    nodes_at_depth_levels = [starting_nodes]
    
    # Traverse the graph using BFS
    while len(queue) > 0:
        cur_node = queue.popleft()
        cur_depth = depth_level[cur_node]
        # print("CUR",cur_node)
        print(queue)
        all_deps_visited = True  # Flag for checking if all dependencies are visited
        
        neighbors = np.where(adj_matrix[cur_node,:])[0]
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
            for dep in np.where(adj_matrix[:,neighbor])[0]:
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

def reduce_shape(shape):
    num_words = 1
    for dim in shape:
        num_words *= dim
    return num_words

parser = argparse.ArgumentParser()
parser.add_argument('--model', help="model", required=False,default='resnet')
parser.add_argument('--hw', type=str, required=False, help='Hardware the model is run on',
                    default='imc')
parser.add_argument('--row', help="row number", type=int,required=False,default=4)
parser.add_argument('--col', help="column number", type=int, required=False,default=4)
# get values from arguments
args = parser.parse_args()


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

print(f"Now scheduling model {args.model}")
output_dir = '../' + args.model
model_info_dir = output_dir + '/model_info/'
model_result_dir = output_dir + '/result/'
# load adjacent matrix
graph = np.loadtxt(model_info_dir + args.model + '_connection_matrix.txt',delimiter=',').astype(int)
# load opt timespace
with open(model_info_dir + args.model + '_' + args.hw + '_bignode_timespace.json') as f:
    comp_lat_per_node = np.array(json.load(f))
    # print(comp_lat_per_node)
# print(graph)
orig_comp_lat_per_node=comp_lat_per_node.copy()
# load baseline time
with open(model_info_dir + args.model + '_' + args.hw + '_baseline_timespace.json') as f:
    baseline_time = np.array(json.load(f))[0]
# Load the bignode file
with open(model_info_dir + args.model + '_bignode_info.json', 'r') as f:
    bignode_config = json.load(f)

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

row_PE = args.row
col_PE = args.col
print(f"Now scheduling {row_PE} x {col_PE} PEs")
num_PE = row_PE * col_PE
distances = gen_dis(row_PE, col_PE)
# print(distances)

# print(bfs(graph))
_,_,partition=bfs_partition(graph)
print("PARTITION",partition)
num_nodes_each_subgraph=np.array([i.sum() for i in partition])
# print("num_nodes_each_subgraph", num_nodes_each_subgraph)
# temporarily make sure this 
assert (num_PE>=max(num_nodes_each_subgraph))
# print scan of num_nodes_each_subgraph

partition_id=get_partition_id(num_nodes_each_subgraph,num_PE)
    # print("SUBG COOR", subgraph_coor_range)
# print("ID",partition_id)

subgraphs=[]
subgraph_comp_lat_per_node = []
parition_sets = []
for i in range(max(partition_id)+1):
    subgraph_coor_range=np.arange(0)
    for j in np.where(partition_id==i)[0]:
        subgraph_coor_range = np.concatenate((subgraph_coor_range, np.where(partition[j])[0]))
    subgraph = graph[subgraph_coor_range,:][:,subgraph_coor_range]
    parition_sets.append(subgraph_coor_range)
    print(subgraph_coor_range)
    subgraphs.append(subgraph)
    subgraph_comp_lat_per_node.append(np.array([comp_lat_per_node[i] for i in subgraph_coor_range]))
# print(subgraphs)
# print(subgraph_comp_lat_per_node)

# Compute extra time required between load store paritions
extra_ls_time = 0
words_per_cycle = 16  # off-ship bandwidth
output_history = []
for parition_idx in range(len(parition_sets)):
    cur_par = parition_sets[parition_idx]
    # test each bignode input
    for bignode_idx in cur_par:
        cur_bignode = bignode_config[bignode_idx]
        inputs = cur_bignode[0]['input']
        # pick one dependent input
        for input in inputs:
            if input['type'] == 'dependent':
                input_name = input['name']
                # search for other parition
                for match_parition_idx in range(len(parition_sets)):
                    # avoid replicate parition test
                    if match_parition_idx == parition_idx:
                        continue
                    match_par = parition_sets[match_parition_idx]
                    for match_bignode_idx in match_par:
                        match_bignode = bignode_config[match_bignode_idx]
                        output_name = match_bignode[-1]['output'][0]['name']
                        if output_name == input_name:
                            assert input['shape'] == match_bignode[-1]['output'][0]['shape']
                            input_shape = eval(input['shape'])
                            num_words = reduce_shape(input_shape)
                            extra_ls_time += ceil(num_words / words_per_cycle)
                            if output_name not in output_history:
                                output_history.append(output_name)
                                extra_ls_time += ceil(num_words / words_per_cycle)  # store time


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
    comp_lat_per_node = subgraph_comp_lat_per_node[i]
    m = gp.Model()
    num_var = graph.shape[0]
    # print(num_var)
    mapping_space = m.addMVar((num_var,num_PE),vtype=gp.GRB.BINARY)
    Q = np.eye(num_PE)
    for i in range(num_var):
        # print(i)
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
# print(np.array([sum(obj_val_list)],dtype=np.int32))
# print(np.array(sum(orig_comp_lat_per_node),dtype=np.int32))
np.savetxt(model_result_dir + f"{args.model}_compare_{args.row}_{args.col}.txt", np.array([baseline_time,sum(obj_val_list)+extra_ls_time,sum(orig_comp_lat_per_node)+extra_ls_time],dtype=np.int32),fmt='%d')
# np.savetxt("baseline.txt", np.array([sum(orig_comp_lat_per_node)],dtype=np.int32),fmt='%d')