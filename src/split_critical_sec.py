import argparse
import json
import copy
import numpy as np
import pdb


# Create an argparse object and add arguments
parser = argparse.ArgumentParser(description='Analyzing transformer and resnet models.')
parser.add_argument('--model', type=str, required=True, help='Model to analyze (transformer or resnet)')
# Parse the arguments
args = parser.parse_args()

# Open the $(MODEL_NAME)_node_info.json file
with open(args.model + '_node_info.json') as f_node:
    node_dic = json.load(f_node)
origin_num_nodes = len(node_dic)
# Augment Node (exclude multi-inputs & multi-outputs node)
print("[+] Augment Graph......")
append_nodes = []
# Traverse all nodes and split MIMO nodes
for node_idx in range(len(node_dic)):
    if node_dic[node_idx]['link_class'] == 'MIMO':
        origin_output_info = copy.deepcopy(node_dic[node_idx]['output'])
        # reconstruct predecessor node
        node_dic[node_idx]['link_class'] = 'MISO'
        node_dic[node_idx]['output'][0]['name'] = origin_output_info[0]['name'] + '_Identity'
        node_dic[node_idx]['output'][0]['num'] = 1
        # construct successor node
        suc_node = {'optype': 'Identity'}
        # input
        suc_node_input = {}
        suc_node_input['name'] = node_dic[node_idx]['output'][0]['name']
        suc_node_input['shape'] = node_dic[node_idx]['output'][0]['shape']
        suc_node_input['type'] = 'dependent'
        suc_node['input'] = [suc_node_input]
        # output
        suc_node['output'] = origin_output_info
        # link class
        suc_node['link_class'] = 'SIMO'
        append_nodes.append(suc_node)
node_dic += append_nodes

with open(args.model + '_node_info_aug.json', 'w') as f_aug:
    # Write the JSON string to the file
    json.dump(node_dic, f_aug, indent=2)
augmented_num_nodes = len(node_dic)
print(f'[+] Finish augmentation process! Num of nodes: {origin_num_nodes} -> {augmented_num_nodes}')
print('[+] Store augmented nodes info' + args.model + '_node_info_aug.json')

# Finding Critical Section
# Initialize the source node list
source_node_list = []
source_node_history = []
for node_idx in range(len(node_dic)):
    if 'source' in node_dic[node_idx]['link_class']:
        source_node_list.append(node_idx)
        source_node_history.append(node_idx)
# Search for critical section starting from a source point
cs_list = []
while len(source_node_list) != 0:
    cur_node_dic_list = []
    # get the source point
    cur_node_idx = source_node_list.pop()
    cur_node = node_dic[cur_node_idx]
    cur_output_name = cur_node['output'][0]['name']
    cur_node_dic_list.append(cur_node)
    # check next node until get MISO or SIMO
    stop_flag = 0
    # if cur node is MO -> stop
    if 'MO' in cur_node['link_class']:
        stop_flag = 1
    while stop_flag == 0:
        # search for the next node
        detect_flag = 0
        for node_idx in range(len(node_dic)):
            for input_port in node_dic[node_idx]['input']:
                if input_port['name'] == cur_output_name:
                    # pdb.set_trace()
                    cur_node_idx = node_idx
                    cur_node = node_dic[cur_node_idx]
                    cur_output_name = cur_node['output'][0]['name']
                    detect_flag = 1
                    break
            if detect_flag == 1:
                break
        # analysis node
        # MO -> include it and push its successor as source nodes and stop
        if 'MO' in cur_node['link_class']:
            stop_flag = 1
            cur_node_dic_list.append(cur_node)
            for node_idx in range(len(node_dic)):
                for input_port in node_dic[node_idx]['input']:
                    if input_port['name'] == cur_output_name:
                        source_node_list.append(node_idx)
                        source_node_history.append(node_idx)
        # MI -> do not include it and put it as source nodes and stop
        elif 'MI' in cur_node['link_class']:
            stop_flag = 1
            if not (node_idx in source_node_history):
                source_node_list.append(node_idx)
                source_node_history.append(node_idx)
        # sink -> include it and stop
        elif 'sink' in cur_node['link_class']:
            stop_flag = 1
            cur_node_dic_list.append(cur_node)
        # SISO -> include it and continue
        else:
            cur_node_dic_list.append(cur_node)

    cs_list.append(cur_node_dic_list)

print('[+] Finish merging nodes!')
total_small_nodes = 0
for big_node in cs_list:
    total_small_nodes += len(big_node)
# pdb.set_trace()
# assert total_small_nodes == augmented_num_nodes
print(f'[+] Get Big Nodes -> {len(cs_list)}')
with open(args.model + '_bignode_info.json', 'w') as f_bignode:
    # Write the JSON string to the file
    json.dump(cs_list, f_bignode, indent=2)
print('[+] Store big nodes into' + args.model + '_bignode_info.json')

# generate connection matrix
c_matrix = np.zeros((len(cs_list), len(cs_list)))
for source_big_node_idx in range(len(cs_list)):
    output_name = cs_list[source_big_node_idx][-1]['output'][0]['name']
    for sink_big_node_idx in range(len(cs_list)):
        for input_port in cs_list[sink_big_node_idx][0]['input']:
            if input_port['name'] == output_name:
                c_matrix[source_big_node_idx, sink_big_node_idx] = 1
print('[+] Finish generating connection matrix!')
np.savetxt(args.model + '_connection_matrix.txt', c_matrix, fmt='%d', delimiter=',')
print('[+] Store connection matrix into' + args.model + '_connection_matrix.txt')
