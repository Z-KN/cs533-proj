import json
import yaml
import sys
import argparse
import re
from math import ceil
import pdb


def reduce_shape(shape):
    num_words = 1
    for dim in shape:
        num_words *= dim
    return num_words

# Create an argparse object and add arguments
parser = argparse.ArgumentParser(description='Analyzing transformer and resnet models.')
parser.add_argument('--model', type=str, required=False, help='Model to analyze (transformer or resnet)',
                    default='transformer')
parser.add_argument('--hw', type=str, required=False, help='Hardware the model is run on',
                    default='imc')

# Parse the arguments
args = parser.parse_args()

output_dir = '../' + args.model
model_info_dir = output_dir + '/model_info/'
cosa_path = output_dir + '/cosa_dir/' + args.hw + '/'

# Load the input JSON file
with open(model_info_dir + args.model + '_bignode_info.json', 'r') as f:
    bignode_config = json.load(f)

bignode_time = []
baseline_time = 0
words_per_cycle = 16  # off-ship bandwidth
# Traverse each node in big node
massive_op = ['Conv', 'MatMul', 'Gemm']
ignore_op = ['BatchNormalization', 'Identity']
for bignode_idx in range(len(bignode_config)):
    bignode = bignode_config[bignode_idx]
    time_sum = 0
    # Assume that load and store data does not require time
    for node_idx in range(len(bignode)):
        node = bignode[node_idx]
        # if the node is sink type, then it needs to store
        if 'sink' in node['link_class']:
            output_shape = eval(node['output'][0]['shape'])
            num_words = reduce_shape(output_shape)
            time_sum += ceil(num_words / words_per_cycle)
        # if the node require to load weight
        # for ignore_op only first input needs to check if it is user input
        if node['optype'] in ignore_op:
            if node['input'][0]['type'] == 'independent':
                input_shape = eval(node['input'][0]['shape'])
                num_words = reduce_shape(input_shape)
                time_sum += ceil(num_words / words_per_cycle)
            continue
        else:
            for input in node['input']:
                # baseline require all input load and output store
                input_shape = eval(input['shape'])
                num_words = reduce_shape(input_shape)
                baseline_time += ceil(num_words / words_per_cycle)
                # opt version only load weight
                if input['type'] == 'independent':
                    input_shape = eval(input['shape'])
                    num_words = reduce_shape(input_shape)
                    time_sum += ceil(num_words / words_per_cycle)
            # baseline
            output_shape = eval(node['output'][0]['shape'])
            num_words = reduce_shape(output_shape)
            baseline_time += ceil(num_words / words_per_cycle)
        # add computation time
        if node['optype'] in massive_op:
            cosa_time = 1
            # open cosa map file
            with open(cosa_path + f'{bignode_idx}_{node_idx}/map_16.yaml', 'r') as f:
                cosa_map = yaml.safe_load(f)
            # traverse temporal mapping
            for mapping in cosa_map['mapping']:
                if ('factors' in mapping) and mapping['type'] == 'temporal':
                    loop_str = mapping['factors']
                    factor_list = [int(elem) for elem in re.sub('[A-Z]=', '', loop_str).split(' ')]
                    for factor in factor_list:
                        cosa_time *= factor
            # Matmul are supposed to repeat the mapping for 3-dimension kernel
            if (node['optype'] == 'MatMul') and (len(eval(node['input'][1]['shape'])) == 3):
                repeat_times = eval(node['input'][0]['shape'])[0]
                cosa_time *= repeat_times
            # Gemm should consider Add if it has input 3
            elif (node['optype'] == 'Gemm') and (len(node['input']) == 3):
                cosa_time += 1
            time_sum += cosa_time
            baseline_time += cosa_time
        else:
            # Other operators require one cycle to complete
            time_sum += 1
            baseline_time += 1
    # Append time for one big node
    bignode_time.append(time_sum)

with open(model_info_dir + args.model + '_' + args.hw + '_bignode_timespace.json', 'w') as f:
    json.dump(bignode_time, f, indent=2)
with open(model_info_dir + args.model + '_' + args.hw + '_baseline_timespace.json', 'w') as f:
    json.dump([baseline_time], f, indent=2)
