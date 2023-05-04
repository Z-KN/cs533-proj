import json
import yaml
import sys
sys.path.append('../cosa')
from cosa import run_timeloop
from pathlib import Path
import argparse
import pdb

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
output_path = output_dir + '/cosa_dir/'
arch_path = Path('../cosa/configs/arch/' + args.hw + '.yaml').resolve()
mapspace_path = Path('../cosa/configs/mapspace/mapspace.yaml').resolve()

# Load the input JSON file
with open(model_info_dir + args.model + '_bignode_info.json', 'r') as f:
    data = json.load(f)

# Find the Conv operations and generate the output YAML files
for i, operation in enumerate(data):
    for j, op in enumerate(operation):
        opt_flag = 0
        if op['optype'] == 'Conv':
            opt_flag = 1
            # Extract the attribute required for the YAML output
            S, R = op['attribute']['kernel_shape']
            
            N, C, _, _ = eval(op['input'][0]['shape'])
            _, K, Q, P = eval(op['output'][0]['shape'])


            Hdilation, Wdilation = op['attribute']['dilations']
            Hstride, Wstride = op['attribute']['strides']

            # Create a dictionary for the YAML output
            output_dict = {'problem': {'C': C,
                                        'Hdilation': Hdilation,
                                        'Hstride': Hstride,
                                        'K': K,
                                        'N': N,
                                        'P': P,
                                        'Q': Q,
                                        'R': R,
                                        'S': S,
                                        'Wdilation': Wdilation,
                                        'Wstride': Wstride,
                                        'shape': 'cnn-layer'}}
        elif op['optype'] == 'MatMul':
            opt_flag = 1
            input_1 = eval(op['input'][0]['shape'])
            input_2 = eval(op['input'][1]['shape'])
            assert len(input_1) == 3
            if len(input_2) == 2:
                K = input_2[-1]
                assert input_2[-2] == input_1[-1]
                C = input_1[-1]
                P = input_1[-3]
                Q = input_1[-2]
            if len(input_2) == 3:
                K = input_2[-1]
                assert input_2[-2] == input_1[-1]
                C = input_1[-1]
                P = 1  # multiple kernel sets, same mapping for input_1[-3] times
                Q = input_1[-2]

            # Create a dictionary for the YAML output
            output_dict = {'problem': {'C': C,
                                        'Hdilation': 1,
                                        'Hstride': 1,
                                        'K': K,
                                        'N': 1,
                                        'P': P,
                                        'Q': Q,
                                        'R': 1,
                                        'S': 1,
                                        'Wdilation': 1,
                                        'Wstride': 1,
                                        'shape': 'cnn-layer'}}
        elif op['optype'] == 'Gemm':
            opt_flag = 1
            input_1 = list(eval(op['input'][0]['shape']))
            input_2 = list(eval(op['input'][1]['shape']))
            if op['attribute']['transA']:
                input_1.reverse()
            if op['attribute']['transB']:
                input_2.reverse()

            assert len(input_1) == 2
            assert len(input_2) == 2
            K = input_2[-1]
            assert input_2[-2] == input_1[-1]
            C = input_1[-1]
            P = 1
            Q = input_1[-2]

            # Create a dictionary for the YAML output
            output_dict = {'problem': {'C': C,
                                        'Hdilation': 1,
                                        'Hstride': 1,
                                        'K': K,
                                        'N': 1,
                                        'P': P,
                                        'Q': Q,
                                        'R': 1,
                                        'S': 1,
                                        'Wdilation': 1,
                                        'Wstride': 1,
                                        'shape': 'cnn-layer'}}

        if opt_flag == 1:
            # Write the YAML output file
            with open(output_path + f'outputs_inputs_{i}_{j}.yaml', 'w') as f:
                yaml.dump(output_dict, f)
            
            prob_path = Path(output_path + f'outputs_inputs_{i}_{j}.yaml').resolve()
            run_timeloop(prob_path, arch_path, mapspace_path, output_path)
