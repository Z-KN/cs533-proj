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

# Parse the arguments
args = parser.parse_args()

output_dir = '../' + args.model
model_info_dir = output_dir + '/model_info/'
output_path = output_dir + '/cosa_dir/'
arch_path = Path('../cosa/configs/arch/simba.yaml').resolve()
mapspace_path = Path('../cosa/configs/mapspace/mapspace.yaml').resolve()

# Load the input JSON file
with open(model_info_dir + args.model + '_bignode_info.json', 'r') as f:
    data = json.load(f)

# print(data)

# Find the Conv operations and generate the output YAML files
for i, operation in enumerate(data):
    for j, op in enumerate(operation):
        if op['optype'] == 'Conv':
            # Extract the attribute required for the YAML output
            R, S = op['attribute']['kernel_shape']
            
            N, C, _, _ = eval(op['input'][0]['shape'])
            _, K, P, Q = eval(op['output'][0]['shape'])


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
            # print(output_dict)

            # Write the YAML output file
            with open(output_path + f'outputs_inputs_{i}_{j}.yaml', 'w') as f:
                yaml.dump(output_dict, f)
            
            prob_path = Path(output_path + f'outputs_inputs_{i}_{j}.yaml').resolve()
            run_timeloop(prob_path, arch_path, mapspace_path, output_path)
