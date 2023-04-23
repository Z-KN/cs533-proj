import json
import yaml

# Load the input JSON file
with open('resnet_bignode_info.json', 'r') as f:
    data = json.load(f)

# print(data)

# Find the Conv operations and generate the output YAML files
for i, operation in enumerate(data):
    for op in operation:
        if op['optype'] == 'Conv':
            # Extract the attribute required for the YAML output
            R, S = op['attribute']['kernel_shape']
            
            N, C, _, _ = eval op['input'][0]['shape'][1: -1].split(", ")))
            _, K, P, Q = list(map(int, op['output'][0]['shape'][1: -1].split(", ")))


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
            with open('outputs_inputs_{}.yaml'.format(i), 'w') as f:
                yaml.dump(output_dict, f)
