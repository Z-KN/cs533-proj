import json
import yaml
import pdb

# Load the input JSON file
with open('transformer_node_info.json', 'r') as f:
    data = json.load(f)

#print(data)

# Find the Conv operations and generate the output YAML files
for i, operation in enumerate(data):
    operator_type = operation['optype']
    if operator_type == 'MatMul':

        # Extract the attribute required for the YAML output
        #R, S = list(map(int, op['input'][1]['shape'][1: -1].split(", ")))
        #P, K, Q = list(map(int, op['output'][0]['shape'][1: -1].split(", ")))
        #_, C, _ = list(map(int, op['input'][0]['shape'][1: -1].split(", ")))
        if len(eval(operation['input'][1]['shape'])) == 2:
            C, R, S = eval(operation['input'][0]['shape'])
            K, P, Q = eval(operation['output'][0]['shape'])
        if len(eval(operation['input'][1]['shape'])) == 3:
            C, R, S = eval(operation['input'][1]['shape'])
            K, P, Q = eval(operation['output'][0]['shape'])

        #Hdilation, Wdilation = op['attribute']['dilations']
        #Hstride, Wstride = op['attribute']['strides']

        # Create a dictionary for the YAML output
        output_dict = {'problem': {'C': C,
                                    'Hdilation': 1,
                                    'Hstride': 1,
                                    'K': K,
                                    'N': 1,
                                    'P': P,
                                    'Q': Q,
                                    'R': R,
                                    'S': S,
                                    'Wdilation': 1,
                                    'Wstride': 1,
                                    'shape': 'cnn-layer'}}
        # print(output_dict)

        # Write the YAML output file
        with open('.\outputs_inputs_{}.yaml'.format(i), 'w') as f:
            yaml.dump(output_dict, f)
