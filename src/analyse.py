import onnx
import onnxruntime as ort
import onnx.numpy_helper
import numpy as np
import argparse
import json
from collections import OrderedDict

# Create an argparse object and add arguments
parser = argparse.ArgumentParser(description='Analyzing transformer and resnet models.')
parser.add_argument('--model', type=str, required=True, help='Model to analyze (transformer or resnet)')

# Parse the arguments
args = parser.parse_args()

# Check which model was selected
if args.model == 'transformer':
    print('Analyzing transformer model...')
    # Call function to analyze transformer model
    model_path = 'transformer.onnx'
    in_img = [np.random.randn(10, 1, 128).astype(np.float32), \
        np.random.randn(10, 1, 128).astype(np.float32)]
elif args.model == 'resnet':
    print('Analyzing resnet model...')
    # Call function to analyze resnet model
    model_path = 'resnet18-v2-7.onnx'
    in_img = [np.random.randn(1,3,224,224).astype(np.float32)]
else:
    print('Error: Model not recognized. Please choose transformer or resnet.')
    exit(1)

ort_session = ort.InferenceSession(model_path)
org_outputs = [x.name for x in ort_session.get_outputs()]

model = onnx.load(model_path)
for node in model.graph.node:
    for output in node.output:
        if output not in org_outputs:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])

# execute onnx
ort_session = ort.InferenceSession(model.SerializeToString())
outputs = [x.name for x in ort_session.get_outputs()]
inputs = {}
for input_name, input_value in zip(ort_session.get_inputs(), in_img):
    inputs[input_name.name] = input_value
ort_outs = ort_session.run(outputs, inputs)
ort_outs = [single_out.shape for single_out in ort_outs]
ort_outs = OrderedDict(zip(outputs, ort_outs))

# extend directory
for input in model.graph.input:
    dim_list = [str(d.dim_value) for d in input.type.tensor_type.shape.dim]
    if len(dim_list) == 0:
        ort_outs[input.name] = '()'
    elif len(dim_list) == 1:
        ort_outs[input.name] = '(' + dim_list[0] + ',)'
    else:
        ort_outs[input.name] = '(' + ', '.join(dim_list) + ')'
for ini in model.graph.initializer:
    dim_list = [str(dim) for dim in ini.dims]
    if len(dim_list) == 0:
        ort_outs[ini.name] = '()'
    elif len(dim_list) == 1:
        ort_outs[ini.name] = '(' + dim_list[0] + ',)'
    else:
        ort_outs[ini.name] = '(' + ', '.join(dim_list) + ')'

# Traverse the nodes in the model
node_list = []
optype_list = []
for i, node in enumerate(model.graph.node):
    print(f'Node {i}: {node.op_type}')
    if not node.op_type in optype_list:
        optype_list.append(node.op_type)
    input_info_list = []
    output_info_list = []
    for input_name in node.input:
        print(f'  Input: {input_name} {ort_outs.get(input_name)}')
        input_info_list.append({'name': f'{input_name}', 'shape': f'{ort_outs.get(input_name)}'})
    for output_name in node.output:
        print(f'  Output: {output_name} {ort_outs.get(output_name)}')
        output_info_list.append({'name': f'{output_name}', 'shape': f'{ort_outs.get(output_name)}'})
    node_list.append({'optype': f'{node.op_type}', 'input':input_info_list, 'output': output_info_list})
print(optype_list)

with open(args.model + '_node_info.json', 'w') as f:
    # Write the JSON string to the file
    json.dump(node_list, f, indent=4)
