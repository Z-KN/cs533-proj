import onnx
import onnxruntime as ort
import onnx.numpy_helper
import numpy as np
import argparse
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

# Traverse the nodes in the model
for i, node in enumerate(model.graph.node):
    print(f'Node {i}: {node.op_type}')
    for input_name in node.input:
        print(f'  Input: {input_name} {ort_outs.get(input_name)}')
    for output_name in node.output:
        print(f'  Output: {output_name} {ort_outs.get(output_name)}')
# print(ort_outs)