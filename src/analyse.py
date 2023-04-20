import onnx
import onnxruntime as ort
import onnx.numpy_helper
import numpy as np
import argparse
import json
from collections import OrderedDict
import os
import shutil
import pdb

# type dic
data_type_dic = {}
data_type_dic[onnx.TensorProto.INT64] = 'int64'
data_type_dic[onnx.TensorProto.INT32] = 'int32'
data_type_dic[onnx.TensorProto.INT16] = 'int16'
data_type_dic[onnx.TensorProto.INT8] = 'int8'
data_type_dic[onnx.TensorProto.UINT64] = 'uint64'
data_type_dic[onnx.TensorProto.UINT32] = 'uint32'
data_type_dic[onnx.TensorProto.UINT16] = 'uint16'
data_type_dic[onnx.TensorProto.UINT8] = 'uint8'
data_type_dic[onnx.TensorProto.INT64] = 'int64'
data_type_dic[onnx.TensorProto.INT32] = 'int32'
data_type_dic[onnx.TensorProto.INT16] = 'int16'
data_type_dic[onnx.TensorProto.FLOAT16] = 'float16'
data_type_dic[onnx.TensorProto.FLOAT] = 'float32'

# Create an argparse object and add arguments
parser = argparse.ArgumentParser(description='Analyzing transformer and resnet models.')
parser.add_argument('--model', type=str, required=False, help='Model to analyze (transformer or resnet)',
                    default='transformer')

# Parse the arguments
args = parser.parse_args()

# Check which model was selected
if args.model == 'transformer':
    print('Analyzing transformer model...')
    # Call function to analyze quantized transformer model
    model_path = 'transformer_q.onnx'
    in_img = [np.random.randn(10, 1, 128).astype(np.float32), \
        np.random.randn(10, 1, 128).astype(np.float32)]
elif args.model == 'resnet':
    print('Analyzing resnet model...')
    # Call function to analyze quantized resnet model
    model_path = 'resnet18-v2-7_q.onnx'
    in_img = [np.random.randn(1,3,224,224).astype(np.float32)]
else:
    print('Error: Model not recognized. Please choose transformer or resnet.')
    exit(1)
# create folder to store model data
data_dir = './' + args.model
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
else:
    shutil.rmtree(data_dir)
    os.makedirs(data_dir)

ort_session = ort.InferenceSession(model_path)
org_outputs = [x.name for x in ort_session.get_outputs()]

model = onnx.load(model_path)

input_names_list = []
output_names_list = []
for node in model.graph.node:
    if node.op_type != 'Constant':
        input_names_list += node.input
        output_names_list += node.output
    for output in node.output:
        # include output name
        if output not in org_outputs:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])

# execute onnx to get output shape
ort_session = ort.InferenceSession(model.SerializeToString())
outputs = [x.name for x in ort_session.get_outputs()]
inputs = {}
for input_name, input_value in zip(ort_session.get_inputs(), in_img):
    inputs[input_name.name] = input_value
ort_outs = ort_session.run(outputs, inputs)
ort_outs = [single_out.shape for single_out in ort_outs]
ort_outs = OrderedDict(zip(outputs, ort_outs))
# add entry point shape
for input_index in range(len(in_img)):
    ort_outs[ort_session.get_inputs()[input_index].name] = in_img[input_index].shape

# extend directory
for input in model.graph.input:
    if input.name not in ort_outs:
        dim_list = [str(d.dim_value) for d in input.type.tensor_type.shape.dim]
        if len(dim_list) == 0:
            ort_outs[input.name] = '()'
        elif len(dim_list) == 1:
            ort_outs[input.name] = '(' + dim_list[0] + ',)'
        else:
            ort_outs[input.name] = '(' + ', '.join(dim_list) + ')'
# Store independent value
ini_value = {}
ini_type = {}
for ini in model.graph.initializer:
    ini_type[ini.name] = data_type_dic[ini.data_type]
    if hasattr(ini, 'int64_data') and (len(ini.int64_data) > 0):
        ini_value[ini.name] = np.array(ini.int64_data, dtype=eval('np.'+ini_type[ini.name])).tobytes()
    elif hasattr(ini, 'int32_data') and (len(ini.int32_data) > 0):
        ini_value[ini.name] = np.array(ini.int32_data, dtype=eval('np.'+ini_type[ini.name])).tobytes()
    elif hasattr(ini, 'float_data') and (len(ini.float_data) > 0):
        ini_value[ini.name] = np.array(ini.float_data, dtype=eval('np.'+ini_type[ini.name])).tobytes()
    elif hasattr(ini, 'raw_data') and (len(ini.raw_data) > 0):
        ini_value[ini.name] = ini.raw_data
    else:
        print("New data type in initializer!")
        exit(1)
    if ini.name not in ort_outs:
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
    # Constant should be excluded and embedded in its output node
    if node.op_type == 'Constant':
        if len(node.attribute[0].t.raw_data) > 0:
            ini_type[node.output[0]] = data_type_dic[node.attribute[0].t.data_type]
            ini_value[node.output[0]] = node.attribute[0].t.raw_data
        else:
            print("Cannot get data of Constant Node!")
            exit(1)
        continue
    print(f'Node {i}: {node.op_type}')
    if not node.op_type in optype_list:
        optype_list.append(node.op_type)
    input_info_list = []
    output_info_list = []
    num_input = 0
    num_output = 0
    for input_name in node.input:
        dup_count = output_names_list.count(input_name)
        if (dup_count == 0) and (node.op_type not in ('Slice', 'Reshape')):
            value = None
            try:
                data_type = ini_type[input_name]
            except:
                data_type = None
                print(f'Cannot find the data type for {input_name}')
            # Store initialized data to .data file
            if ('scale' not in input_name) and ('zero_point' not in input_name):
                try:
                    with open(data_dir+'/'+input_name+'.data', 'wb') as f_indep:
                        f_indep.write(ini_value[input_name])
                except:
                    print(f'Cannot find the value for {input_name}')
            else:
                # Only scale/zero point in QuantizeLinear,DequantizeLinear should be included in json
                if node.op_type in ('QuantizeLinear', 'DequantizeLinear'):
                    value = np.frombuffer(ini_value[input_name], dtype=eval('np.'+ini_type[input_name])).tolist()
                else:
                    continue  # ignore scale and zero point for other operator
            print(f'  Input: {input_name} {ort_outs.get(input_name)} independent')
            input_info_list.append({'name': f'{input_name}', 'shape': f'{ort_outs.get(input_name)}', \
                                    'value': value, 'type': f'{data_type}', 'dependency': 'independent'})
        elif dup_count == 1:
            num_input += 1
            print(f'  Input: {input_name} {ort_outs.get(input_name)} dependent')
            input_info_list.append({'name': f'{input_name}', 'shape': f'{ort_outs.get(input_name)}', \
                                    'dependency': 'dependent'})
    for output_name in node.output:
        dup_count = input_names_list.count(output_name)
        num_output += dup_count
        print(f'  Output: {output_name} {ort_outs.get(output_name)} {dup_count}')
        output_info_list.append({'name': f'{output_name}', 'shape': f'{ort_outs.get(output_name)}', 'num': dup_count})
    link_class = ''
    if num_input == 1:
        link_class += 'SI'
    elif num_input == 0:
        link_class += 'source'
    else:
        link_class += 'MI'
    if num_output == 1:
        link_class += 'SO'
    elif num_output == 0:
        link_class += 'sink'
    else:
        link_class += 'MO'

    # add attribute for some node type
    att_dic = {}
    for att_item in node.attribute:
        att_name = att_item.name
        if att_item.ints:
            att_dic[att_name] = list(att_item.ints)
        elif att_item.floats:
            att_dic[att_name] = list(att_item.floats)
        elif att_item.HasField('i'):
            att_dic[att_name] = att_item.i
        elif att_item.HasField('f'):
            att_dic[att_name] = att_item.f
        else:
            print('New data type in attribute!')
    # Operator Slice/Reshape has their attribute in initializer
    if node.op_type == 'Slice':
        slice_att = ['starts', 'ends', 'axes']
        value_list = [np.frombuffer(ini_value[node.input[slice_idx]], \
                               dtype=eval('np.'+ini_type[node.input[slice_idx]])).tolist() \
                                for slice_idx in range(1,4)]
        for slice_idx in range(3):
            att_dic[slice_att[slice_idx]] = value_list[slice_idx]
    elif node.op_type == 'Reshape':
        value = np.frombuffer(ini_value[node.input[1]], \
                                dtype=eval('np.'+ini_type[node.input[1]])).tolist()
        att_dic['shape'] = value
    node_list.append({'optype': f'{node.op_type}', 'input': input_info_list, 'attribute': att_dic, 'output': output_info_list, 'link_class': link_class})
print(optype_list)

with open(args.model + '_node_info.json', 'w') as f:
    # Write the JSON string to the file
    json.dump(node_list, f, indent=2)
