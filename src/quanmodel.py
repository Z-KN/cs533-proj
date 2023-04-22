import onnx
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process
import onnxruntime as ort
import numpy as np
import argparse
import copy
import pdb

class DataReader(CalibrationDataReader):
    def __init__(self, input_name, imgs):
        self.input_name = input_name
        self.data = imgs
        self.pos = -1

    def get_next(self):
        if self.pos >= len(self.data) - 1:
            return None
        self.pos += 1
        data_dic = {}
        for input_idx in range(len(self.input_name)):
            data_dic[self.input_name[input_idx]] = self.data[input_idx][self.pos]
        return data_dic

    def rewind(self):
        self.pos = -1


# Create an argparse object and add arguments
parser = argparse.ArgumentParser(description='Analyzing transformer and resnet models.')
parser.add_argument('--model', type=str, required=False, help='Model to analyze (transformer or resnet)',
                    default='transformer')

# Parse the arguments
args = parser.parse_args()

# Check which model was selected
if args.model == 'transformer':
    print('Converting transformer model...')
    model_path = 'transformer.onnx'
elif args.model == 'resnet':
    print('Converting resnet model...')
    model_path = 'resnet18-v2-7.onnx'
else:
    print('Error: Model not recognized. Please choose transformer or resnet.')
    exit(1)

# Load the original ONNX model
model = onnx.load(model_path)

# Add BN node index and its dependent input/output (default 0 index)
bn_node_indices = []
bn_name_transfer = {}
for i, node in enumerate(model.graph.node):
    if node.op_type == 'BatchNormalization':
        bn_node_indices.append(i)
        bn_name_transfer[node.output[0]] = node.input[0]
# Remove all the batch normalization nodes from the graph
for i in reversed(bn_node_indices):
    del model.graph.node[i]

# Modify the input name to reconnect the remaining components
for node in model.graph.node:
    for input_name in node.input:
        if input_name in bn_name_transfer:
            for idx in range(len(node.input)):
                if node.input[idx] == input_name:
                    target_idx = idx
            node.input[target_idx] = bn_name_transfer[input_name]

# int type dic
# int_type_dic = {}
# int_type_dic[onnx.TensorProto.INT8] = np.int8
# int_type_dic[onnx.TensorProto.INT16] = np.int16
# int_type_dic[onnx.TensorProto.INT32] = np.int32
# int_type_dic[onnx.TensorProto.INT64] = np.int64

# # 'Quantize' the value in initializer
# ini_value = {}
# for ini in model.graph.initializer:
#     if hasattr(ini, 'int64_data') and (len(ini.int64_data) > 0):
#         new_tensor = np.array(ini.int64_data).tobytes()
#         ini.ClearField('int64_data')
#         ini.raw_data = new_tensor
#     elif hasattr(ini, 'float_data') and (len(ini.float_data) > 0):
#         ini.ClearField('float_data')
#         new_tensor = np.random.randint(low=0, high=256, size=ini.dims, dtype=np.int64).tobytes()
#         ini.raw_data = new_tensor
#         ini.data_type = onnx.TensorProto.INT64
#     elif hasattr(ini, 'raw_data') and (len(ini.raw_data) > 0):
#         if ini.data_type == onnx.TensorProto.INT64:
#             continue
#         else:
#             new_tensor = np.random.randint(low=0, high=256, size=ini.dims, dtype=np.int64).tobytes()
#             ini.raw_data = new_tensor
#             ini.data_type = onnx.TensorProto.INT64
#     else:
#         print("Initializer includes new data type!")
#         exit(1)

# Extend the model info
# origin_output = copy.deepcopy(model.graph.output)
# for node in model.graph.node:
#     for output in node.output:
#         if output not in origin_output:
#             model.graph.output.extend([onnx.ValueInfoProto(name=output)])
# pdb.set_trace()

# Change Constant value type
# for node in model.graph.node:
#     if node.op_type == 'Constant':
#         tensor_obj = node.attribute[0].t
#         if hasattr(tensor_obj, 'raw_data') and (len(tensor_obj.raw_data) > 0):
#             if tensor_obj.data_type in int_type_dic:
#                 if tensor_obj.data_type == onnx.TensorProto.INT64:
#                     continue
#                 else:
#                     tensor_obj.data_type = onnx.TensorProto.INT64
#                     origin_arr = np.frombuffer(tensor_obj.raw_data, dtype=int_type_dic[tensor_obj.data_type])
#                     tensor_obj.raw_data = origin_arr.astype(dtype=np.int64).tobytes()
#             else:
#                 tensor_obj.data_type = onnx.TensorProto.INT64
#                 new_tensor = np.random.randint(low=0, high=256, size=tensor_obj.dims, dtype=np.int64).tobytes()
#                 tensor_obj.raw_data = new_tensor
#         else:
#             print("Constant node does not use raw data!")
#             exit(1)
#     else:
#         if len(node.attribute) > 0:
#             for att_item in node.attribute:
#                 if att_item.ints:
#                     for index in range(len(att_item.ints)):
#                         att_item.ints[index] = np.int64(att_item.ints[index]).astype(np.int64)
#                     att_item.type = onnx.AttributeProto.INTS
#                 elif att_item.floats:
#                     for index in range(len(att_item.floats)):
#                         att_item.floats[index] = np.int64(att_item.floats[index]).astype(np.int64)
#                     att_item.type = onnx.AttributeProto.INTS
#                 elif att_item.HasField('i'):
#                     att_item.i = np.int64(att_item.i).astype(np.int64)
#                     att_item.type = onnx.AttributeProto.INT
#                 elif att_item.HasField('f'):
#                     att_item.f = np.int64(att_item.f).astype(np.int64)
#                     att_item.type = onnx.AttributeProto.INT
#                 else:
#                     print("Attribute exists other data type!")
#                     exit(1)
#             pass

# # Change input/output type
# for input_obj in model.graph.input:
#     input_obj.type.tensor_type.elem_type = onnx.TensorProto.INT64
# for output_obj in model.graph.output:
#     output_obj.type.tensor_type.elem_type = onnx.TensorProto.INT64

# # Recover model graph output
# trace_idx = -1
# while len(model.graph.output) != len(origin_output):
#     output_elem = model.graph.output[trace_idx]
#     if output_elem.name in origin_output:
#         trace_idx -= 1
#         continue
#     else:
#         model.graph.output.remove(output_elem)

# Save the modified model to disk
nobn_model_path = '.'.join(model_path.split('.')[:-1]) + '_nobn.onnx'
onnx.save(model, nobn_model_path)

preprocessed_model_path = '.'.join(model_path.split('.')[:-1]) + '_pre.onnx'

quant_pre_process(nobn_model_path, preprocessed_model_path)

sess_full = ort_session = ort.InferenceSession(preprocessed_model_path)

# Get input name
input_name_list = []
input_shape_list = []
for input in sess_full.get_inputs():
    input_name_list.append(input.name)
    input_shape = list(input.shape)
    for shape_idx in range(len(input_shape)):
        if input_shape[shape_idx] in [None, "batch_size", "N"]:
            input_shape[shape_idx] = 1
    input_shape_list.append(input_shape)

# Generate random input
maxN = 50  # number of random input
imgs = []
for input_idx in range(len(input_name_list)):
    imgs.append([np.random.rand(*input_shape_list[input_idx]).astype(np.float32) \
        for _ in range(maxN)])

# Static Quantization
quantize_model_path = '.'.join(model_path.split('.')[:-1]) + '_q.onnx'
quantize_static(preprocessed_model_path, quantize_model_path, \
                calibration_data_reader=DataReader(input_name_list, imgs), \
                weight_type=QuantType.QInt8, \
                quant_format=QuantFormat.QOperator)