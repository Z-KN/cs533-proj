import onnx
import onnx.numpy_helper
import onnxruntime
import numpy as np
# Load the ONNX model
model = onnx.load('resnet18-v2-7.onnx')

# Create a new graph with batch normalization nodes removed
# new_graph = onnx.GraphProto()
# new_graph.CopyFrom(model.graph)

input_info = model.graph.input
output_info = model.graph.output

print("Input shapes:")
for input in input_info:
    print("  ", input.name, ":", [d.dim_value for d in input.type.tensor_type.shape.dim])

print("Output shapes:")
for output in output_info:
    print("  ", output.name, ":", [d.dim_value for d in output.type.tensor_type.shape.dim])

# session = onnxruntime.InferenceSession('resnet18-v2-7.onnx')
# print("Output shapes:")
# for input, output in zip(input_info, output_info):
#     output_name = output.name
#     output_shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
#     # Create a random input tensor with the same shape as the input tensor
#     input_name = input.name
#     input_shape = [d.dim_value for d in input.type.tensor_type.shape.dim]
#     print(input_shape)
#     input_data = np.random.rand(*(0,3,224,224)).astype(np.float32)
#     print(input_data)
#     # Create an ONNXRuntime InferenceSession and run the model to obtain the shape of the output tensor
#     output_data = session.run([output_name], {input_name: input_data})
#     output_shape = [d for d in output_data[0].shape]
#     print("  ", output.name, ":", output_shape)


# # Loop through all of the nodes in the graph
# for node in new_graph.node:
#     # If the node is a batch normalization node, remove it
#     if node.op_type == 'BatchNormalization':
#         for output in node.output:
#             # Find all of the nodes that use the output of the batch normalization node
#             for i, n in enumerate(new_graph.node):
#                 if output in n.input:
#                     # Replace the input of the following nodes with the input of the batch normalization node
#                     new_graph.node[i].input[list(n.input).index(output)] = node.input[0]
#         # Remove the batch normalization node from the graph
#         new_graph.node.remove(node)

# # Save the modified ONNX model
# onnx.save(onnx.ModelProto(graph=new_graph), 'resnet_no_bn.onnx')

# # Load the ONNX model
# model = onnx.load("resnet_no_bn.onnx")
# model.ir_version = onnx.IR_VERSION
# opset_import = model.opset_import.add()
# opset_import.version = 11
# # Infer the shapes of all tensors in the model
# onnx.checker.check_model(model)
# onnx.helper.printable_graph(model.graph)

# # Get the graph from the model
# graph = model.graph

# # Build a dictionary mapping tensor names to their shapes
# tensor_shapes = {}
# for tensor in graph.initializer:
#     tensor_shapes[tensor.name] = tuple(tensor.dims)
# for input in graph.input:
#     tensor_shapes[input.name] = tuple(dim.dim_value for dim in input.type.tensor_type.shape.dim)
# for node in graph.node:
#     for output in node.output:
#         print("----", node.attribute)
#         # here are some problems
#         tensor_shapes[output] = tuple(dim.dim_value for dim in node.attribute[0].t.dims)

# Loop through all of the nodes in the graph
# for node in new_graph.node:
#     # Print the operator type
#     print("Operator Type:", node.op_type)
#     # Print the input names and shapes
#     print("Inputs:")
#     for i, input in enumerate(node.input):
#         print("  Input %d: %s" % (i, input))
#     # Print the output names and shapes
#     print("Outputs:")
#     for i, output in enumerate(node.output):
#         print("  Output %d: %s" % (i, output))
#     # Print any attributes for the operator
#     print("Attributes:")
#     for attribute in node.attribute:
#         print("  %s: %s" % (attribute.name, attribute))

