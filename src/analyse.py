import onnx
import onnx.numpy_helper
# Load the ONNX model
model = onnx.load('resnet18-v2-7.onnx')

# Create a new graph with batch normalization nodes removed
new_graph = onnx.GraphProto()
new_graph.CopyFrom(model.graph)

# Loop through all of the nodes in the graph
for node in new_graph.node:
    # If the node is a batch normalization node, remove it
    if node.op_type == 'BatchNormalization':
        for output in node.output:
            # Find all of the nodes that use the output of the batch normalization node
            for i, n in enumerate(new_graph.node):
                if output in n.input:
                    # Replace the input of the following nodes with the input of the batch normalization node
                    new_graph.node[i].input[list(n.input).index(output)] = node.input[0]
        # Remove the batch normalization node from the graph
        new_graph.node.remove(node)

# Save the modified ONNX model
onnx.save(onnx.ModelProto(graph=new_graph), 'resnet_no_bn.onnx')

# Load the ONNX model
model = onnx.load("resnet_no_bn.onnx")
model.ir_version = onnx.IR_VERSION
opset_import = model.opset_import.add()
opset_import.version = 11
# Infer the shapes of all tensors in the model
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)

# Get the graph from the model
graph = model.graph

# Build a dictionary mapping tensor names to their shapes
tensor_shapes = {}
for tensor in graph.initializer:
    tensor_shapes[tensor.name] = tuple(tensor.dims)
for input in graph.input:
    tensor_shapes[input.name] = tuple(dim.dim_value for dim in input.type.tensor_type.shape.dim)
for node in graph.node:
    for output in node.output:
        print("----", node.attribute)
        # here are some problems
        tensor_shapes[output] = tuple(dim.dim_value for dim in node.attribute[0].t.dims)

# Loop through all of the nodes in the graph
for node in graph.node:
    # Print the operator type
    print("Operator Type:", node.op_type)
    # Print the input names and shapes
    print("Inputs:")
    for i, input in enumerate(node.input):
        input_shape = tensor_shapes[input]
        print("  Input %d: %s (shape=%s)" % (i, input, input_shape))
    # Print the output names and shapes
    print("Outputs:")
    for i, output in enumerate(node.output):
        try:
            output_shape = tensor_shapes[output]
        except KeyError:
            output_shape = "Unknown"
        print("  Output %d: %s (shape=%s)" % (i, output, output_shape))
        tensor_shapes[output] = output_shape
    # Print any attributes for the operator
    print("Attributes:")
    for attribute in node.attribute:
        print("  %s: %s" % (attribute.name, attribute))

