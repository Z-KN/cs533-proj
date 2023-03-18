import onnx
import onnxruntime as ort
import onnx.numpy_helper
import numpy as np
from collections import OrderedDict

# load ONNX model
# model_path = 'resnet18-v2-7.onnx'
model_path = 'transformer.onnx'
# model=onnx.load(model_path)
# input_info = model.graph.input
# output_info = model.graph.output

# print("Input shapes:")
# for input in input_info:
#     print("  ", input.name, ":", [d.dim_value for d in input.type.tensor_type.shape.dim])

# print("Output shapes:")
# for output in output_info:
#     print("  ", output.name, ":", [d.dim_value for d in output.type.tensor_type.shape.dim])

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
# in_img = np.random.randn(1,3,224,224).astype(np.float32)
x = np.random.randn(10, 1, 128).astype(np.float32)
y = np.random.randn(10, 1, 128).astype(np.float32)
ort_outs = ort_session.run(outputs, {ort_session.get_inputs()[0].name: x, \
                                     ort_session.get_inputs()[1].name: y} )
# ort_outs = ort_session.run(outputs, {ort_session.get_inputs()[0].name: in_img} )
ort_outs = [single_out.shape for single_out in ort_outs]
ort_outs = OrderedDict(zip(outputs, ort_outs))
print(ort_outs)