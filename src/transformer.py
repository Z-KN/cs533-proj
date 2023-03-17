import torch
import torch.nn as nn


class M(nn.Module):
  def __init__(self):
    super().__init__()
    self.transformer = nn.Transformer(d_model=128)

  def forward(self, x, y):
    return self.transformer(x, y)


module = M()
x = torch.randn([10, 1, 128])
y = torch.randn([10, 1, 128])
dummy_input = (x, y)
torch.onnx.export(module, dummy_input, 'transformer.onnx', verbose=True, opset_version=14)