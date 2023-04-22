import torch

# Define a tensor with some floating-point values
float_tensor = torch.tensor([1.2666, 3.4888, 5.776, 7.8888])

# Apply PyTorch's quantization function to convert the tensor to fixed-point values
quantized_tensor1 = torch.quantize_per_tensor(float_tensor, scale=0.1, zero_point=5, dtype=torch.qint8)
# quantized_tensor2 = torch.quantize_per_tensor(float_tensor, scale=0.3, zero_point=5, dtype=torch.qint8)

# Print the original floating-point tensor
print("Original tensor (floating-point values):", float_tensor)
print("Original tensor (floating-point values):", float_tensor)

# Print the quantized tensor with fixed-point values
print("Quantized tensor (fixed-point values):", quantized_tensor1)
print("Quantized tensor (fixed-point values):", quantized_tensor1.int_repr())
print("Quantized tensor (fixed-point values):", quantized_tensor1.type())

# Dequantize the quantized tensor back to floating-point values
dequantized_tensor = quantized_tensor1.dequantize()

# Print the dequantized tensor with floating-point values
print("Dequantized tensor (floating-point values):", dequantized_tensor)
print("Dequantized tensor (floating-point values):", dequantized_tensor.type())
