
import torch
# 1. Get a quantized Tensor by quantizing unquantized float Tensors
float_tensor = torch.randn(2, 2, 3)

print (float_tensor.is_quantized)

scale, zero_point = 1e-4, 2
dtype = torch.qint8
q_per_tensor = torch.quantize_per_tensor(float_tensor, scale, zero_point, dtype)
q_per_tensor2 = torch.quantize_per_tensor(float_tensor, scale, zero_point, dtype)

print (q_per_tensor)
try:
    print (q_per_tensor+q_per_tensor2)
    assert False
except:
    dequantized_tensor = torch.quantize_per_tensor(q_per_tensor.dequantize() + q_per_tensor2.dequantize(), scale, zero_point, dtype)
    assert True

    print (dequantized_tensor)

    assert q_per_tensor.q_scale() == q_per_tensor2.q_scale()
    assert q_per_tensor.q_zero_point() == q_per_tensor2.q_zero_point()

    print (q_per_tensor.int_repr())