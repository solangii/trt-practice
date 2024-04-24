import os
import onnx
import torch
import time
import psutil
import os
import onnxruntime

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# serialization
onnx_model = onnx.load("super_resolution.onnx")

with open("super_resolution_serialization.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# model size comparison
onnx_memory = os.path.getsize("super_resolution.onnx")
onnx_serialization_memory = os.path.getsize("super_resolution_serialization.onnx")

print(f"ONNX model memory usage: {onnx_memory} bytes")
print(f"ONNX serialization model memory usage: {onnx_serialization_memory} bytes")

# inference speed
options = onnxruntime.SessionOptions()
options.inter_op_num_threads = 8
options.intra_op_num_threads = 8
ort_session = onnxruntime.InferenceSession("super_resolution.onnx", options)
ort_session_serialization = onnxruntime.InferenceSession("super_resolution_serialization.onnx", options)

x = torch.randn(1, 1, 224, 224, requires_grad=True)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}

start_time = time.time()
start_mem = psutil.Process(os.getpid()).memory_info().rss
ort_outs = ort_session.run(None, ort_inputs)
end_time = time.time()
end_mem = psutil.Process(os.getpid()).memory_info().rss

onnx_time = end_time - start_time
onnx_mem = end_mem - start_mem
print(f"ONNX Runtime execution time: {onnx_time} seconds, memory usage: {onnx_mem} bytes")

start_time = time.time()
start_mem = psutil.Process(os.getpid()).memory_info().rss
ort_outs = ort_session_serialization.run(None, ort_inputs)
end_time = time.time()
end_mem = psutil.Process(os.getpid()).memory_info().rss

onnx_time = end_time - start_time
onnx_mem = end_mem - start_mem
print(f"ONNX-serialization Runtime execution time: {onnx_time} seconds, memory usage: {onnx_mem} bytes")
