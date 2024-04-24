import time
import psutil
import os
import onnx
import torch

from onnx_example.model import SR
import onnxruntime

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    torch_model = SR(upscale_factor=3)
    torch_model.eval()

    options = onnxruntime.SessionOptions()
    options.inter_op_num_threads = 8
    options.intra_op_num_threads = 8
    ort_session = onnxruntime.InferenceSession("super_resolution.onnx", options)
    ort_session_serialization = onnxruntime.InferenceSession("super_resolution_serialization.onnx", options)

    x = torch.randn(1, 1, 224, 224, requires_grad=True)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}

    # Measure the memory usage and execution time in PyTorch
    start_time = time.time()
    start_mem = psutil.Process(os.getpid()).memory_info().rss
    torch_out = torch_model(x)
    end_time = time.time()
    end_mem = psutil.Process(os.getpid()).memory_info().rss

    torch_time = end_time - start_time
    torch_mem = end_mem - start_mem
    print(f"PyTorch execution time: {torch_time} seconds, memory usage: {torch_mem} bytes")

    # Measure the memory usage and execution time in ONNX Runtime
    start_time = time.time()
    start_mem = psutil.Process(os.getpid()).memory_info().rss
    ort_outs = ort_session.run(None, ort_inputs)
    end_time = time.time()
    end_mem = psutil.Process(os.getpid()).memory_info().rss

    onnx_time = end_time - start_time
    onnx_mem = end_mem - start_mem
    print(f"ONNX Runtime execution time: {onnx_time} seconds, memory usage: {onnx_mem} bytes")

    # Measure the memory usage and execution time in ONNX Runtime with serialization
    start_time = time.time()
    start_mem = psutil.Process(os.getpid()).memory_info().rss
    ort_outs = ort_session_serialization.run(None, ort_inputs)
    end_time = time.time()
    end_mem = psutil.Process(os.getpid()).memory_info().rss

    onnx_time = end_time - start_time
    onnx_mem = end_mem - start_mem
    print(f"ONNX-serialization Runtime execution time: {onnx_time} seconds, memory usage: {onnx_mem} bytes")