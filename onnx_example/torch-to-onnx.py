import numpy as np
import onnx
import onnxruntime
import torch.onnx
import torch.utils.model_zoo as model_zoo
from onnx_example.model import SR


# ONNX's supports operators: https://github.com/onnx/onnx/blob/main/docs/Operators.md
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def torch_to_onnx(torch_model, x):
    # convert to onnx using ONNX API
    torch.onnx.export(torch_model,
                      x,
                      "super_resolution.onnx",
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
    print("Model converted to ONNX.")

    # Load the ONNX model
    onnx_model = onnx.load("super_resolution.onnx")
    onnx.checker.check_model(onnx_model)  # 모델의 구조를 확인하고 모델이 valid schema를 가지고 있는지를 체크
    print("Model checked.")

    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(onnx_model.graph))
    # print("Model graph printed.")

    # Inference with ONNX Runtime
    options = onnxruntime.SessionOptions()
    options.inter_op_num_threads = 8
    options.intra_op_num_threads = 8
    ort_session = onnxruntime.InferenceSession("super_resolution.onnx", options)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    return ort_outs

def compare_torch_onnx(torch_out, ort_outs):
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == '__main__':
    torch_model = SR(upscale_factor=3)

    # weights import
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    torch_model.load_state_dict(model_zoo.load_url(model_url))

    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))  # Load the model from the URL

    # Input to the model
    torch_model.eval()
    x = torch.randn(1, 1, 224, 224, requires_grad=True)
    torch_out = torch_model(x)  # model output for checking

    ort_outs = torch_to_onnx(torch_model, x)
    compare_torch_onnx(torch_out, ort_outs)