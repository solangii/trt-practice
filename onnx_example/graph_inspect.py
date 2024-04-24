# most code is from https://onnx.ai/onnx/intro/python.html

import onnx
from onnx.checker import check_model

def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)


onnx_model = onnx.load("super_resolution.onnx")
check_model(onnx_model)

# the list of inputs
print('******** list of inputs ********')
print(onnx_model.graph.input)

# in a more nicely format
print('******** inputs ********')
for obj in onnx_model.graph.input:
    print("name=%r dtype=%r shape=%r" % (
        obj.name, obj.type.tensor_type.elem_type,
        shape2tuple(obj.type.tensor_type.shape)))

# the list of outputs
print('******** list of outputs ********')
print(onnx_model.graph.output)

# in a more nicely format
print('******** outputs ********')
for obj in onnx_model.graph.output:
    print("name=%r dtype=%r shape=%r" % (
        obj.name, obj.type.tensor_type.elem_type,
        shape2tuple(obj.type.tensor_type.shape)))

# the list of nodes
print('******** list of nodes ********')
print(onnx_model.graph.node)

# in a more nicely format
print('******** nodes ********')
for node in onnx_model.graph.node:
    print("name=%r type=%r input=%r output=%r" % (
        node.name, node.op_type, node.input, node.output))

