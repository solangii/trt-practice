## What I Learned about ONNX

> ONNX 공식 문서 요약정리 및 분석 공부

### Concepts

- ONNX(Open Neural Network Exchange)는 **ML프레임워크에 상관없이 독립적으로 모델을 제공**할 수 있도록하는 공통 언어를 제공하고, ONNX runtime (or interpreter)와 함께 배포 환경에서 원하는 작업에 맞게 **최적화** 시킬 수 있음.
- ONNX graph를 만든다는 것은 [ONNX Operator](https://onnx.ai/onnx/operators/index.html#l-onnx-operators)로 함수를 구현한다는 것. 기본적으로 input, output, node(**operator**), initializer로 구성됨.
  - 현재 지원하는 operator 목록 및 schema ([Link](https://github.com/onnx/onnx/blob/main/docs/Operators.md))
  - 지원하지 않는 operator에 대해서는 사용자가 따로 정의 가능하며, inference시의 최적화를 위한 operator custom도 가능함. ([Sample code](https://onnx.ai/onnx/expect_onnxruntime.html#l-function-expect))
- tests([If](https://onnx.ai/onnx/operators/onnx__If.html#l-onnx-doc-if))나 loops([loop](https://onnx.ai/onnx/operators/onnx__Loop.html#l-onnx-doc-loop), [scan](https://onnx.ai/onnx/operators/onnx__Scan.html#l-onnx-doc-scan))는 **attribute**로 구현됨. 그러나 느리고 복잡해서 가능한 사용을 피하는 것이 좋음.
- input, outpu의 Type이나 Shape을 아는 것은 inference 최적화에 도움을 줄 수 있음. (**Human handling이 필요한 부분**임)
  - 예를 들어서, `Add(x, y) -> z , Abs(z) -> w`의 경우, out-place연산으로 z라는 새로운 객체를 생성해서 계산하고 있는 상황. 이때, 만약 x,y의 shape이 같다면 z, w도 같은 shape일 태니, **z에 대한 buffer를 재사용**하도록 w를 계산하면 memory, runtime에서 효율이 있음.



### Serialization

[Protos](https://onnx.ai/onnx/api/classes.html#l-onnx-classes) page에 list되어있는 모든 객체들은 serialization가능함. Serialization의 이점은 [이 글]( https://hub1234.tistory.com/26) 참고.

#### Model Serialization

- model의 구조와 가중치 등을 포함하는 binary형식의 문자열로 변환하는 과정.

- ONNX model은 protobuf기반으로 저장되어, graph를 disk에 저장하기 위한 공간을 최소화 한다.

  - [protobuf](https://protobuf.dev/getting-started/pythontutorial/#parsing-and-serialization)(protocol buffer): Google에서 개발한 데이터 직렬화 형식. 

- `SerializeToString()` 함수를 이용해서 직렬화 한다.

  ```python
  with open("model.onnx", "wb") as f:
      f.write(onnx_model.SerializeToString())
  ```

#### Data Serialization

- Serialization: `SerializeToString()`
- Deserialization: byte →`TensorProto` → array, `load_tensor_from_string()`

```python
# serialization
numpy_tensor = numpy.array([0, 1, 4, 5, 3], dtype=numpy.float32) # <class 'numpy.ndarray'>
onnx_tensor = onnx.numpy_helper.from_array(numpy_tensor) # <class 'onnx.onnx_ml_pb2.TensorProto'>
serialized_tensor = onnx_tensor.SerializeToString() # <class 'bytes'>

with open("saved_tensor.pb", "wb") as f:
    f.write(serialized_tensor)

# deserialization
with open("saved_tensor.pb", "rb") as f:
    serialized_tensor = f.read() # <class 'bytes'>
onnx_tensor = onnx.TensorProto() # <class 'onnx.onnx_ml_pb2.TensorProto'>
numpy_tensor = to_array(onnx_tensor) # <class 'numpy.ndarray'>

# simplified version. (using load_tensor_from_string)
with open("saved_tensor.pb", "rb") as f:
    serialized = f.read()
proto = onnx.load_tensor_from_string(serialized) # <class 'onnx.onnx_ml_pb2.TensorProto'>
```



### CheatSheet

#### onnx.helper 

- [helper module](https://onnx.ai/onnx/api/helper.html#l-onnx-make-function)
- `onnx.helper.printable_graph()`: ONNX 모델의 그래프를 사람이 읽을 수 있는 형태로 출력하는 역할. 

graph build에 관한 함수들:

- `onnx.helper.make_tensor_value_info()`: input 또는 output의 shape과 type을 정의

  - `None`은 any shape을 의미, [`None`, `None`]은 2차원을 의미

  ```python
  # (name, type, shape)
  X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
  A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
  B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
  Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
  ```

- `onnx.helper.make_node()`: node의 operation과 input, output을 정의

  ```python
  # (operator type, input, output)
  node1 = make_node('MatMul', ['X', 'A'], ['XA']) 
  node2 = make_node('Add', ['XA', 'B'], ['Y'])
  ```

- `onnx.helper.make_graph()`: 위 두 함수에 의해 생성된 객체들로 onnx graph 생성

  ```python
  # (nodes, name, input, output)
  graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
  ```

- `onnx.helper.make_model()`: graph와 metadata(optional)을 merge해줌

  ```python
  # onnx graph (metadata 없음)
  onnx_model = make_model(graph)
  ```

#### onnx.chcker 

- [checker 모듈](https://onnx.ai/onnx/api/checker.html)은 onnx모델의 유효성을 검사하는 함수들을 제공함. 
- `onnx.checker.check_model()`: ONNX 모델이 ONNX 명세에 따라 모델의 구조, 데이터 타입, 다른 속성들이 올바르게 정의되었는지 확인하는 역할. 만약 모델이 ONNX 명세를 준수하지 않으면, 이 함수는 에러를 발생시킴. 



### Some Useful Tools

- [netron](https://netron.app/): `.onnx`파일을 로드해서 [Image](./results/super_resolution.onnx.png)처럼 graph visualize할 수 있다.
- [onnx2py.py](https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/onnx2py.py): ONNX graph로부터 python file을 만들어줌. 
- [zetane](https://github.com/zetane/viewer) : `.onnx`파일을 로드해서 intermediate results들을 볼 수 있는데, open되어있는 무료버전에선 안되는 걸 확인..



### Analysis

1. torch model: super resolution model을 예시로 진행 ([code](./model.py))

2. convert: torch 모델을 onnx로 변환 ([code](./torch_to_onnx.py))

3. graph inspect: onnx graph의 input, output, node의 list,shape, type등 확인 ([code](./graph_inspect.py))

4. model serialization: ([code](./model_serialization.py))

   onnx 모델과 serialization된 모델의 disk 차지 용량은 정확하게 동일

   ```
   ONNX model memory usage: 240587 bytes
   ONNX serialization model memory usage: 240587 bytes
   ```

   Inference speed와 프로세스 실행시 사용되는 메모리 양에서는 **serialization된 모델이 이점**을 가짐

   ```
   ONNX Runtime execution time: 0.06416535377502441 seconds, memory usage: 27205632 bytes
   ONNX-serialization Runtime execution time: 0.0459902286529541 seconds, memory usage: 25071616 bytes
   ```

5. performance comparision: pytorch model과 onnx model & onnxruntime의 memory usage, inference time 비교 ([code](./performance.py))

   **ONNX-serialization > ONNX > PyTorch**

   ```
   PyTorch execution time: 0.08090472221374512 seconds, memory usage: 49864704 bytes
   ONNX Runtime execution time: 0.04454326629638672 seconds, memory usage: 24227840 bytes
   ONNX-serialization Runtime execution time: 0.03925037384033203 seconds, memory usage: 24014848 bytes
   ```

   

---

**Random Thought**

- ONNX는 플랫폼 간 호환성을 위한 것이고, 성능 최적화는 Runtime에서 수행하는 것. 따라서, ONNX가 **항상** TensorFlow나 PyTorch보다 추론 속도, 메모리 대비 효과적이라고 말할 수는 없다. (그래도 결과 분석해보니 효과가 있긴한 것 같다.)
- ONNX모델 자체에서 더 개선을 시키려면, shape/type을 확인해보면서 in-place연산이 가능한 것들을 변경해서 memory buffer를 줄인다거나, if / loop와 같은 attributes 사용을 줄인다거나 하는 정도로만 try해볼만 한 듯.



**References** 

- https://onnx.ai/onnx/intro/index.html
- https://hub1234.tistory.com/26
