# Ultra_Fast_Lane_Detection_TensorRT
An ultra fast tiny model for lane detection, using onnx_parser, TensorRTAPI to accelerate. our model support for int8, dynamic input and profiling. (TRT-hackathon2021)<br/>
这是一个基于TensorRT加速UFLD的repo，包含PyThon ONNX Parser以及C++ TensorRT API版本, 还包括Torch2TRT版本, 
对源码和论文感兴趣的请参见：https://github.com/cfzd/Ultra-Fast-Lane-Detection <br/> <br/>

## 一. PyThon ONNX Parser
### 1. Build ONNX（将训练好的pth/pt模型转换为onnx）
```
1) static（生成静态onnx模型）:
python3 torch2onnx.py onnx_dynamic_int8/configs/tusimple_4.py --test_model ./tusimple_18.pth 

2) dynamic（生成支持动态输入的onnx模型）:
First: vim torch2onnx.py
second: change "fix" from "True" to "False"
python3 torch2onnx.py onnx_dynamic_int8/configs/tusimple_4.py --test_model ./tusimple_18.pth

```

### 2. Build trt engine（将onnx模型转换为TensorRT的推理引擎）<br/>
We support many different types of engine export, such as static fp32, fp16, dynamic fp32, fp16, and int8 quantization<br/>
我们支持多种不同类型engine的导出，例如：静态fp32、fp16，动态fp32、fp16，以及int8的量化<br/>


### static(fp32, fp16): 对于静态模型的导出，终端输入：
```
fp32:
python3 build_engine.py --onnx_path model_static.onnx --mode fp32<br/>
fp16:
python3 build_engine.py --onnx_path model_static.onnx --mode fp16<br/>
```

### dynamic(fp32, fp16): 对于动态模型的导出，终端输入：
```
fp32:
python3 build_engine.py --onnx_path model_dynamic.onnx --mode fp32 --dynamic
fp16:
python3 build_engine.py --onnx_path model_dynamic.onnx --mode fp16 --dynamic
```

### int8 quantization 如果想使用int8量化，终端输入：

```
python3 build_engine.py --onnx_path model_static.onnx --mode int8 --int8_data_path data/testset1000
# （int8_data_Path represents the calibration dataset）
# （其中int8_data_path表示校正数据集)
```

### 3. evaluate(compare)<br/>
（If you want to compare the acceleration and accuracy of reasoning through TRT with using pytorch, you can run the script）<br/>
（如果您想要比较通过TRT推理后，相对于使用PyTorch的加速以及精确度情况，可以运行该脚本）<br/>

```
python3 evaluate.py --pth_path PATH_OF_PTH_MODEL --trt_path PATH_OF_TRT_MODEL
```

## 二. torch2trt
  torch2trt is an easy tool to convert pytorch model to tensorrt, you can check model details here:  <br/>
  https://github.com/NVIDIA-AI-IOT/torch2trt <br/>
（torch2trt 是一个易于使用的PyTorch到TensorRT转换器）<br/>
#### 生成trt模型
```
python3 export_trt.py

```
#### torch2trt 预测demo (可视化)
```
python3 demo_torch2trt.py --trt_path PATH_OF_TRT_MODEL --data_path PATH_OF_YOUR_IMG
```
#### evaluated
```
python3 evaluate.py --pth_path PATH_OF_PTH_MODEL --trt_path PATH_OF_TRT_MODEL --data_path PATH_OF_YOUR_IMG --torch2trt
```

## 三. C++ TensorRT API
### 生成权重文件 
```
python3 export_trtcy.py
```
### trt模型生成
#### 修改第十行为 #define USE_FP32,则为FP32模式, 修改第十行为 #define USE_FP16,则为FP16模式
```
mkdir build
cd build
cmake ..
make
./lane_det -transfer             //  'lane_det.engine'
```
### Tensorrt预测
```
./lane_det -infer  ../imgs 

```

## 四. trtexec
trtexec  --explicitBatch --minShapes=1x3x288x800 --optShapes=1x3x288x800 --maxShapes=16x3x288x800 --shapes=4x3x288x800 --loadEngine=lane_fp32_dynamic.trt --noDataTransfers --dumpProfile --separateProfileRun
