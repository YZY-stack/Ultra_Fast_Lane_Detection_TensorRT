# Ultra_Fast_Lane_Detection_TensorRT
An ultra fast tiny model for lane detection, using onnx_parser, TensorRTAPI to accelerate. our model support for int8, dynamic input and profiling. (TRT-hackathon2021)<br/>
这是一个基于TensorRT加速UFLD的repo，对源码和论文感兴趣的请参见：https://github.com/cfzd/Ultra-Fast-Lane-Detection <br/> <br/>

1. build onnx（将训练好的pth/pt模型转换为onnx）<br/>
---

<br/>static（生成静态onnx模型）:<br/> python3 torch2onnx.py onnx_dynamic_int8/configs/tusimple_4.py --test_model /home/stevenyan/TRT_python/UFLD_torch2trt/tusimple_18.pth<br/> dynamic（生成支持动态输入的onnx模型）:
first: vim torch2onnx.py<br/>
second: change "fix" from "True" to "False"<br/>
last: python3 torch2onnx.py onnx_dynamic_int8/configs/tusimple_4.py --test_model /home/stevenyan/TRT_python/UFLD_torch2trt/tusimple_18.pth<br/>

2. build engine（将onnx模型转换为TensorRT的推理引擎）<br/>
---
We support many different types of engine export, such as static fp32, fp16, dynamic fp32, fp16, and int8 quantization<br/>
我们支持多种不同类型engine的导出，例如：静态fp32、fp16，动态fp32、fp16，以及int8的量化<br/>


###static(fp32, fp16):<br/>
对于静态模型的导出，终端输入：<br/>
fp32:<br/>
python3 build_engine.py --onnx_path model_static.onnx --mode fp32<br/>
fp16:<br/>
python3 build_engine.py --onnx_path model_static.onnx --mode fp16<br/>


###dynamic(fp32, fp16):<br/>
对于动态模型的导出，终端输入：<br/>
fp32<br/>
python3 build_engine.py --onnx_path model_dynamic.onnx --mode fp32 --dynamic<br/>
fp16<br/>
python3 build_engine.py --onnx_path model_dynamic.onnx --mode fp16 --dynamic<br/>


###int8 quantization<br/>
如果想使用int8量化，终端输入：<br/>
python3 build_engine.py --onnx_path model_static.onnx --mode int8 --int8_data_path data/testset<br/>
（int8_data_Path represents the calibration dataset）
（其中int8_data_path表示校正数据集）<br/>


3. evaluate(compare)<br/>
---
python3 evaluate.py --pth_path PATH_OF_PTH_MODEL --trt_path PATH_OF_TRT_MODEL<br/>
（If you want to compare the acceleration and accuracy of reasoning through TRT with using pytorch, you can run the script）<br/>
（如果您想要比较通过TRT推理后，相对于使用PyTorch的加速以及精确度情况，可以运行该脚本）<br/>


