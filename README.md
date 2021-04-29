# Ultra_Fast_Lane_Detection_TensorRT
An ultra fast tiny model for lane detection, using onnx_parser, TensorRTAPI to accelerate. our model support for int8, dynamic input and profiling. (TRT-hackathon2021)

# Ultra_Fast_Lane_Detection_TensorRT
An ultra fast tiny model for lane detection, using onnx_parser, TensorRTAPI to accelerate. our model support for int8, dynamic input and profiling. (TRT-hackathon2021)


1. build onnx

static:
python3 torch2onnx.py onnx_dynamic_int8/configs/tusimple_4.py --test_model /home/stevenyan/TRT_python/UFLD_torch2trt/tusimple_18.pth

dynamic:
vim torch2onnx.py
change "fix" from "True" to "False", then, run:
python3 torch2onnx.py onnx_dynamic_int8/configs/tusimple_4.py --test_model /home/stevenyan/TRT_python/UFLD_torch2trt/tusimple_18.pth



2. build engine

---------static(fp32, fp16):
fp32:
python3 build_engine.py --onnx_path model_static.onnx --mode fp32
fp16:
python3 build_engine.py --onnx_path model_static.onnx --mode fp16

---------dynamic(fp32, fp16):
fp32
python3 build_engine.py --onnx_path model_dynamic.onnx --mode fp32 --dynamic
fp16
python3 build_engine.py --onnx_path model_dynamic.onnx --mode fp16 --dynamic

---------int8
python3 build_engine.py --onnx_path model_static.onnx --mode int8 --int8_data_path data/testset



3. evaluate(compare)
python3 evaluate.py --pth_path PATH_OF_PTH --trt_path PATH_OF_TRT 
