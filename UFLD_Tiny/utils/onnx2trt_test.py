import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit 
import torch
import os
import common

max_batch_size = 1
onnx_model_name = 'model.onnx'
TRT_LOGGER = trt.Logger()  # This logger is required to build an engine

def build_engine(onnx_file_path):
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.fp16_mode = False
        builder.max_batch_size = 1
        builder.max_workspace_size = 1<<30
        print('Loading ONNX file from path {}...'.format(onnx_model_name))
        with open(onnx_model_name, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print("num layers:",network.num_layers)

    network.get_input(0).shape = [1, 3, 288, 800]
    
    print('Completed parsing of ONNX file')
    print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
    engine = builder.build_cuda_engine(network)
    print("Completed creating Engine")
    with open(trt_model_name, "wb") as f:
        f.write(engine.serialize())
    return engine


# These two modes are dependent on hardwares
fp16_mode = False
int8_mode = False
trt_engine_path = './model.trt'
# Build an engine
engine = build_engine(onnx_model_name)
print("OK")



