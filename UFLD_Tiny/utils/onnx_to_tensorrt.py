from __future__ import print_function

import os
import argparse
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

mode = 'fp16'

def build_engine(onnx_file_path, mode, verbose=False):
    """Build a TensorRT engine from an ONNX file."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30
        builder.max_batch_size = 1
        if mode=='fp16':
            builder.fp16_mode = True
        else:
            builder.fp16_mode = False
        #builder.strict_type_constraints = True

        # Parse model file
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        if trt.__version__[0] >= '7':
            # Reshape input to batch size 1
            shape = list(network.get_input(0).shape)
            shape[0] = 1
            network.get_input(0).shape = shape

        model_name = onnx_file_path[:-5]

        print('Building an engine.  This would take a while...')
        print('(Use "--verbose" to enable verbose logging.)')
        engine = builder.build_cuda_engine(network)
        print('Completed creating engine.')
        return engine


def main():
    """Create a TensorRT engine for ONNX-based Model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose output (for debugging)')
    parser.add_argument(
        '-m', '--model', type=str, default='model.onnx')
    parser.add_argument(
        '-p', '--precision', type=str, default='fp16')
    args = parser.parse_args() 

    mode = args.precision
    onnx_file_path = args.model
    if not os.path.isfile(onnx_file_path):
        raise SystemExit('ERROR: file (%s) not found!' % onnx_file_path)
    if mode=='fp16':
        engine_file_path = '%s_fp16.trt'% args.model[:-5]
    elif mode == 'fp32':
        engine_file_path = '%s_fp32.trt'% args.model[:-5]
    else:
        print("illegal mode")
        exit(0)
    engine = build_engine(onnx_file_path, mode,args.verbose)
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    print('Serialized the TensorRT engine to file: %s' % engine_file_path)



if __name__ == '__main__':
    main()
