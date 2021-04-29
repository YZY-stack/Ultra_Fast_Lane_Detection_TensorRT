from __future__ import print_function

from functools import reduce
import tensorrt as trt
from onnx_dynamic_int8.common import *
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
import time
import torchvision.transforms as transforms
from PIL import Image
import torch, os, cv2
from onnx_dynamic_int8.model.model import parsingNet
import torch

from onnx_dynamic_int8.onnx_parser import *


def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verbose', action='store_true')
    parser.add_argument(
        '--onnx_path', type=str, default='model.onnx')
    parser.add_argument(
        '--trt_path', type=str, default='lane.trt')
    parser.add_argument(
        '--mode', type=str, default='fp16')
    parser.add_argument(
        '--dynamic', action='store_true')
    parser.add_argument(
        '--int8_data_path', type=str,
        default='calibration_data/testset/',
        help='set if you want to do int8 inference')
    parser.add_argument(
        '--data_path', type=str, default='/home/stevenyan/TRT/Inference/5.jpg')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # paths and configs
    configs = set_config()
    onnx_path = configs.onnx_path
    int8_data_path = configs.int8_data_path
    trt_file_path = configs.trt_path
    dynamic_input = configs.dynamic
    data_path = configs.data_path
    mode = configs.mode

    # build engine
    engine = build_engine(onnx_path, trt_file_path, mode, int8_data_path,
                          dynamic_input=dynamic_input, verbose=True)

