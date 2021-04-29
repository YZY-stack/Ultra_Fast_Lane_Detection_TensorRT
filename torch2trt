import argparse
from onnx_dynamic_int8.UFLD import *
import cv2
import time
import numpy as np
import torch
import onnx


detector = laneDetection()
detector.setResolution(640, 480)
frame = 0
currentImage = None


if __name__ == "__main__":
    fix = True
    input_data = torch.rand((1, 3, 288, 800)).cuda()
    if fix:
        file_path = 'model_static.onnx'
        torch.onnx.export(detector.net, input_data, file_path, verbose=True)
    else:
        '''export dynamic shape for TRT'''
        input_names = ["input"]
        output_names = ["output"]
        file_path = 'model_dynamic.onnx'
        torch.onnx.export(detector.net, input_data, file_path, verbose=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
