from UFLD import *
import cv2
import time
import numpy as np
import torch
import torch.onnx as onnx


detector = laneDetection()
detector.setResolution(640, 480)
frame = 0
currentImage = None
        

if __name__ == "__main__":
    filepath = "model.onnx"
    dummy_input = torch.rand((1,3,288,800)).cuda()
    torch.onnx.export(detector.net, dummy_input, filepath)
