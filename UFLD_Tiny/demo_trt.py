from __future__ import print_function

import os
import argparse
import cv2
import tensorrt as trt
import common
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
import time
import scipy.special
import torchvision.transforms as transforms
from PIL import Image

# python demo_trt.py --model weights/lane_fp16
# python demo_trt.py --model weights/lane_int8

img_transforms = transforms.Compose([
    transforms.Resize((128, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

img_w = 960
img_h = 480
griding_num = 96
col_sample = np.linspace(0, 384 - 1, griding_num)
col_sample_w = col_sample[1] - col_sample[0]



row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
              116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
              168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
              220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
              272, 276, 280, 284]

color = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)]

EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))


def load_engine(trt_file_path, verbose=False):
    """Build a TensorRT engine from a TRT file."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    print('Loading TRT file from path {}...'.format(trt_file_path))
    with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose output (for debugging)')
    parser.add_argument(
        '-m', '--model', type=str, default='model',
    )
    args = parser.parse_args()

    trt_file_path = '%s.trt' % args.model
    if not os.path.isfile(trt_file_path):
        raise SystemExit('ERROR: file (%s) not found!' % trt_file_path)
    engine_file_path = '%s.trt' % args.model
    engine = load_engine(trt_file_path, args.verbose)

    h_inputs, h_outputs, bindings, stream = common.allocate_buffers(engine)

    with engine.create_execution_context() as context:
        for i in range(30):
            path = "./Inference/5.jpg"
            frame = cv2.imread(path)
            t1 = time.time()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img_transforms(img).numpy()

            h_inputs[0].host = img

            t3 = time.time()
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=h_inputs, outputs=h_outputs,
                                                 stream=stream)
            t4 = time.time()

            out_j = trt_outputs[0].reshape(97, 56, 4)  # tiny版本不一样

            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)

            idx = np.arange(griding_num) + 1
            idx = idx.reshape(-1, 1, 1)

            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == griding_num] = 0
            out_j = loc

            # import pdb; pdb.set_trace()
            vis = frame
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (
                            int(out_j[k, i] * col_sample_w * img_w / 384) - 1, int(img_h * (row_anchor[k] / 128)) - 1)
                            cv2.circle(vis, ppp, img_w // 300, color[i], -1)

            t2 = time.time()
            print('Inference time', (t4 - t3) * 1000)
            print('FPS', int(1 / ((t2 - t1))))
            cv2.imwrite( "results_tiny/" + str( i ) + "_res.jpg", vis )


if __name__ == '__main__':
    main()
