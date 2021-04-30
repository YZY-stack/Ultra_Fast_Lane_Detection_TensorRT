from __future__ import print_function

import os
import argparse
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

img_transforms = transforms.Compose([
    transforms.Resize((128, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))



class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=16):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = self.load_data(training_data)
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    # Returns a numpy buffer of shape (num_images, 1, 28, 28)
    def load_data(self, datapath):
        print("loading image data")
        imgs = os.listdir(datapath)
        dataset = []
        for data in imgs:
            img = cv2.imread(datapath+data)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img_transforms(img).numpy()
            dataset.append(img)
            print(dataset)
        return np.array(dataset)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_int8_engine(onnx_file_path, calib, batch_size, verbose=False):
    """Build a TensorRT engine from an ONNX file."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30
        builder.max_batch_size = 1
        builder.int8_mode = True
        builder.int8_calibrator = calib

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

        print('Adding yolo_layer plugins...')
        model_name = onnx_file_path[:-5]

        print('Building an engine.  This would take a while...')
        print('(Use "--verbose" to enable verbose logging.)')
        engine = builder.build_cuda_engine(network)
        print('Completed creating engine.')
        return engine


def main():
    """Create a TensorRT engine for ONNX-based model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose output (for debugging)')
    parser.add_argument(
        '-m', '--model', type=str, default='model.onnx',
        )
    args = parser.parse_args() 

    calibration_cache = "calibration.cache"  
    data_path = 'calibration_data/testset/' 
    calib = EntropyCalibrator(data_path, cache_file=calibration_cache)    

    onnx_file_path = args.model
    if not os.path.isfile(onnx_file_path):
        raise SystemExit('ERROR: file (%s) not found!' % onnx_file_path)
    engine_file_path = '%s_int8.trt' % args.model[:-5]
    engine =  build_int8_engine(onnx_file_path, calib, 16)
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    print('Serialized the TensorRT engine to file: %s' % engine_file_path)



if __name__ == '__main__':
    main()
