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


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = img_transforms(img)
    return img


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=1):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.data = self.load_data(training_data)
        self.batch_size = batch_size
        self.current_index = 0
        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    # Returns a numpy buffer of shape (num_images, 1, 288, 800)
    def load_data(self, datapath):
        imgs = os.listdir(datapath)
        dataset = []
        for data in imgs:
            img = cv2.imread(datapath + data)
            img = preprocessing(img).numpy()
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


def build_engine(onnx_path, trt_file_path, mode, int8_data_path, dynamic_input=False, verbose=False):
    # export fixed input model
    if not dynamic_input:
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            print('Loading TRT file from path {}...'.format(trt_file_path))
            builder.max_batch_size = 1
            builder.max_workspace_size = 1 << 30
            print('Loading ONNX file from path {}...'.format(onnx_path))
            with open(onnx_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print("num layers:", network.num_layers)
                    return None
            if mode == 'fp16':
                builder.fp16_mode = True
                network.get_input(0).shape = [1, 3, 288, 800]
                trt_file_path = trt_file_path[:-4] + '_fp16_static.trt'
                print("build fp16 engine...")
            elif mode == 'fp32':
                network.get_input(0).shape = [1, 3, 288, 800]
                trt_file_path = trt_file_path[:-4] + '_fp32_static.trt'
                print("build fp32 engine...")
            else:
                # build an int8 engine
                calibration_cache = "calibration.cache"
                calib = EntropyCalibrator(int8_data_path, cache_file=calibration_cache)
                builder.int8_mode = True
                builder.int8_calibrator = calib
                network.get_input(0).shape = [16, 3, 288, 800]
                trt_file_path = trt_file_path[:-4] + '_int8.trt'
                print("build int8 engine...")
            engine = builder.build_cuda_engine(network)
            print("engine:", engine)
            print("Completed creating Engine")
            with open(trt_file_path, "wb") as f:
                f.write(engine.serialize())
        return engine

    # export dynamic input model
    else:
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size = 1
            config = builder.create_builder_config()
            print('Loading ONNX file from path {}...'.format(onnx_path))
            if mode == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
                trt_file_path = trt_file_path[:-4] + '_fp16_dynamic.trt'
            else:
                trt_file_path = trt_file_path[:-4] + '_fp32_dynamic.trt'
            # Load the Onnx model and parse it in order to populate the TensorRT network.
            with open(onnx_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # for dynamic input
            if dynamic_input:
                profile = builder.create_optimization_profile()
                data = network.get_input(0)
                print(data.shape)
                profile.set_shape(data.name, (1, 3, 288, 800), (6, 3, 288, 800), (8, 3, 288, 800))
                config.add_optimization_profile(profile)
                config.flags = 1 << int(trt.BuilderFlag.FP16)
                config.max_workspace_size = 1 << 30
            engine = builder.build_engine(network, config)
            print("Completed creating a dynamic Engine")
            print("engine:", engine)
            with open(trt_file_path, "wb") as f:
                f.write(engine.serialize())
        return engine


def load_engine(trt_file_path, verbose=False):
    """Build a TensorRT engine from a TRT file."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    print('Loading TRT file from path {}...'.format(trt_file_path))
    with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def inference_with_trt(trt_file_path, data_path, dynamic_input=False):
    global inputH0, inputD0, outputH0, outputD0, t1
    # inference with static shape
    if not dynamic_input:
        engine = load_engine(trt_file_path, verbose=False)
        h_inputs, h_outputs, bindings, stream = allocate_buffers(engine)
        t0 = 0
        with engine.create_execution_context() as context:
            for i in range(10):
                path = data_path
                frame = cv2.imread(path)
                img = preprocessing(frame).numpy()

                h_inputs[0].host = img
                t1 = time.time()
                trt_outputs = do_inference_static(context, bindings=bindings, inputs=h_inputs, outputs=h_outputs,
                                                  stream=stream, batch_size=1)
                t2 = time.time()
                time_trt_static = t2 - t1
                time_trt_static = t0 + time_trt_static
            print("Inference time with TensorRT_static = %.3f ms" % (time_trt_static * 1000))
        return time_trt_static, trt_outputs
    # inference with dynamic shape
    else:
        engine = load_engine(trt_file_path, verbose=False)
        with engine.create_execution_context() as context:
            context.set_binding_shape(0, (1, 3, 288, 800))
            stream = cuda.Stream()
            t0 = 0
            for i in range(10):
                path = data_path
                frame = cv2.imread(path)
                data = preprocessing(frame).numpy().reshape(3, 288, 800)
                inputH0 = np.ascontiguousarray(data.reshape(-1))
                inputD0 = cuda.mem_alloc(inputH0.nbytes)
                outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
                outputD0 = cuda.mem_alloc(outputH0.nbytes)

                bindings = []
                for binding in engine:
                    if binding == 'input':
                        size = trt.volume(context.get_binding_shape(0))
                    else:
                        size = trt.volume(context.get_binding_shape(1))
                    dtype = trt.nptype(engine.get_binding_dtype(1))
                    # Allocate host and device buffers
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    # Append the device buffer to device bindings.
                    bindings.append(int(device_mem))
                    t1 = time.time()

                trt_outputs_dynamic = do_inference_dynamic(context, bindings, inputH0, inputD0,
                                                           outputH0, outputD0, stream)
                t2 = time.time()
                time_trt_dynamic = t2 - t1
                time_trt_dynamic = t0 + time_trt_dynamic
            print("Inference time with TensorRT_dynamic = %.3f ms" % (time_trt_dynamic * 1000))
            return time_trt_dynamic, trt_outputs_dynamic


def inference_with_pytorch(model_path, data_path):
    net = parsingNet(pretrained=False, cls_dim=(100 + 1, 56, 4),
                     use_aux=False).cuda()

    state_dict = torch.load(model_path, map_location='cpu')['model']
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    time_torch = 0
    for i in range(10):
        path = data_path
        frame = cv2.imread(path)
        img = preprocessing(frame).cuda()
        img = img.unsqueeze(0)
        t3 = time.time()
        with torch.no_grad():
            torch_outputs = net(img)
        torch_outputs[0].data.cpu().numpy()
        t4 = time.time()
        time_torch = t4 - t3
        time_torch = time_torch + time_torch
    print("Inference time with PyTorch = %.3f ms" % (time_torch * 1000))
    return time_torch, torch_outputs


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
    # engine = build_engine_int8(onnx_path, 16, int8_data_path, verbose=True)

    # do inference
    # inference_with_trt(trt_file_path, data_path, dynamic_input)
