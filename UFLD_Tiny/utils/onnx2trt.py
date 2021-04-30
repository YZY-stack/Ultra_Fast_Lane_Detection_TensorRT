import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit 
import laneDetection
import torch
import os

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def build_engine(onnx_file_path):
    G_LOGGER = trt.Logger()
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, G_LOGGER) as parser:
        builder.fp16_mode = True
        builder.max_batch_size = 1
        builder.max_workspace_size = 1 << 30
        print('Loading ONNX file from path {}...'.format(onnx_model_name))
        with open(onnx_model_name, 'rb') as model:
            print('Beginning ONNX file parsing')
            #b = parser.parse(model.read())
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print("num layers:",network.num_layers)

    network.get_input(0).shape = [1, 3, 288, 800]
    engine = builder.build_cuda_engine(network)
    print("engine:",engine)
    print("Completed creating Engine")
    with open(trt_model_name, "wb") as f:
        f.write(engine.serialize())
    return engine


# This function builds an engine from a Caffe model.
def build_engine_int8(onnx_file_path, calib):
    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = 1  # calib.get_batch_size()
        builder.max_workspace_size = 1 << 30
        builder.int8_mode = True
        builder.int8_calibrator = calib
        with open(onnx_file_path, 'rb') as model:
           parser.parse(model.read())   # , dtype=trt.float32
        return builder.build_cuda_engine(network)

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs


detector = laneDetection.laneDetection()
detector.setResolution(640, 480)
detector.setScaleFactor(4)
max_batch_size = 1
onnx_model_name = 'model.onnx'
TRT_LOGGER = trt.Logger()  # This logger is required to build an engine


img_np_nchw = torch.rand((1,3,288,800)).numpy()

# These two modes are dependent on hardwares
fp16_mode = False
int8_mode = False
trt_engine_path = './model_fp16_{}_int8_{}.trt'.format(fp16_mode, int8_mode)
# Build an engine
engine = build_engine(onnx_model_name)
print("OK")

# Create the context for this engine
context = engine.create_execution_context()
# Allocate buffers for input and output
inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

# Do inference
shape_of_output = (max_batch_size, 1000)
# Load data to the buffer
inputs[0].host = img_np_nchw.reshape(-1)

# inputs[1].host = ... for multiple input
t1 = time.time()
trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
t2 = time.time()
feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)

print('TensorRT ok')

model = detector.net
model = model.eval()

input_for_torch = torch.from_numpy(img_np_nchw).cuda()
t3 = time.time()
feat_2= model(input_for_torch)
t4 = time.time()
feat_2 = feat_2.cpu().data.numpy()
print('Pytorch ok!')


mse = np.mean((feat - feat_2)**2)
print("Inference time with the TensorRT engine: {}".format(t2-t1))
print("Inference time with the PyTorch model: {}".format(t4-t3))
print('MSE Error = {}'.format(mse))

print('All completed!')


