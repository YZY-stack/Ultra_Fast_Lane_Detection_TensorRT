#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from itertools import chain
import argparse
import os
from functools import reduce
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import tensorrt as trt
import torch

try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30


def add_help(description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()


def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[]):
    '''
    Parses sample arguments.

    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.

    Returns:
        str: Path of data directory.
    '''

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", help="Location of the TensorRT sample data directory, and any additional data directories.", action="append", default=[kDEFAULT_DATA_ROOT])
    args, _ = parser.parse_known_args()

    def get_data_path(data_dir):
        # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
        data_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(data_path):
            print("WARNING: " + data_path + " does not exist. Trying " + data_dir + " instead.")
            data_path = data_dir
        # Make sure data directory exists.
        if not (os.path.exists(data_path)):
            print("WARNING: {:} does not exist. Please provide the correct data path with the -d option.".format(data_path))
        return data_path

    data_paths = [get_data_path(data_dir) for data_dir in args.datadir]
    return data_paths, locate_files(data_paths, find_files)

def locate_files(data_paths, filenames):
    """
    Locates the specified files in the specified data directories.
    If a file exists in multiple data directories, the first directory is used.

    Args:
        data_paths (List[str]): The data directories.
        filename (List[str]): The names of the files to find.

    Returns:
        List[str]: The absolute paths of the files.

    Raises:
        FileNotFoundError if a file could not be located.
    """
    found_files = [None] * len(filenames)
    for data_path in data_paths:
        # Find all requested files.
        for index, (found, filename) in enumerate(zip(found_files, filenames)):
            if not found:
                file_path = os.path.abspath(os.path.join(data_path, filename))
                if os.path.exists(file_path):
                    found_files[index] = file_path

    # Check that all files were found
    for f, filename in zip(found_files, filenames):
        if not f or not os.path.exists(f):
            raise FileNotFoundError("Could not find {:}. Searched in data paths: {:}".format(filename, data_paths))
    return found_files

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
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

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_static(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_dynamic(context, bindings, inh, ind, outh, outd, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(ind, inh, stream)
    # Run inference.
    context.execute_async_v2([int(ind), int(outd)], stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(outh, outd, stream)
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return outh


class TrtLite:
    def __init__(self, build_engine_proc=None, build_engine_params=None, engine_file_path=None):
        logger = trt.Logger(trt.Logger.INFO)
        if engine_file_path is None:
            with trt.Builder(logger) as builder:
                if build_engine_params is not None:
                    self.engine = build_engine_proc(builder, *build_engine_params)
                else:
                    self.engine = build_engine_proc(builder)
        else:
            with open(engine_file_path, 'rb') as f, trt.Runtime(logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def __del__(self):
        self.engine = None
        self.context = None

    def save_to_file(self, engine_file_path):
        with open(engine_file_path, 'wb') as f:
            f.write(self.engine.serialize())

    def get_io_info(self, input_desc):
        def to_numpy_dtype(trt_dtype):
            tb = {
                trt.DataType.BOOL: np.dtype('bool'),
                trt.DataType.FLOAT: np.dtype('float32'),
                trt.DataType.HALF: np.dtype('float16'),
                trt.DataType.INT32: np.dtype('int32'),
                trt.DataType.INT8: np.dtype('int8'),
            }
            return tb[trt_dtype]

        if isinstance(input_desc, dict):
            if self.engine.has_implicit_batch_dimension:
                print('Engine was built with static-shaped input so you should provide batch_size instead of i2shape')
                return
            i2shape = input_desc
            for i, shape in i2shape.items():
                self.context.set_binding_shape(i, shape)
            return [(self.engine.get_binding_name(i), self.engine.binding_is_input(i),
                     tuple(self.context.get_binding_shape(i)), to_numpy_dtype(self.engine.get_binding_dtype(i))) for i
                    in range(self.engine.num_bindings)]

        batch_size = input_desc
        return [(self.engine.get_binding_name(i),
                 self.engine.binding_is_input(i),
                 (batch_size,) + tuple(self.context.get_binding_shape(i)),
                 to_numpy_dtype(self.engine.get_binding_dtype(i))) for i in range(self.engine.num_bindings)]

    def allocate_io_buffers(self, input_desc, on_gpu):
        io_info = self.get_io_info(input_desc)
        if io_info is None:
            return
        if on_gpu:
            cuda = torch.device('cuda')
            np2pth = {
                np.dtype('bool'): torch.bool,
                np.dtype('float32'): torch.float32,
                np.dtype('float16'): torch.float16,
                np.dtype('int32'): torch.int32,
                np.dtype('int8'): torch.int8,
            }
            return [torch.empty(i[2], dtype=np2pth[i[3]], device=cuda) for i in io_info]
        else:
            return [np.zeros(i[2], i[3]) for i in io_info]

    def execute(self, bindings, input_desc, stream_handle=0, input_consumed=None):
        if isinstance(input_desc, dict):
            i2shape = input_desc
            for i, shape in i2shape.items():
                self.context.set_binding_shape(i, shape)
            self.context.execute_async_v2(bindings, stream_handle, input_consumed)
            return

        batch_size = input_desc
        self.context.execute_async(batch_size, bindings, stream_handle, input_consumed)

    def print_info(self):
        print("Batch dimension is", "implicit" if self.engine.has_implicit_batch_dimension else "explicit")
        for i in range(self.engine.num_bindings):
            print("input" if self.engine.binding_is_input(i) else "output",
                  self.engine.get_binding_name(i), self.engine.get_binding_dtype(i),
                  self.engine.get_binding_shape(i),
                  -1 if -1 in self.engine.get_binding_shape(i) else reduce(
                      lambda x, y: x * y, self.engine.get_binding_shape(i)) * self.engine.get_binding_dtype(i).itemsize)