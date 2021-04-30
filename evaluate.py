import argparse
import numpy as np
from UFLD_torch2trt.inference_torch2trt import inference_with_torch2trt
from onnx_dynamic_int8.onnx_parser import inference_with_trt, inference_with_pytorch


def compare(time_torch, time_trt, out_torch, out_trt):
    print(f'Speedup: {time_torch / time_trt}')
    Average_diff = np.abs(out_torch.cpu() - out_trt[0].reshape(101, 56, 4)) / np.abs(out_torch.cpu())
    print(Average_diff)
    # print("Acc_Loss(Order of magnitude):", np.sum(Average_diff) / len(Average_diff))


def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pth_path', type=str, default='./tusimple_18.pth')
    parser.add_argument(
        '--trt_path', type=str, default='./lane_fp16_static.trt')
    parser.add_argument(
        '--torch2trt_path', type=str, default='UFLD_torch2trt/UFLD_trt.pth')
    parser.add_argument(
        '--data_path', type=str, default='./5.jpg')
    parser.add_argument(
        '--dynamic', action='store_true')
    parser.add_argument(
        '--torch2trt', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # paths and configs
    configs = set_config()
    pth_path = configs.pth_path
    trt_path = configs.trt_path
    torch2trt_path = configs.torch2trt_path
    dynamic_input = configs.dynamic
    torch2trt = configs.torch2trt
    data_path = configs.data_path

    # do inference
    if torch2trt:
        torch_time, torch_out = inference_with_pytorch(pth_path, data_path)
        torch2trt_time, torch2trt_out = inference_with_torch2trt(torch2trt_path, data_path)
        compare(torch_time, torch2trt_time, torch_out, torch2trt_out)
    else:
        torch_time, torch_out = inference_with_pytorch(pth_path, data_path)
        trt_time, trt_out = inference_with_trt(trt_path, data_path, dynamic_input)
        compare(torch_time, trt_time, torch_out, trt_out)

    print("finished!")

