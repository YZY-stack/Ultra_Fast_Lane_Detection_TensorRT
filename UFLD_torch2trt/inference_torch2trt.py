from torch2trt import TRTModule
import cv2
import torch
import torchvision.transforms as transforms
import time
from PIL import Image


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


def inference_with_torch2trt(trt_file_path, data_path):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(trt_file_path))

    time_torch = 0
    for i in range(10):
        path = data_path
        frame = cv2.imread(path)
        img = preprocessing(frame).cuda()
        img = img.unsqueeze(0)
        t3 = time.time()
        with torch.no_grad():
            torch_outputs = model_trt(img)
        torch_outputs[0].data.cpu().numpy()
        t4 = time.time()
        time_torch = t4 - t3
        time_torch = time_torch + time_torch
    print("Inference time with torch2trt = %.3f ms" % (time_torch * 1000))
    return time_torch, torch_outputs
