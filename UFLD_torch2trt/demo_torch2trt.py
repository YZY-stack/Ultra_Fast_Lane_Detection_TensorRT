from torch2trt import TRTModule
import cv2
import torch
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import time
from PIL import Image
from data.constant import tusimple_row_anchor


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


def demo_with_torch2trt(trt_file_path, data_root):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(trt_file_path))
    row_anchor = tusimple_row_anchor
    img_w, img_h = 1280, 720

    img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    for i in range(10):
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        img_ori = cv2.imread(data_root)
        img = preprocessing(img_ori)
        img = img.unsqueeze(0)
        img = img.cuda()

        t1 = time.time()
        with torch.no_grad():
            out = model_trt(img)

        col_sample = np.linspace(0, 800 - 1, 100)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        t2 = time.time()
        print("Inference time = %.3f ms" % ((t2 - t1)*1000))
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(100) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == 100] = 0
        out_j = loc

        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                               int(img_h * (row_anchor[56 - 1 - k] / 288)) - 1)
                        cv2.circle(img_ori, ppp, img_w // 300, (0, 255, 0), 2)
        cv2.imshow("result", img_ori)
        cv2.imwrite("demo_using_torch2trt.jpg", img_ori)
    cv2.destroyAllWindows()


def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trt_path', type=str, default='lane_fp16_static.trt')
    parser.add_argument(
        '--data_path', type=str, default='/home/stevenyan/TRT/Inference/5.jpg')
    args = parser.parse_args()
    return args

    
if __name__ == "__main__":
    configs = set_config()
    trt_path = configs.trt_path
    data_path = configs.data_path

    demo_with_torch2trt(trt_path, data_path)
