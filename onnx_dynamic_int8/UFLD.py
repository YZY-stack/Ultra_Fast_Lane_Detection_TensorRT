import torch, os, cv2
from onnx_dynamic_int8.model.model import parsingNet
from onnx_dynamic_int8.utils.common import merge_config
from onnx_dynamic_int8.utils.dist_utils import dist_print
from onnx_dynamic_int8.configs.constant import tusimple_row_anchor
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import time
from torch.autograd import Variable
import torch.onnx as onnx


class laneDetection():
    def __init__(self):
        torch.backends.cudnn.benchmark = True
        self.args, self.cfg = merge_config() 
        self.cls_num_per_lane = 56
        self.row_anchor = tusimple_row_anchor
        self.net = parsingNet(pretrained = False, backbone=self.cfg.backbone, cls_dim = (self.cfg.griding_num+1, self.cls_num_per_lane, self.cfg.num_lanes), use_aux=False).cuda()

        state_dict = torch.load(self.cfg.test_model, map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        self.net.load_state_dict(compatible_state_dict, strict=False)
        self.net.eval()
 
        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.img_w = 960
        self.img_h = 480
        self.scale_factor = 1
        self.color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]  
        self.idx = np.arange(self.cfg.griding_num) + 1
        self.idx = self.idx.reshape(-1, 1, 1)      

        self.cpu_img = None
        self.gpu_img = None
        self.type = None
        self.gpu_output = None
        self.cpu_output = None
        
        col_sample = np.linspace(0, 800 - 1, self.cfg.griding_num)
        self.col_sample_w = col_sample[1] - col_sample[0]
        
    def setResolution(self, w, h):
        self.img_w = w
        self.img_h = h

    def getFrame(self, frame):
        self.cpu_img = frame

    def setScaleFactor(self, factor=1):
        self.scale_factor = factor

    def preprocess(self):
        tmp_img = cv2.cvtColor(self.cpu_img, cv2.COLOR_BGR2RGB)
        if self.scale_factor != 1:
            tmp_img = cv2.resize(tmp_img, (self.img_w//self.scale_factor, self.img_h//self.scale_factor))
        tmp_img = Image.fromarray(tmp_img)
        tmp_img = self.img_transforms(tmp_img)
        self.gpu_img = tmp_img.unsqueeze(0).cuda()

    def inference(self):
        self.gpu_output = self.net(self.gpu_img)

    def parseResults(self): 
        self.cpu_output = self.gpu_output[0].data.cpu().numpy()
        self.prob = scipy.special.softmax(self.cpu_output[:-1, :, :], axis=0)

        self.loc = np.sum(self.prob * self.idx, axis=0)
        self.cpu_output = np.argmax(self.cpu_output, axis=0)

        self.loc[self.cpu_output == self.cfg.griding_num] = 0
        #self.cpu_output = self.loc

        # import pdb; pdb.set_trace()
        vis = self.cpu_img
        for i in range(self.loc.shape[1]):
            if np.sum(self.loc[:, i] > 0) > 40:
                for k in range(self.loc.shape[0]):
                    if self.loc[k, i] > 0:
                        ppp = (int(self.loc[k, i] * self.col_sample_w * self.img_w / 800) - 1, int(self.img_h * (self.row_anchor[k]/288)) - 1 )
                        cv2.circle(vis,ppp,3, self.color[i], -1)
     
        cv2.imshow("output",vis)
        cv2.waitKey(1)
        return vis


