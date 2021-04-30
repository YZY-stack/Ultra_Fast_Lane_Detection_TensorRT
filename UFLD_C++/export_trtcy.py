import torch
import struct
from model.model import parsingNet

def export(path):                               # 将权重保存

    file = open(path, 'w')
    file.write('{}\n'.format(len(state_dict.keys())))
    for k, v in state_dict.items():
        vr = v.reshape(-1).cpu().numpy()
        file.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            file.write(' ')
            file.write(struct.pack('>f',float(vv)).hex())
        file.write('\n')

if __name__ == "__main__":

    # 初始化网络
    model = parsingNet(pretrained=False, backbone='18', cls_dim=(101, 56, 4), use_aux=False).cuda()
    device = 'cpu'

    # 载入模型
    state_dict = torch.load('../tusimple_18.pth', map_location='cpu')['model']
    model.to(device).eval() # 转为推理模式

    path = 'weights/lane.trtcy'
    export(path)
