import torch
from torch2trt import torch2trt
from model.model import parsingNet
from torch2trt import TRTModule

# create some regular pytorch model...
model = parsingNet(pretrained=False, cls_dim=(100 + 1, 56, 4),
                     use_aux=False).cuda()
state_dict = torch.load("/home/dji/Lane_fast/Ultra-Fast-Lane-Detection-ori/tusimple_18.pth", map_location='cpu')['model']
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v

model.load_state_dict(compatible_state_dict, strict=False)
model.eval()

# create example data
x = torch.ones((1, 3, 288, 800)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])

y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(y.shape)
print(y_trt.shape)
print(torch.max(torch.abs(y - y_trt)))

# save model
torch.save(model_trt.state_dict(), 'UFLD_trt.pth')
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('UFLD_trt.pth'))
