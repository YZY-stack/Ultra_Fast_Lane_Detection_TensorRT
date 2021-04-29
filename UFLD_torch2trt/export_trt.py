import torch
from torch2trt import torch2trt
from model.model import parsingNet

# create a pytorch model
model = parsingNet(pretrained=False, cls_dim=(100 + 1, 56, 4),
                     use_aux=False).cuda()
state_dict = torch.load("./tusimple_18.pth", map_location='cpu')['model']
model.load_state_dict(state_dict, strict=False)
model.eval()

# create an input data
x = torch.ones((1, 3, 288, 800)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])

y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(f'The shape of output using PyTorch: {y.shape}')
print(f'The shape of output using TensorRT: {y_trt.shape}')
print(f'The accuracy loss: {torch.max(torch.abs(y - y_trt))}')

# save model
torch.save(model_trt.state_dict(), 'UFLD_trt.pth')

print('Successfully convert pth to trt using torch2trt')

