import torchvision.models
import torch
from thop import profile
from thop import clever_format
from manet import *
from time import time
from tqdm import tqdm
# import torchsummary
# model = torchvision.models.vgg16(pretrained=False)

model = HybridAttionNetwork('test')
device = torch.device('cuda')
model.to(device)
myinput = torch.zeros((1, 3, 384, 384)).to(device)
flops, params = profile(model.to(device), inputs=(myinput,))
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)
print('---------------------')

x = torch.randn(12,3,384,384).cuda()
total_t = 0
with torch.no_grad():
    for i in tqdm(range(100)):
        start = time()
        p,_,_,_ = model(x)
        # p = p + 1 # replace torch.cuda.synchronize()
        total_t += time() - start
# print(total_t)
print("FPS", 100 / total_t * 12)


# net = HybridAttionNetwork()
# device = torch.device('cuda')
# net.to(device)
# myinput = torch.zeros((1, 3, 384, 384)).to(device)
# flops, params = profile(net.to(device), inputs=(myinput,))
# flops, params = clever_format([flops, params], "%.3f")
# print(flops, params)
# print('---------------------')