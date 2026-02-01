import torch
from model.adc_res_transxnet import ADCResTransXNet

model = ADCResTransXNet()
x = torch.randn(1, 3, 256, 256)

y = model(x)
print("Output shape:", y.shape)