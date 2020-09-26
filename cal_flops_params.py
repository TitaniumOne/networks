from thop import profile
import torch
#from resnet_18 import Resnet_18, resnet18
from temporal_models import TemporalModel
model = TemporalModel(48,
                      8,
                      alpha=1,
                      beta=1,
                      model='TMP',
                      backbone='resnet50_tmp',
                      resi=False)
input = torch.randn(1, 3 * 8, 224, 224)
flops, params = profile(model, inputs=(input))
print(flops)
print(params)