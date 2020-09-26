import argparse
import time
import os
import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision
from sklearn.metrics import confusion_matrix
from transforms import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torch.nn as nn
import matplotlib.pyplot as plt
import resnet
from temporal_models import TemporalModel
import datasets_video
# options
parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('dataset',
                    type=str,
                    choices=['ucf101', 'something-v1', 'diving48'])
parser.add_argument('modality', type=str, choices=['RGB', 'flow', 'RGBDiff'])
parser.add_argument('weights', type=str)
parser.add_argument('--alpha',
                    type=int,
                    default=4,
                    help='spatial temporal split for output channels')
parser.add_argument(
    '--beta',
    type=int,
    default=2,
    choices=[1, 2],
    help='channel splits for input channels, 1 for GST-Large and 2 for GST')
parser.add_argument('--resi',
                    dest='resi',
                    action='store_true',
                    help='open residual connection or not',
                    default=False)
parser.add_argument('--num_segments', type=int, default=64)
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()
writer = SummaryWriter()
dummy_s1 = torch.rand(1)
writer.add_scalar('data/scalar1', dummy_s1[0], 0)

if args.dataset == 'something-v1':
    num_class = 174
elif args.dataset == 'diving48':
    num_class = 48
elif args.dataset == 'ucf101':
    num_class = 101
else:
    raise ValueError('Unknown dataset ' + args.dataset)

if 'something' in args.dataset:
    # label transformation for left/right categories
    target_transforms = {86: 87, 87: 86, 93: 94, 94: 93, 166: 167, 167: 166}
    print('Target transformation is enabled....')
else:
    target_transforms = None
args.train_list, args.val_list, args.root_path, args.rgb_prefix = datasets_video.return_dataset(
    args.dataset)

if args.modality == 'RGB':
    if 'gst' in args.arch:
        model = TemporalModel(num_class,
                              args.num_segments,
                              model='GST',
                              backbone=args.arch,
                              alpha=args.alpha,
                              beta=args.beta,
                              dropout=args.dropout,
                              target_transforms=target_transforms,
                              resi=args.resi)
    elif 'stm' in args.arch:
        model = TemporalModel(num_class,
                              args.num_segments,
                              model='STM',
                              backbone=args.arch,
                              alpha=args.alpha,
                              beta=args.beta,
                              dropout=args.dropout,
                              target_transforms=target_transforms,
                              resi=args.resi)
    elif 'tmp' in args.arch:
        model = TemporalModel(num_class,
                              args.num_segments,
                              model='TMP',
                              backbone=args.arch,
                              alpha=args.alpha,
                              beta=args.beta,
                              dropout=args.dropout,
                              target_transforms=target_transforms,
                              resi=args.resi)
    elif 'ori' in args.arch:
        model = TemporalModel(num_class,
                              args.num_segments,
                              model='ORI',
                              backbone=args.arch,
                              alpha=args.alpha,
                              beta=args.beta,
                              dropout=args.dropout,
                              target_transforms=target_transforms,
                              resi=args.resi)
    else:
        model = TemporalModel(num_class,
                              args.num_segments,
                              model='ORI',
                              backbone=args.arch + '_ori',
                              alpha=args.alpha,
                              beta=args.beta,
                              dropout=args.dropout,
                              target_transforms=target_transforms,
                              resi=args.resi)
    if os.path.isfile(args.weights):
        print(("=> loading checkpoint '{}'".format(args.weights)))
        checkpoint = torch.load(args.weights)
        original_checkpoint = checkpoint['state_dict']
        best_prec1 = checkpoint['best_prec1']
        pretrained_dict = {k[7:]: v for k, v in original_checkpoint.items()}
        model.load_state_dict(pretrained_dict)
        print(("=> loaded checkpoint '{}' (epoch {} ) best_prec1 : {} ".format(
            args.weights, checkpoint['epoch'], best_prec1)))
    else:
        print(("=> no checkpoint found at '{}'".format(args.weights)))

#writer.add_graph(model, torch.rand([1,3,64,224,224]))
'''
img_grid = vutils.make_grid(x, normalize=True, scale_each=True, nrow=6)
# 绘制原始图像
writer.add_image('raw img', img_grid, global_step=666)  # j 表示feature map数
print(x.size())
'''


class FeatureExtractor(nn.Module):
    def __init__(self, submodule):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule

    def forward(self, x):
        outputs = []
        for i, (name, param) in enumerate(self.submodule.named_parameters()):
            if '.conv3d' in name and 'weight' in name:
                '''
                tmp = layer.conv3d(x)
                print(tmp.size())
                in_channels = layer.in_features
                kernel_grid = vutils.make_grid(torch.squeeze(tmp), normalize=True, scale_each=True, nrow=8)
                writer.add_image(f'{name}_all', kernel_grid, global_step=0) 
                '''
                print('name : %s paramsize: %s' % (name, param.size()))
                in_channels = param.size()[1]
                out_channels = param.size()[0]  # 输出通道，表示卷积核的个数
                k_c, k_w, k_h = param.size()[2], param.size()[4], param.size(
                )[3]
                kernel_all = param[0].view(-1, 1, k_w, k_h)  # 每个通道的卷积核
                kernel_grid = vutils.make_grid(kernel_all,
                                               normalize=True,
                                               scale_each=True,
                                               nrow=in_channels)
                writer.add_image(f'{name}', kernel_grid, global_step=0)
        return 1


model.eval()
#x= torch.rand(1,3,64,224, 224)

path = '/share/something_v1/20bn-something-something-v1/42728/'  #36805
x = Image.open(path + args.rgb_prefix.format(1)).convert('RGB').resize(
    (256, 256)).crop((16, 16, 240, 240))
x = torch.from_numpy(np.asarray(x)).permute(2, 0, 1).contiguous()
x = torch.unsqueeze(x.float().div(255), 1)
for i in range(5, 33, 4):
    img = Image.open(path + args.rgb_prefix.format(i)).convert('RGB').resize(
        (256, 256)).crop((16, 16, 240, 240))
    img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).contiguous()
    img = torch.unsqueeze(img.float().div(255), 1)
    x = torch.cat((x, img), 1)
x = torch.unsqueeze(x, 0)
print('input image size %s' % str(x.size()))
'''
for name, module in model.base_model.named_children():
    print(name)
    if 'layer4' in name :
        for name1, module1 in module.named_children():
            if '0' in name1:
                for name2, module2 in module1.named_children():
                    print(name2)
'''

x = model.base_model.conv1(x)
x1 = x
print((x1.size()))
x1 = x1[0][0].reshape(-1, 1,
                      x1.size()[3],
                      x1.size()[4])  # C，B, H, W  ---> B，C, H, W
img_grid = vutils.make_grid(x1, normalize=True, scale_each=True,
                            nrow=4)  # normalize进行归一化处理
writer.add_image(f'conv1_feature_maps', img_grid, global_step=0)
print('name: %s size: %s' % ('conv1', x1.size()))

x = model.base_model.bn1(x)
x = model.base_model.relu(x)
x = model.base_model.maxpool(x)

for name1, module1 in model.base_model.layer1.named_children():
    if '0' == name1:  #use for resnet
        for name2, module2 in module1.named_children():
            if 'conv1' in name2:
                x1 = module2(x)
                x1 = x1[0][0].reshape(
                    -1, 1,
                    x1.size()[3],
                    x1.size()[4])  # C，B, H, W  ---> B，C, H, W
                img_grid = vutils.make_grid(x1,
                                            normalize=True,
                                            scale_each=True,
                                            nrow=4)  # normalize进行归一化处理
                writer.add_image(f'layer1_{name1}{name2}_feature_maps',
                                 img_grid,
                                 global_step=0)
                print('name2: %s size: %s' %
                      (['layer1', name1, name2], x1.size()))

x = model.base_model.layer1(x)

for name1, module1 in model.base_model.layer2.named_children():
    if '0' == name1:  #use for resnet
        for name2, module2 in module1.named_children():
            if 'conv1' in name2:
                x1 = module2(x)
                x1 = x1[0][0].reshape(
                    -1, 1,
                    x1.size()[3],
                    x1.size()[4])  # C，B, H, W  ---> B，C, H, W
                img_grid = vutils.make_grid(x1,
                                            normalize=True,
                                            scale_each=True,
                                            nrow=4)  # normalize进行归一化处理
                writer.add_image(f'layer2_{name1}{name2}_feature_maps',
                                 img_grid,
                                 global_step=0)
                print('name2: %s size: %s' %
                      (['layer2', name1, name2], x1.size()))

x = model.base_model.layer2(x)
if args.resi:
    x = model.base_model.lateral_connection(x, model.base_model.lateral_layer3,
                                            model.base_model.backward_layer3)

for name1, module1 in model.base_model.layer3.named_children():
    if '0' == name1:  #use for resnet
        for name2, module2 in module1.named_children():
            if 'conv1' in name2:
                x1 = module2(x)
                x1 = x1[0][:512:16].reshape(
                    -1, 1,
                    x1.size()[3],
                    x1.size()[4])  # C，B, H, W  ---> B，C, H, W
                img_grid = vutils.make_grid(x1,
                                            normalize=True,
                                            scale_each=True,
                                            nrow=4)  # normalize进行归一化处理
                writer.add_image(f'layer3_{name1}{name2}_feature_maps',
                                 img_grid,
                                 global_step=0)
                print('name2: %s size: %s' %
                      (['layer3', name1, name2], x1.size()))

x = model.base_model.layer3(x)
if args.resi:
    x = model.base_model.lateral_connection(x, model.base_model.lateral_layer4,
                                            model.base_model.backward_layer4)

for name1, module1 in model.base_model.layer4.named_children():
    if '0' == name1:  #use for resnet
        for name2, module2 in module1.named_children():
            if 'conv1' in name2:
                x1 = module2(x)
                x1 = x1[0][:1024:32].reshape(
                    -1, 1,
                    x1.size()[3],
                    x1.size()[4])  # C，B, H, W  ---> B，C, H, W
                img_grid = vutils.make_grid(x1,
                                            normalize=True,
                                            scale_each=True,
                                            nrow=4)  # normalize进行归一化处理
                writer.add_image(f'layer4_{name1}{name2}_feature_maps',
                                 img_grid,
                                 global_step=0)
                print('name2: %s size: %s' %
                      (['layer4', name1, name2], x1.size()))

x = model.base_model.layer4(x)
x = x.transpose(1, 2).contiguous()
x = x.view((-1, ) + x.size()[2:])

x = model.base_model.avgpool(x)
x = x.view(x.size(0), -1)
x = model.base_model.fc(x)
'''
input = torch.rand(1,3,64,224, 224)
myexactor = FeatureExtractor(i3d_model)
x=myexactor(input)
'''
'''
print('starting to prepare embedding')
labels=[]
with open(args.val_list, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 
        if not lines:
            break
        pass
        labels.append(int(lines.split()[-1]))
features=np.load("features_res101_openpose_crop_val_0.npy")
label = np.asarray(labels).reshape(-1,1)
print('features shape: %s'%(str(features.shape)))
print('label shape: %s'%(str(label.shape)))
writer.add_embedding(torch.from_numpy(features), metadata=torch.from_numpy(label))#, label_img=torch.from_numpy(np.expand_dims(features,1)))
'''
writer.close()
