from torch import nn
import resnet
from basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal_, constant_


class TSN(nn.Module):
    def __init__(self,
                 num_class,
                 num_segments,
                 modality,
                 base_model='resnet50',
                 new_length=None,
                 consensus_type='avg',
                 before_softmax=True,
                 dropout=0.8,
                 partial_bn=True,
                 target_transform=None):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.consensus_type = consensus_type
        self.target_transform = target_transform
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments,
                   self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model, num_class)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model,
                              self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(
                getattr(self.base_model,
                        self.base_model.last_layer_name).weight, 0, std)
            constant_(
                getattr(self.base_model, self.base_model.last_layer_name).bias,
                0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model, num_class):

        if 'resnet' in base_model:
            self.base_model = getattr(resnet, base_model)(num_classes=1000)
            model_dict = self.base_model.state_dict()
            checkpoint = torch.load(
                '/home/hulianyu/.torch/models/resnet50-19c8e357.pth')
            model_dict.update(checkpoint)
            self.base_model.load_state_dict(model_dict)
            self.base_model.last_layer_name = 'fc'
            self.base_model.fc = nn.Linear(512 * 4, num_class)
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            if self.modality == 'flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]
            if self.modality == 'flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(
            input.view((-1, sample_len) + input.size()[-2:]))

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) +
                                     base_out.size()[1:])

        output = self.consensus(base_out)
        return output.squeeze(1)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([
                GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                GroupRandomHorizontalFlip(
                    is_flow=False, target_transform=self.target_transform)
            ])
        elif self.modality == 'flow':
            return torchvision.transforms.Compose([
                GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                GroupRandomHorizontalFlip(is_flow=True)
            ])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([
                GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                GroupRandomHorizontalFlip(is_flow=False)
            ])
