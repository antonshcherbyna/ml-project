import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pdb import set_trace as breakpoint

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        '''''num_pool5_feats = 256
        fc_block = nn.Sequential(
            Flatten(),
            nn.Linear(num_pool5_feats, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6, bias=False),
        )'''''
#        classifier = nn.Sequential(
#            nn.Linear(4096, num_classes),
#        )

        self._feature_blocks = nn.ModuleList([
            conv1,
            pool1,
            conv2,
            pool2,
            conv3,
            conv4,
            conv5,
            pool5
            #fc_block
        ])
        self.all_feat_names = [
            'conv1',
            'pool1',
            'conv2',
            'pool2',
            'conv3',
            'conv4',
            'conv5',
            'pool5'
            #'fc_block'
        ]
        assert(len(self.all_feat_names) == len(self._feature_blocks))

    def _parse_out_keys_arg(self, out_feat_keys):

        # By default return the features of the last layer / module.
        out_feat_keys = [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys

        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')
        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError('Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
            elif key in out_feat_keys[:f]:
                raise ValueError('Duplicate output feature key: {0}.'.format(key))

        # Find the highest output feature in `out_feat_keys
        max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat

    def forward(self, x, out_feat_keys=None):
        """Forward an image `x` through the network and return the asked output features.

        Args:
          x: input image.
          out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. By default the last feature of
                the network is returned.

        Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
        """
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)

        feat = x
        for f in range(max_out_feat+1):
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats)==1 else out_feats
        return out_feats

    def get_L1filters(self):
        convlayer = self._feature_blocks[0][0]
        batchnorm = self._feature_blocks[0][1]
        filters = convlayer.weight.data
        scalars = (batchnorm.weight.data / torch.sqrt(batchnorm.running_var + 1e-05))
        filters = (filters * scalars.view(-1, 1, 1, 1).expand_as(filters)).cpu().clone()

        return filters

#def create_model(opt):
#    return AlexNet(opt)

class Regressor(nn.Module):
    def __init__(self, num_classes, indim=256):
        super(Regressor, self).__init__()
        num_classes = num_classes
        self.Alexnet = AlexNet()

        #self.fc = nn.Linear(indim, num_classes)
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(2*indim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fin = m.in_features
                fout = m.out_features
                std_val = np.sqrt(2.0/fout)
                m.weight.data.normal_(0.0, std_val)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x1, x2, out_feat_keys=None):
        x1 = self.Alexnet(x1, out_feat_keys)
        x2 = self.Alexnet(x2, out_feat_keys)

        if out_feat_keys==None:
            x = torch.cat((x1,x2), dim=1)
            return x1, x2, self.fc(x)
        else:
            return x1, x2

def create_model(opt):
    return Regressor(opt)

if __name__ == '__main__':
    size = 32
    opt = {'num_classes': 4}

    net = create_model(6)
    x = torch.autograd.Variable(torch.FloatTensor(2, 3, size, size).uniform_(-1,1))

    out = net.Alexnet(x, out_feat_keys=net.Alexnet.all_feat_names)
    for f in range(len(out)):
        print('Output feature {0} - size {1}'.format(
            net.Alexnet.all_feat_names[f], out[f].size()))

    filters = net.Alexnet.get_L1filters()

    print('First layer filter shape: {0}'.format(filters.size()))
