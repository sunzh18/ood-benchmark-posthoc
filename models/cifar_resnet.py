
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from models.route import *

normalization = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    
class Identity(nn.Module):
    def forward(self, input):
        return input + 0.0

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = normalization(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = normalization(planes)
        self.shortcut = Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.shortcut(out)
        out = self.relu(out)

        return out

    def forward_masked(self, x, mask_weight=None, mask_bias=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.shortcut(out)
        out = self.relu(out)

        if mask_weight is not None:
            out = out * mask_weight[None,:,None,None]
        if mask_bias is not None:
            out = out + mask_bias[None,:,None,None]
        return out

    def forward_threshold(self, x, threshold=1e10):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        b, c, w, h = out.shape
        mask = out.view(b, c, -1).mean(2) < threshold
        out = mask[:, :, None, None] * out
        # print(mask.sum(1).float().mean(0))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normalization(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = normalization(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = normalization(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.shortcut = Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.shortcut(out)
        out = self.relu(out)

        return out


class AbstractResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(AbstractResNet, self).__init__()
        self.gradients = []
        self.activations = []
        self.handles_list = []
        self.integrad_handles_list = []
        self.integrad_scores = []
        self.integrad_calc_activations_mask = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pruned_activations_mask = []
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.bn1 = normalization(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _initial_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                normalization(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _forward(self, x):
        self.activations = []
        self.gradients = []
        self.zero_grad()
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)

        if strict:
            error_msg = ''
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            print('Warning(s) in loading state_dict for {}:\n\t{}'.format(self.__class__.__name__, "\n\t".join(error_msgs)))
        # LINE 
    def remove_handles(self):
        for handle in self.handles_list:
            handle.remove()
        self.handles_list.clear()
        self.activations = []
        self.gradients = []
    # LINE 
    def _compute_taylor_scores(self, inputs, labels):
        self._hook_layers()
        outputs = self._forward(inputs)
        outputs[0, labels.item()].backward(retain_graph=True)

        first_order_taylor_scores = []
        self.gradients.reverse()

        for i, layer in enumerate(self.activations):
            first_order_taylor_scores.append(torch.mul(layer, self.gradients[i]))
                
        self.remove_handles()
        return first_order_taylor_scores, outputs
    # LINE 
    def _hook_layers(self):
        def backward_hook_relu(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].to(self.device))

        def forward_hook_relu(module, input, output):
            # mask output by pruned_activations_mask
            # In the first model(input) call, the pruned_activations_mask
            # is not yet defined, thus we check for emptiness
            if self.pruned_activations_mask:
              output = torch.mul(output, self.pruned_activations_mask[len(self.activations)].to(self.device)) #+ self.pruning_biases[len(self.activations)].to(self.device)
            self.activations.append(output.to(self.device))
            return output

        for module in self.modules():
            if isinstance(module, nn.AvgPool2d):
                self.handles_list.append(module.register_forward_hook(forward_hook_relu))
                self.handles_list.append(module.register_backward_hook(backward_hook_relu))

class ResNetCifar(AbstractResNet):
    def __init__(self, block, layers, num_classes=10, method='', p=None, p_w=None, p_a=None, info=None, clip_threshold=1e10, LU = False):
        super(ResNetCifar, self).__init__(block, layers, num_classes)
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.method = method
        self.clip_threshold = clip_threshold
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        if p is None or info is None:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            if LU:
                # print('use LINE')
                self.fc = RouteLUNCH(512 * block.expansion, num_classes, p_w=p_w, p_a=p_a, info=info, clip_threshold = clip_threshold)
            else:
                # print('use dice')
                self.fc = RouteDICE(512 * block.expansion, num_classes, p=p, info=info)

        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self._initial_weight()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def features(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
    
    def forward_pool_feat(self, feat):
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return out

    def forward_features(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)  
        return feat
    
    def forward_head(self, feat):
        out = self.fc(feat)
        return out
    
    def forward_threshold_features(self, x, threshold=1e10):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.clip(max=threshold)
        feat = feat.view(feat.size(0), -1)  
        return feat

    def forward(self, x, fc_params=None):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return out

    def forward_LINE(self, x, threshold=1e10):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.clip(max=threshold)
        feat = feat.view(feat.size(0), -1)
        feature = feat
        out = self.fc(feat)
        return out, feature
    
    def forward_threshold(self, x, threshold=1e10):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.clip(max=threshold)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_list(self, x):
        out_list = []
        out = self.relu(self.bn1(self.conv1(x)))
        # out_list.append(out)
        out = self.layer1(out)
        # out_list.append(out)
        out = self.layer2(out)
        # out_list.append(out)
        out = self.layer3(out)
        # out_list.append(out)
        out = self.layer4(out)
        
        out_list.append(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        y = self.fc(out)
        return y, out_list

    def intermediate_forward(self, x, layer_index=4):
        if layer_index >= 0:
            # out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            out = self.relu(self.bn1(self.conv1(x)))
        if layer_index >= 1:
            out = self.layer1(out)
        if layer_index >= 2:
            out = self.layer2(out)
        if layer_index >= 3:
            out = self.layer3(out)
        if layer_index >= 4:
            out = self.layer4(out)
        # out = out.clip(max=1.0)
        return out





def resnet18_cifar(**kwargs):
    return ResNetCifar(BasicBlock, [2,2,2,2], **kwargs)

def resnet34_cifar(**kwargs):
    return ResNetCifar(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50_cifar(**kwargs):
    return ResNetCifar(Bottleneck, [3, 4, 6, 3], **kwargs)