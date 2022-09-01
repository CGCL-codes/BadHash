import math
import torch
import torch.nn as nn
import torchvision
from torchvision import models


# vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn, "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn}
# class VGG(nn.Module):
#     def __init__(self, model_name, bit):
#         super(VGG, self).__init__()
#         original_model = vgg_dict[model_name](pretrained=True)
#         self.features = original_model.features
#         self.cl1 = nn.Linear(25088, 4096)
#         self.cl1.weight = original_model.classifier[0].weight
#         self.cl1.bias = original_model.classifier[0].bias
#
#         cl2 = nn.Linear(4096, 4096)
#         cl2.weight = original_model.classifier[3].weight
#         cl2.bias = original_model.classifier[3].bias
#
#         self.classifier = nn.Sequential(
#             self.cl1,
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             cl2,
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, bit),
#         )
#
#         self.tanh = nn.Tanh()
#         self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
#         self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()
#
#     def forward(self, x, alpha=1):
#         # x = (x-self.mean)/self.std
#         f = self.features(x)
#         f = f.view(f.size(0), -1)
#         y = self.classifier(f)
#         y = self.tanh(alpha * y)
#         return y


# resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}
# class ResNet(nn.Module):
#   def __init__(self, model_name, hash_bit):
#     super(ResNet, self).__init__()
#     model_resnet = resnet_dict[model_name](pretrained=True)
#     self.conv1 = model_resnet.conv1
#     self.bn1 = model_resnet.bn1
#     self.relu = model_resnet.relu
#     self.maxpool = model_resnet.maxpool
#     self.layer1 = model_resnet.layer1
#     self.layer2 = model_resnet.layer2
#     self.layer3 = model_resnet.layer3
#     self.layer4 = model_resnet.layer4
#     self.avgpool = model_resnet.avgpool
#     self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
#                          self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
#
#     self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
#     self.hash_layer.weight.data.normal_(0, 0.01)
#     self.hash_layer.bias.data.fill_(0.0)
#
#     self.activation = nn.Tanh()
#     self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
#     self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()
#
#   def forward(self, x, alpha=1):
#     # x = (x-self.mean)/self.std
#     x = self.feature_layers(x)
#     x = x.view(x.size(0), -1)
#     y = self.hash_layer(x)
#     y = self.activation(alpha*y)
#     return y


class VGGFc(nn.Module):
  def __init__(self, name, hash_bit):
    super(VGGFc, self).__init__()
    model_vgg = vgg_dict[name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    # self.use_hashnet = use_hashnet
    self.hash_layer = nn.Linear(model_vgg.classifier[6].in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

    self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
    self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

  def forward(self, x):
    # x = (x-self.mean)/self.std
    if self.training:
        self.iter_num += 1
    x = self.features(x)
    x = x.view(x.size(0), 25088)
    x = self.classifier(x)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features


class ResNetFc(nn.Module):
  def __init__(self, name, hash_bit):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[name](pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features

resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}

class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet50"):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[res_model](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y

    def adv_forward(self, x, alpha=1):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.hash_layer(x)
        layer = nn.Tanh()
        y = layer(alpha * x)
        return y

    def forward_(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        # y = self.hash_layer(x)
        return x



vgg_dict = {"Vgg11": models.vgg11, "Vgg13": models.vgg13, "Vgg16": models.vgg16,
            "Vgg19": models.vgg19}


class Vgg(nn.Module):
    def __init__(self, hash_bit, vgg_model="Vgg16"):
        super(Vgg, self).__init__()
        model_vgg = vgg_dict[vgg_model](pretrained=True)
        self.features = model_vgg.features

        cl1 = nn.Linear(512 * 7 * 7, 4096)
        cl1.weight = model_vgg.classifier[0].weight
        cl1.bias = model_vgg.classifier[0].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_vgg.classifier[3].weight
        cl2.bias = model_vgg.classifier[3].bias

        self.hash_layer = nn.Sequential(
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.hash_layer(x)
        return x

    def adv_forward(self, x, alpha=1):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.hash_layer(x)
        layer = nn.Tanh()
        y = layer(alpha * x)
        return y


densenet_dict = {"Densenet121": models.densenet121, "Densenet169": models.densenet169, "Densenet201": models.densenet201,
            "Densenet161": models.densenet161}


class Densenet(nn.Module):
    def __init__(self, hash_bit, densenet_model="Densenet121"):
        super(Densenet, self).__init__()
        model_densenet = densenet_dict[densenet_model](pretrained=True)
        self.features = model_densenet.features
        # cl1 = nn.Linear(512 * 7 * 7, 4096)
        # cl1.weight = model_densenet.classifier[0].weight
        # cl1.bias = model_densenet.classifier[0].bias

        cl2 = nn.Linear(1024 * 7 * 7, 4096)
        cl2.weight = model_densenet.classifier.weight
        cl2.bias = model_densenet.classifier.bias

        self.hash_layer = nn.Sequential(
            # cl1,
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 1024 * 7 * 7)
        x = self.hash_layer(x)
        return x

