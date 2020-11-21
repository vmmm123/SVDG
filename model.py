import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import torch.nn.init as init


class classifier(nn.Module):
    def __init__(self, num_classes=5):
        super(classifier, self).__init__()

        self.num_classes = num_classes
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout())]))


    def forward(self, x):
        x = self.classifier(x)
        return x


class features(nn.Module):
    def __init__(self):
        super(features, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))




    def forward(self, x):
        x = self.features(x * 57.6)
        features = x.view(x.size(0),-1)
        return features


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd, reverse=True):
        ctx.lambd = lambd
        ctx.reverse = reverse
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.lambd), None, None
        else:
            return (grad_output * ctx.lambd), None, None


def grad_reverse(x, lambd=1.0, reverse=True):
    return GradReverse.apply(x, lambd, reverse)


class Discriminator(nn.Module):
    def __init__(self, dims, grl=True, reverse=True):
        if len(dims) != 4:
            raise ValueError("Discriminator input dims should be three dim!")
        super(Discriminator, self).__init__()
        self.grl = grl
        self.reverse = reverse
        self.model = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dims[2], dims[3]),
        )
        self.lambd = 1.0

    def set_lambd(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        if self.grl:
            x = grad_reverse(x, self.lambd, self.reverse)
        x = self.model(x)
        return x


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Network(nn.Module):
    def __init__(self,num_classes=5,num_domains=3,grl=1,reverse=True,num_features=128):
        super(Network, self).__init__()
        self.features=features()
        self.classifier = classifier()
        self.class_classifier = nn.Linear(4096, num_classes)
        self.projection_original_features = nn.Sequential(nn.Linear(256 * 6 * 6, num_features),
            Normalize(2))
        self.discriminator = Discriminator([4096, 1024, 1024, num_domains], grl=grl, reverse=reverse)


    def return_reduced_image_features(self, original):
        features = self.features(original)
        reduced_features = self.projection_original_features(features)
        return features,reduced_features


    def forward(self, images=None, images2=None, mode=1):
        '''
        mode 0: get  features for initialize memory bank,
        mode 1: classification for validation and training

        '''

        if mode == 0:
            return self.return_reduced_image_features(images)
        if mode == 1:
            if images2 is None:
                cls = self.features(images)
                cls = self.classifier(cls)
                cls = self.class_classifier(cls)
                return cls
            else:
                image_features, image_reduced_features = self.return_reduced_image_features(images)
                _, images2_reduced_features = self.return_reduced_image_features(images2)
                cls = self.classifier(image_features)
                ouput_class=self.class_classifier(cls)
                output_domain = self.discriminator(cls)
                return image_reduced_features,images2_reduced_features, ouput_class,output_domain
                
                