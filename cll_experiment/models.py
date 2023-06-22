import torchvision
import torch.nn as nn

def get_resnet18(num_classes):
    resnet = torchvision.models.resnet18(weights=None)
    # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # resnet.maxpool = nn.Identity()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    return resnet

def get_modified_resnet18(num_classes):
    resnet = torchvision.models.resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    return resnet