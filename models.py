import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Literal

# VGG16 series
class VGG16(nn.Module):
    def __init__(self, input_size: tuple[int, int, int], num_classes: int) -> None:
        """
        Args:
            input_size (tuple): input size of the image (C, H, W)
            num_classes (int): number of classes
        """
        super(VGG16, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * (input_size[1] // 32) * (input_size[2] // 32), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DualVGG16(nn.Module):
    def __init__(self, input_size: tuple[int, int, int], num_classes: int, merge_at: str="late") -> None:
        """
        Args:
            input_size (tuple): input size of the image (C, H, W)
            num_classes (int): number of classes
            merge_at (str): merge the two VGG16 networks early or late (default: "late")
        """
        super(DualVGG16, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.merge_at = merge_at

        self.branch_1_conv_block = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.branch_2_conv_block = nn.Sequential(
                
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
    
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
    
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
    
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
    
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.conv_block =nn.Sequential(
            # Channel swap to 6=2*3
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        if self.merge_at == "early":
            factor = 1
        else:
            factor = 2
        self.fc = nn.Sequential(
            nn.Linear(512 * (input_size[1] // 32) * (input_size[2] // 32) * factor, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x1, x2):
        if self.merge_at == "early":
            x = torch.cat([x1, x2], dim=1)
            x = self.conv_block(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        elif self.merge_at == "late":
            x1 = self.branch_1_conv_block(x1)
            x2 = self.branch_2_conv_block(x2)
            x = torch.cat([x1, x2], dim=1)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


# VGG11 series
class VGG11(nn.Module):
    def __init__(self, input_size: tuple[int, int, int], num_classes: int) -> None:
        """
        Args:
            input_size (tuple): input size of the image (C, H, W)
            num_classes (int): number of classes
        """
        super(VGG11, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * (input_size[1] // 32) * (input_size[2] // 32), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DualVGG11(nn.Module):

    def __init__(self, input_size: tuple[int, int, int], num_classes: int, merge_at: str="late") -> None:
        """
        Args:
            input_size (tuple): input size of the image (C, H, W)
            num_classes (int): number of classes
            merge_at (str): merge the two VGG16 networks early or late (default: "late")
        """
        super(DualVGG11, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.merge_at = merge_at

        self.branch_1_conv_block = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.branch_2_conv_block = nn.Sequential(
                
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            )

        self.conv_block =nn.Sequential(
            # Channel swap to 6=2*3
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        if self.merge_at == "early":
            factor = 1
        else:
            factor = 2
        self.fc = nn.Sequential(
            nn.Linear(512 * (input_size[1] // 32) * (input_size[2] // 32) * factor, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x1, x2):
        if self.merge_at == "early":
            x = torch.cat([x1, x2], dim=1)
            x = self.conv_block(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        elif self.merge_at == "late":
            x1 = self.branch_1_conv_block(x1)
            x2 = self.branch_2_conv_block(x2)
            x = torch.cat([x1, x2], dim=1)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x

class ReducedVGG11(nn.Module):
    def __init__(self, input_size: tuple[int, int, int], num_classes: int) -> None:
        """
        Reduced VGG11 model where channels are diveded by 8
        Args:
            input_size (tuple): input size of the image (C, H, W)
            num_classes (int): number of classes
        """
        super(ReducedVGG11, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * (input_size[1] // 32) * (input_size[2] // 32), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DualReducedVGG11(nn.Module):

    def __init__(self, input_size: tuple[int, int, int], num_classes: int, merge_at: str="late") -> None:
        """
        Args:
            input_size (tuple): input size of the image (C, H, W)
            num_classes (int): number of classes
            merge_at (str): merge the two VGG16 networks early or late (default: "late")
        """
        super(DualReducedVGG11, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.merge_at = merge_at

        self.branch_1_conv_block = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.branch_2_conv_block = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_block = nn.Sequential(

            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        if self.merge_at == "early":
            factor = 1
        else:
            factor = 2
        self.fc = nn.Sequential(
            nn.Linear(64 * (input_size[1] // 32) * (input_size[2] // 32) * factor, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x1, x2):
        if self.merge_at == "early":
            x = torch.cat([x1, x2], dim=1)
            x = self.conv_block(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        elif self.merge_at == "late":
            x1 = self.branch_1_conv_block(x1)
            x2 = self.branch_2_conv_block(x2)
            x = torch.cat([x1, x2], dim=1)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


# Spatial Transformer Networks
class STCNN(nn.Module):
    def __init__(self, input_size: tuple[int, int, int], num_classes: int) -> None:
        """
        Args:
            input_size (tuple): input size of the image (C, H, W)
            num_classes (int): number of classes
        """
        super(STCNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # According to STN paper
        self.localisation_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True), # Extra
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True) # Extra
        )

        # Also regressor block, for the 2 * 3 affine matrix
        self.localisation_fc_block = nn.Sequential(
            nn.Linear(64 * input_size[1] // 4 * input_size[2] // 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2 * 3) # 2 * 3 affine matrix
        )

        # Initialize the weights/bias with identity transformation
        self.localisation_fc_block[2].weight.data.zero_()
        self.localisation_fc_block[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )

        self.fc_block = nn.Sequential(
            nn.Linear(192 * input_size[1] // 16 * input_size[2] // 16, 3072),
            nn.ReLU(inplace=True),
            nn.Linear(3072, 3072),
            nn.ReLU(inplace=True),
            nn.Linear(3072, num_classes)
        )

    def stn(self, x):
        xs = self.localisation_conv_block(x)
        xs = xs.view(-1, 64 * self.input_size[1] // 4 * self.input_size[2] // 4)
        theta = self.localisation_fc_block(xs)
        theta = theta.view(-1, 2, 3)

        # Shortcut for affine_grid and grid_sample
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

class DualSTCNN(nn.Module):
    def __init__(self, input_size: tuple[int, int, int], num_classes: int, merge_at: Literal["early", "during_stn", "late"]) -> None:
        """
        Args:
            input_size (tuple): input size of the image (C, H, W)
            num_classes (int): number of classes
            merge_at (str): merge the two VGG16 networks early, during_stn, or late (default: "late")
        """
        super(DualSTCNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.merge_at = merge_at

        if self.merge_at != "early" :
            il = 3
        else:
            il = 6

        self.branch_1_localisation_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=il, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.branch_2_localisation_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=il, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.branch_1_localisation_fc_block = nn.Sequential(
            nn.Linear(64 * input_size[1] // 4 * input_size[2] // 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2 * 3)
        )

        self.branch_2_localisation_fc_block = nn.Sequential(
            nn.Linear(64 * input_size[1] // 4 * input_size[2] // 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2 * 3)
        )

        self.branch_1_localisation_fc_block[2].weight.data.zero_()
        self.branch_1_localisation_fc_block[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.branch_2_localisation_fc_block[2].weight.data.zero_()
        self.branch_2_localisation_fc_block[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.localisation_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=il, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.localisation_fc_block = nn.Sequential(
            nn.Linear(64 * input_size[1] // 4 * input_size[2] // 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2 * 3),
        )

        self.localisation_fc_block[2].weight.data.zero_()
        self.localisation_fc_block[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


        if self.merge_at == "late" :
            ic = 3
        else:
            ic = 6
        self.bran_1_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )

        self.bran_2_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )


        if self.merge_at != "late":
            factor = 1
        else:
            factor = 2
        self.fc_block = nn.Sequential(
            nn.Linear(192 * input_size[1] // 16 * input_size[2] // 16 * factor, 3072),
            nn.ReLU(inplace=True),
            nn.Linear(3072, 3072),
            nn.ReLU(inplace=True),
            nn.Linear(3072, num_classes)
        )

    def stn(self, x1, x2):
        if self.merge_at != "early":
            xs1 = self.branch_1_localisation_conv_block(x1)
            xs1 = xs1.view(-1, 64 * self.input_size[1] // 4 * self.input_size[2] // 4)
            theta1 = self.branch_1_localisation_fc_block(xs1)
            theta1 = theta1.view(-1, 2, 3)
            grid1 = F.affine_grid(theta1, x1.size())
            x1 = F.grid_sample(x1, grid1)

            xs2 = self.branch_2_localisation_conv_block(x2)
            xs2 = xs2.view(-1, 64 * self.input_size[1] // 4 * self.input_size[2] // 4)
            theta2 = self.branch_2_localisation_fc_block(xs2)
            theta2 = theta2.view(-1, 2, 3)
            grid2 = F.affine_grid(theta2, x2.size())
            x2 = F.grid_sample(x2, grid2)
            
            return x1, x2
        else:
            xs = x1
            xs = self.localisation_conv_block(xs)
            xs = xs.view(-1, 64 * self.input_size[1] // 4 * self.input_size[2] // 4)
            theta = self.localisation_fc_block(xs)
            theta = theta.view(-1, 2, 3)

            grid = F.affine_grid(theta, x1.size())
            x1 = F.grid_sample(x1, grid)

            return x1, None
        
    def forward(self, x1, x2):
        if self.merge_at == "early":
            x = torch.cat([x1, x2], dim=1)
            x, _ = self.stn(x, None)
            x = self.conv_block(x)

        elif self.merge_at == "during_stn":
            x1, x2 = self.stn(x1, x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.conv_block(x)

        elif self.merge_at == "late":
            x1, x2 = self.stn(x1, x2)
            x1 = self.bran_1_conv_block(x1)
            x2 = self.bran_2_conv_block(x2)
            x = torch.cat([x1, x2], dim=1)

        else:
            raise ValueError("merge_at should be 'early', 'during_stn', or 'late'")

        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x


# Auxiliary functions
def affine_grid(theta, size):
    """
    Generate a grid of (x, y) coordinates using the affine transformation matrix theta.
    Args:
        theta (torch.Tensor): affine transformation matrix (B, 2, 3)
        size (tuple): size of the output grid
    Returns:
        torch.Tensor: grid of (x, y) coordinates (B, H, W, 2)
    """
    B, H, W = size

    # Generate a grid of (x, y) coordinates, should range from -1 to 1 and evenly spaced. Shape: (H, W, 2)
    grid = torch.tensor([[[(w / W) * 2 - 1, (h / H) * 2 - 1] for w in range(W)] for h in range(H)], dtype=torch.float32)
    
    # Transform grid from (H, W, 2) to -> (1, H, W, 2) -> (B, H, W, 2) -> (B, H * W , 2) -> (B, H * W, 3) -> (B, H, W, 3) -> (B, H, W, 2)
    grid = grid.view(1, H, W, 2).repeat(B, 1, 1, 1)
    grid = grid.view(B, H * W, 2) # Flatten grid
    grid = torch.cat([grid, torch.ones(B, H * W, 1)], dim=2) # Add homogeneous coordinates
    grid = torch.bmm(grid, theta.transpose(1, 2))
    grid = grid.view(B, H, W, 2)
    return grid

def grid_sample(x, grid):
    """
    Perform bilinear interpolation on the input tensor x using the grid.
    Args:
        x (torch.Tensor): input tensor (B, C, H, W)
        grid (torch.Tensor): grid of (x, y) coordinates (B, H, W, 2)
    Returns:
        torch.Tensor: interpolated tensor
    """

    B, C, H, W = x.size()
    x = x.view(B, C, H * W)
    grid = grid.view(B, H * W, 2)

    # Normalise grid back to [0, W] and [0, H]
    grid = (grid + 1) / 2 * torch.tensor([W, H], dtype=torch.float32)
    grid = grid.clamp(min=0, max=1) # Crop

    # Get the four nearest grid points
    x0 = torch.floor(grid[..., 0]).long()
    y0 = torch.floor(grid[..., 1]).long()
    x1 = x0 + 1
    y1 = y0 + 1

    # Calculate the distance between the grid point and the target point
    dx = grid[..., 0] - x0.float()
    dy = grid[..., 1] - y0.float()

    # Calculate the weights based on the distance
    w00 = (1 - dx) * (1 - dy)
    w01 = (1 - dx) * dy
    w10 = dx * (1 - dy)
    w11 = dx * dy

    # Perform bilinear interpolation
    x0 = x0.clamp(0, W - 1)
    x1 = x1.clamp(0, W - 1)
    y0 = y0.clamp(0, H - 1)
    y1 = y1.clamp(0, H - 1)

    x0y0 = x[..., y0, x0]
    x0y1 = x[..., y1, x0]
    x1y0 = x[..., y0, x1]
    x1y1 = x[..., y1, x1]

    output = w00.unsqueeze(1) * x0y0 + w01.unsqueeze(1) * x0y1 + w10.unsqueeze(1) * x1y0 + w11.unsqueeze(1) * x1y1
    output = output.view(B, C, H, W)

    return output