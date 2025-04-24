import torch
import torch.nn as nn
from base.base_net import BaseNet


class HSNet(BaseNet):

    def __init__(self):

        super().__init__()

        # Input parameters
        self.input_channels = 31
        self.patch_size = 32
        self.alpha = 0.1 # Leaky ReLU slope, default is 0.01

        # Representation dimensionality
        self.rep_dim = 128  # Output dimension of the last fully connected layer

        # Convolutional layers
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1)  # (31 -> 64 channels)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)             # (64 -> 128 channels)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)            # (128 -> 256 channels)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.bn3 = nn.BatchNorm2d(256, eps=1e-04, affine=False)

        # Fully connected layers
        flattened_size = 256 * (self.patch_size // 2 // 2 // 2) ** 2                            # After 3x2 max-pooling layers
        self.fc1 = nn.Linear(flattened_size, self.rep_dim, bias=False)  # Fully connected layer
        
        # Activation function
        self.leaky_relu = nn.LeakyReLU(self.alpha)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):

        # Convolutional layers with Leaky ReLU, batch normalization, and pooling
        x = self.conv1(x)                               # Conv1
        x = self.pool(self.leaky_relu(self.bn1(x)))     # MaxPool + Leaky ReLU + BatchNorm
        x = self.conv2(x)                               # Conv2
        x = self.pool(self.leaky_relu(self.bn2(x)))     # MaxPool + Leaky ReLU + BatchNorm
        x = self.conv3(x)                               # Conv3
        x = self.pool(self.leaky_relu(self.bn3(x)))     # MaxPool + Leaky ReLU + BatchNorm

        # Flatten the tensor for fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = self.fc1(x)                                 # FC1 (embedding layer)
 
        return x
    

class HSNetAutoencoder(BaseNet):
    """
    Autoencoder for HSNet, used for pretraining.
    """

    def __init__(self):
        super().__init__()

        # Input parameters
        self.input_channels = 31
        self.patch_size = 32
        self.alpha = 0.1  # Leaky ReLU slope, default is 0.01

        # Representation dimensionality
        self.rep_dim = 128  # Output dimension of the encoder

        # Encoder (same as HSNet)
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256, eps=1e-04, affine=False)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky_relu = nn.LeakyReLU(self.alpha)

        # Fully connected layers for latent representation
        flattened_size = 256 * (self.patch_size // 2 // 2 // 2) ** 2
        self.fc1 = nn.Linear(flattened_size, self.rep_dim, bias=False)

        # Decoder
        self.fc2 = nn.Linear(self.rep_dim, flattened_size, bias=False)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, self.input_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        # Encoder
        x = self.conv1(x)
        x = self.pool(self.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(self.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(self.leaky_relu(self.bn3(x)))

        # Flatten and fully connected layers
        x = torch.flatten(x, start_dim=1)
        latent = self.fc1(x)  # Latent representation

        # Decoder
        x = self.fc2(latent)
        x = x.view(-1, 256, self.patch_size // 8, self.patch_size // 8)  # Reshape to match Conv3 output
        x = self.deconv1(x)
        x = nn.functional.interpolate(self.leaky_relu(self.bn4(x)), scale_factor=2) # default mode='nearest' 
        x = self.deconv2(x)
        x = nn.functional.interpolate(self.leaky_relu(self.bn5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = nn.functional.interpolate(x, scale_factor=2)  # Final upsampling to original size

        return x