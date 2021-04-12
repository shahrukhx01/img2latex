import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        self.unfold = nn.Unfold(1) ## transforms (flattens) spatial dimensions of image after convolution block
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(True)
        )
        self.linear = nn.Linear(512, embed_size)

    def forward(self, images):
        ## convolution based feature extraction
        image_features = self.cnn(images)  # [B, 512, H', W']
        
        ## flatten convolution features for linear layer
        image_features = self.unfold(encoded_imgs).permute(0,2,1)  # [B, L=W'*H', 512]

        ## feed the encoded image to linear layer
        encoded_imgs = self.linear(image_features)

        return encoded_imgs
