import sys
import time
import torch

import torch.nn as nn

from lectures.seismic.model.blocks import EncoderBlock, DecoderBlock


class UnetModel(nn.Module):

    def __init__(self, in_channels, out_channels, model_depth=4):
        super(UnetModel, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
#        print("Final output shape: ", x.shape)
        return x


if __name__ == "__main__":
    inputs = torch.randn(1, 1, 96, 96, 96)
    print("The shape of inputs: ", inputs.shape)
    data_folder = "../processed"
    model = UnetModel(in_channels=1, out_channels=1)
    inputs = inputs.cuda()
    model.cuda()
    x = model(inputs)
    print(model)
