import torch
from torch import nn
from modelBackbone import S3D, load_weights
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class VisualModule(nn.Module):
    def __init__(self, num_class, classify=True, backbone_weights="./S3D_kinetics400.pt"):
        super(VisualModule, self).__init__()
        self.classify = classify
        self.backbone = load_weights(S3D(num_class), backbone_weights)
        self.head = nn.Sequential(
            # Projection block
            nn.Linear(832, 832), # Input: N x T/4 x 832
            BatchNormTemporal1d(832), #Batchnorm over temporarly variant data
            # TimeDistributed(nn.BatchNorm1d(832)),
            nn.ReLU(),
            # Conv Block - Temporal Convolutional (1d convolution) (N, Time, Len)
            Conv1dTemporal(832, 832, kernel_size=3, stride=1, padding='same'),
            Conv1dTemporal(832, 832, kernel_size=3, stride=1, padding='same'),
            nn.Linear(832, 512),
            nn.ReLU() # The output from this is  going into VL-Mapper or Classifier
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, num_class),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        y = self.backbone(x)
        y = self.head(y)
        gloss_pred = None
        if self.classify:
            gloss_pred = self.classifier(y).permute(1, 0, 2)


        return y, gloss_pred



class Conv1dTemporal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
        super(Conv1dTemporal, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding)

    def forward(self, x):
        y = self.conv(x.permute(0, 2, 1))
        return y.permute(0, 2, 1)


class BatchNormTemporal1d(nn.Module):
    def __init__(self, features):
        super(BatchNormTemporal1d, self).__init__()
        self.bn = nn.BatchNorm1d(features)

    def forward(self, x):
        y = self.bn(x.permute(0, 2, 1))
        return y.permute(0, 2, 1)

class TimeDistributed(nn.Module):
    """
    TimeDistributed for Pytorch which allows to apply a layer to every temporal slice of an input
    Args:
        Module: a Module instance
    PS : Input must be in the shape of (Seq_length, BS, )
    """

    def __init__(self, module, batch_first=False):
        if not isinstance(module, nn.Module):
            raise ValueError(
                "Please initialize `TimeDistributed` with a "
                f"`torch.nn.Module` instance. Received: {module.type}"
            )
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        print(x.shape)
        # Input shape is (default)
        #   - Seq_length, BS, *
        # or if batch_first
        #   - BS, Seq_length, *
        # It can also be provided as PackedSequence
        orig_x = x
        if isinstance(x, PackedSequence):
            x, lens_x = pad_packed_sequence(x, batch_first=self.batch_first)

        if self.batch_first:
            # BS, Seq_length, * -> Seq_length, BS, *
            x = x.transpose(0, 1)
        output = torch.stack([self.module(xt) for xt in x], dim=0)
        if self.batch_first:
            # Seq_length, BS, * -> BS, Seq_length, *
            output = output.transpose(0, 1)

        if isinstance(orig_x, PackedSequence):
            output = pack_padded_sequence(output, lens_x, batch_first=self.batch_first)
        return output