import numpy as np
import torch.nn as nn

from processing.models.pytorch_tools.gradient_reversal import RevGrad


class FcnEncoderModule(nn.Module):
    def __init__(self, n_in, channels, kernel_sizes, dropout):
        super(FcnEncoderModule, self).__init__()
        input_dims = self._compute_input_dims(n_in, channels)
        self.encoder = _create_sequential(input_dims, channels, kernel_sizes, dropout)

    def forward(self, input_):
        return self.encoder(input_)

    def _compute_input_dims(self, n_in, channels):
        return [n_in] + channels[:-1]


class FcnRegressorModule(nn.Module):
    def __init__(self, input_dims, channels, kernel_sizes, dropout):
        super(FcnRegressorModule, self).__init__()
        self.regressor = _create_sequential(input_dims, channels, kernel_sizes, dropout)
        self.regressor.add_module("conv_pred_last", nn.Conv1d(channels[-1], 1, 1))

    def forward(self, features):
        return self.regressor(features)


class FcnDomainClassifierModule(nn.Module):
    def __init__(self, input_dims, channels, kernel_sizes, dropout, n_domains):
        super(FcnDomainClassifierModule, self).__init__()
        self.domain_classifier = nn.Sequential(
            RevGrad(),
            *np.concatenate(
                [_create_conv_layer(input_dim, channel, kernel_size, dropout) for input_dim, channel, kernel_size
                 in zip(input_dims, channels, kernel_sizes)]),
            nn.Conv1d(channels[-1], n_domains, 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, features):
        return self.domain_classifier(features)


def _create_conv_layer(input_dim, channels, kernel_size, dropout):
    return [
        nn.Conv1d(input_dim, channels, kernel_size),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(channels),
        nn.Dropout(dropout)
    ]


def _create_sequential(input_dims, channels, kernel_sizes, dropout):
    return nn.Sequential(*np.concatenate(
        [_create_conv_layer(input_dim, channel, kernel_size, dropout) for input_dim, channel, kernel_size
         in zip(input_dims, channels, kernel_sizes)]))


def _compute_decoder_kernel_size(encoder_kernel_sizes, history_length, pooling=1):
    kernel_size = history_length
    for encoder_kernel_size in encoder_kernel_sizes:
        kernel_size -= (encoder_kernel_size - 1) + 1 * (pooling - 1) + 1
        kernel_size = np.ceil(kernel_size / pooling + 1)
    return int(kernel_size)
