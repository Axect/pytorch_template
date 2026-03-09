"""Regression model for sine wave fitting.

A simple MLP (Multi-Layer Perceptron) that learns to approximate the function:
    y = 1.0 * sin(4*pi*x) + 0.5 * sin(10*pi*x) + 1.5 * x^2 + noise

Architecture: Linear(1, nodes) -> [GELU -> Linear(nodes, nodes)] x (layers-1) -> Linear(nodes, 1)
"""

from torch import nn


class MLP(nn.Module):
    def __init__(self, hparams, device="cpu"):
        super(MLP, self).__init__()
        self.hparams = hparams
        self.device = device

        nodes = hparams["nodes"]
        layers = hparams["layers"]
        input_size = 1
        output_size = 1

        net = [nn.Linear(input_size, nodes), nn.GELU()]
        for _ in range(layers - 1):
            net.append(nn.Linear(nodes, nodes))
            net.append(nn.GELU())
        net.append(nn.Linear(nodes, output_size))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
