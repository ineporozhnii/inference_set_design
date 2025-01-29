import math
from typing import Tuple

import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    """
    Simple MLP model
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_hidden_layers: int,
        skip_connections: bool,
        dropout: float,
    ):
        super(MLP, self).__init__()
        self.skip_connections = skip_connections
        self.relu = torch.nn.ReLU()
        self.use_dropout = dropout > 0.0
        self.dropout = torch.nn.Dropout(dropout) if self.use_dropout else None
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, hidden_size))
        for i in range(n_hidden_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))

        self.output_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for layer_idx, layer in enumerate(self.layers):
            if self.skip_connections and layer_idx % 2:
                x = x + layer(x)
            else:
                x = layer(x)
            x = self.relu(x)
            if self.use_dropout:
                x = self.dropout(x)

        x = self.output_layer(x)
        return x


class ResidualBlockLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, use_batch_norm: bool = False, leaky_slope: float = 0.01):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(leaky_slope)
        self.batch_norm1 = nn.BatchNorm1d(output_dim) if use_batch_norm else None
        self.batch_norm2 = nn.BatchNorm1d(output_dim) if use_batch_norm else None

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        if self.batch_norm1:
            out = self.batch_norm1(out)
        out = self.leaky_relu(out)
        out = self.linear2(out)
        if self.batch_norm2:
            out = self.batch_norm2(out)
        out += residual
        out = self.leaky_relu(out)
        return out


class ResMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: Tuple[int],
        n_res_block: int,
        output_size: int,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        leaky_slope: float = 0.01,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.LeakyReLU(leaky_slope))
        for _ in range(n_res_block):
            self.layers.append(ResidualBlockLayer(hidden_size, hidden_size, use_batch_norm, leaky_slope))
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LinearEnsemble(nn.Module):
    def __init__(self, n_models: int, input: int, output: int):
        super(LinearEnsemble, self).__init__()

        self.weight_matrix = nn.Parameter(torch.randn(n_models, input, output))
        self.bias_vector = nn.Parameter(torch.randn(n_models, 1, output))

        stdv = 1.0 / math.sqrt(self.weight_matrix.size(1))
        self.weight_matrix.data.uniform_(-stdv, stdv)
        self.bias_vector.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = torch.bmm(x, self.weight_matrix) + self.bias_vector
        return x


class MLPEnsemble(torch.nn.Module):
    """
    Simple MLP model
    """

    def __init__(
        self,
        n_models: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_hidden_layers: int,
        skip_connections: bool,
    ):
        super(MLPEnsemble, self).__init__()
        self.skip_connections = skip_connections
        self.relu = torch.nn.ReLU()
        self.layers = torch.nn.ModuleList()
        self.layers.append(LinearEnsemble(n_models, input_size, hidden_size))
        for i in range(n_hidden_layers - 1):
            self.layers.append(LinearEnsemble(n_models, hidden_size, hidden_size))

        self.output_layer = LinearEnsemble(n_models, hidden_size, output_size)

    def forward(self, x):
        for layer_idx, layer in enumerate(self.layers):
            if self.skip_connections and layer_idx % 2:
                x = x + layer(x)
            else:
                x = layer(x)
            x = self.relu(x)

        x = self.output_layer(x)
        return x


class ResidualBlockLayerEnsemble(nn.Module):
    def __init__(
        self, n_models: int, input_dim: int, output_dim: int, use_batch_norm: bool = False, leaky_slope: float = 0.01
    ):
        super().__init__()
        self.n_models = n_models
        self.output_dim = output_dim
        self.linear1 = LinearEnsemble(n_models, input_dim, output_dim)
        self.linear2 = LinearEnsemble(n_models, output_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(leaky_slope)
        self.batch_norm1 = nn.BatchNorm1d(n_models * output_dim) if use_batch_norm else None
        self.batch_norm2 = nn.BatchNorm1d(n_models * output_dim) if use_batch_norm else None

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        if self.batch_norm1:
            out = self.batch_norm1(out.reshape(out.shape[1], self.n_models * self.output_dim)).reshape(
                self.n_models, out.shape[1], self.output_dim
            )
        out = self.leaky_relu(out)
        out = self.linear2(out)
        if self.batch_norm2:
            out = self.batch_norm2(out.reshape(out.shape[1], self.n_models * self.output_dim)).reshape(
                self.n_models, out.shape[1], self.output_dim
            )
        out += residual
        out = self.leaky_relu(out)
        return out


class ResMLPEnsemble(nn.Module):
    def __init__(
        self,
        n_models: int,
        input_size: int,
        hidden_size: Tuple[int],
        n_res_block: int,
        output_size: int,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        leaky_slope: float = 0.01,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(LinearEnsemble(n_models, input_size, hidden_size))
        self.layers.append(nn.LeakyReLU(leaky_slope))
        for _ in range(n_res_block):
            self.layers.append(
                ResidualBlockLayerEnsemble(n_models, hidden_size, hidden_size, use_batch_norm, leaky_slope)
            )
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(LinearEnsemble(n_models, hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ParallelLinearLayer(nn.Module):
    """
    Multiple LinearLayers applied in parallel to the input
    - makes use of torch.bmm to apply multiple independent linear transformations to the same input
    - can be used for ensembling or multi-task learning
    """

    def __init__(self, n_models: int, input: int, output: int):
        super().__init__()

        self.weight_matrix = nn.Parameter(torch.randn(n_models, input, output))
        self.bias_vector = nn.Parameter(torch.randn(n_models, 1, output))

        stdv = 1.0 / math.sqrt(self.weight_matrix.size(1))
        self.weight_matrix.data.uniform_(-stdv, stdv)
        self.bias_vector.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = torch.bmm(x, self.weight_matrix) + self.bias_vector
        return x


class ParallelMLPs(torch.nn.Module):
    def __init__(
        self,
        n_models: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_hidden_layers: int,
        skip_connections: bool,
        dropout: float,
    ):
        super().__init__()
        self.skip_connections = skip_connections
        self.relu = torch.nn.ReLU()
        self.use_dropout = dropout > 0.0
        self.dropout = torch.nn.Dropout(dropout) if self.use_dropout else None
        self.layers = torch.nn.ModuleList()
        self.layers.append(ParallelLinearLayer(n_models, input_size, hidden_size))
        for i in range(n_hidden_layers - 1):
            self.layers.append(ParallelLinearLayer(n_models, hidden_size, hidden_size))

        self.output_layer = ParallelLinearLayer(n_models, hidden_size, output_size)

    def forward(self, x):
        for layer_idx, layer in enumerate(self.layers):
            if self.skip_connections and layer_idx % 2:
                x = x + layer(x)
            else:
                x = layer(x)
            x = self.relu(x)
            if self.use_dropout:
                x = self.dropout(x)

        x = self.output_layer(x)
        return x


class ParallelResidualBlockLayer(nn.Module):
    def __init__(
        self, n_models: int, input_dim: int, output_dim: int, use_batch_norm: bool = False, leaky_slope: float = 0.01
    ):
        super().__init__()
        self.n_models = n_models
        self.output_dim = output_dim
        self.linear1 = ParallelLinearLayer(n_models, input_dim, output_dim)
        self.linear2 = ParallelLinearLayer(n_models, output_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(leaky_slope)
        self.batch_norm1 = nn.BatchNorm1d(n_models * output_dim) if use_batch_norm else None
        self.batch_norm2 = nn.BatchNorm1d(n_models * output_dim) if use_batch_norm else None

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        if self.batch_norm1:
            out = self.batch_norm1(out.reshape(out.shape[1], self.n_models * self.output_dim)).reshape(
                self.n_models, out.shape[1], self.output_dim
            )
        out = self.leaky_relu(out)
        out = self.linear2(out)
        if self.batch_norm2:
            out = self.batch_norm2(out.reshape(out.shape[1], self.n_models * self.output_dim)).reshape(
                self.n_models, out.shape[1], self.output_dim
            )
        out += residual
        out = self.leaky_relu(out)
        return out


class ParallelResMLPs(nn.Module):
    def __init__(
        self,
        n_models: int,
        input_size: int,
        hidden_size: int,
        n_res_block: int,
        output_size: int,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        leaky_slope: float = 0.01,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ParallelLinearLayer(n_models, input_size, hidden_size))
        self.layers.append(nn.LeakyReLU(leaky_slope))
        for _ in range(n_res_block):
            self.layers.append(
                ParallelResidualBlockLayer(n_models, hidden_size, hidden_size, use_batch_norm, leaky_slope)
            )
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(ParallelLinearLayer(n_models, hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MultiTaskMLP(nn.Module):
    """
    This model has common trunk but after n_common_layers it splits into n_tasks different heads,
    each having n_task_layers.
    """

    def __init__(
        self,
        input_size: int,
        trunk_hidden_size: int,
        n_trunk_res_block: int,
        n_tasks: int,
        task_hidden_size: int,
        n_task_layers: int,
        output_size: int,
        skip_connections: bool,
        dropout: float,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.trunk = MLP(
            input_size=input_size,
            hidden_size=trunk_hidden_size,
            output_size=trunk_hidden_size,
            n_hidden_layers=n_trunk_res_block,
            skip_connections=skip_connections,
            dropout=dropout,
        )
        self.heads = ParallelMLPs(
            n_models=n_tasks,
            input_size=trunk_hidden_size,
            hidden_size=task_hidden_size,
            output_size=output_size,
            n_hidden_layers=n_task_layers,
            skip_connections=skip_connections,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.trunk(x)
        x = x.unsqueeze(0).repeat(self.n_tasks, 1, 1)
        x = self.heads(x)
        return x.movedim(0, 1)


def count_parameters(model: nn.Module):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
