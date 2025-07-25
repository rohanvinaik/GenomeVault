"""
KAN Layer Implementation

Implements Kolmogorov-Arnold Network layers based on the theorem:
f(x) = Σ_q Φ_q(Σ_p φ_{q,p}(x_p))

Where each φ_{q,p} is a learnable univariate function (spline).
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SplineFunction(nn.Module):
    """Learnable univariate spline function"""

    def __init__(self, num_knots: int = 10, degree: int = 3):
        super().__init__()
        self.num_knots = num_knots
        self.degree = degree

        # Initialize knot positions uniformly
        self.register_buffer("knots", torch.linspace(-1, 1, num_knots))

        # Learnable coefficients for each spline segment
        self.coefficients = nn.Parameter(torch.randn(num_knots - 1, degree + 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate spline at input points"""
        # Ensure input is in [-1, 1]
        x = torch.clamp(x, -1, 1)

        # Find which segment each point belongs to
        segment_idx = torch.searchsorted(self.knots[1:], x)
        segment_idx = torch.clamp(segment_idx, 0, self.num_knots - 2)

        # Get local position within segment
        t = (x - self.knots[segment_idx]) / (self.knots[segment_idx + 1] - self.knots[segment_idx])

        # Evaluate polynomial for each segment
        result = torch.zeros_like(x)
        for i in range(self.degree + 1):
            result += self.coefficients[segment_idx, i] * (t**i)

        return result


class KANLayer(nn.Module):
    """
    Single KAN layer implementing the Kolmogorov-Arnold representation

    This layer transforms n inputs to m outputs using learnable univariate functions
    on each edge, following: y_j = Σ_i φ_{j,i}(x_i)
    """

    def __init__(
        self, in_features: int, out_features: int, num_knots: int = 10, spline_degree: int = 3
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create a spline function for each edge in the layer
        self.splines = nn.ModuleList(
            [
                nn.ModuleList(
                    [SplineFunction(num_knots, spline_degree) for _ in range(in_features)]
                )
                for _ in range(out_features)
            ]
        )

        # Optional: learnable scaling factors
        self.scale = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.out_features, device=x.device)

        for j in range(self.out_features):
            for i in range(self.in_features):
                # Apply univariate function φ_{j,i} to input x_i
                output[:, j] += self.splines[j][i](x[:, i])

        # Apply scaling
        output = output * self.scale

        return output

    def get_symbolic_expression(
        self, idx_out: int, idx_in: int, num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the learned function φ_{idx_out,idx_in} for visualization

        Returns:
            x_values: Input points
            y_values: Output values of the learned function
        """
        x = torch.linspace(-1, 1, num_points)
        with torch.no_grad():
            y = self.splines[idx_out][idx_in](x)
        return x.numpy(), y.numpy()


class LinearKAN(nn.Module):
    """
    Linear variant of KAN with simplified spline basis

    Uses piecewise linear functions instead of polynomials for efficiency.
    This is particularly effective for genomic data where patterns are often linear.
    """

    def __init__(self, in_features: int, out_features: int, num_segments: int = 20):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_segments = num_segments

        # Learnable breakpoints and slopes for piecewise linear functions
        self.breakpoints = nn.Parameter(
            torch.linspace(-1, 1, num_segments + 1).repeat(out_features, in_features, 1)
        )
        self.values = nn.Parameter(torch.randn(out_features, in_features, num_segments + 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fast piecewise linear transformation"""
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(1).expand(-1, self.out_features, -1)

        # Compute piecewise linear interpolation
        output = torch.zeros(batch_size, self.out_features, device=x.device)

        for i in range(self.num_segments):
            # Find points in this segment
            mask = (x_expanded >= self.breakpoints[:, :, i]) & (
                x_expanded < self.breakpoints[:, :, i + 1]
            )

            # Linear interpolation within segment
            t = (x_expanded - self.breakpoints[:, :, i]) / (
                self.breakpoints[:, :, i + 1] - self.breakpoints[:, :, i] + 1e-8
            )

            interp_values = self.values[:, :, i] * (1 - t) + self.values[:, :, i + 1] * t
            output += (mask * interp_values).sum(dim=2)

        return output


class ConvolutionalKAN(nn.Module):
    """
    Convolutional variant of KAN for sequence data

    Applies KAN transformations in a convolutional manner, useful for
    processing genomic sequences with local patterns.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_knots: int = 10,
        spline_degree: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Create KAN layer for each position in the kernel
        self.kan_layers = nn.ModuleList(
            [
                KANLayer(in_channels, out_channels, num_knots, spline_degree)
                for _ in range(kernel_size)
            ]
        )

        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolutional KAN

        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns:
            Output tensor of shape (batch_size, out_channels, sequence_length)
        """
        batch_size, _, seq_len = x.shape

        # Pad input
        x_padded = F.pad(x, (self.padding, self.padding))

        # Apply KAN transformation at each position
        output = torch.zeros(batch_size, self.out_channels, seq_len, device=x.device)

        for i in range(seq_len):
            # Extract local window
            window = x_padded[:, :, i : i + self.kernel_size]

            # Apply KAN layer for each position in kernel
            for j, kan in enumerate(self.kan_layers):
                output[:, :, i] += kan(window[:, :, j].permute(0, 1))

        return output
