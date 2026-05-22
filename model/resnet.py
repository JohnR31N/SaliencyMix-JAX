from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn


class BasicBlock(nn.Module):
    planes: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x

        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
        )(x)
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)

        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(out)
        out = nn.BatchNorm(use_running_average=not train)(out)

        if self.stride != 1 or residual.shape[-1] != self.planes:
            residual = nn.Conv(
                features=self.planes,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="VALID",
                use_bias=False,
            )(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        out = out + residual
        out = nn.relu(out)

        return out


class Bottleneck(nn.Module):
    planes: int
    stride: int = 1
    expansion: int = 4

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        out_channels = self.planes * self.expansion

        out = nn.Conv(
            features=self.planes,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=False,
        )(x)
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)

        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
        )(out)
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)

        out = nn.Conv(
            features=out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=False,
        )(out)
        out = nn.BatchNorm(use_running_average=not train)(out)

        if self.stride != 1 or residual.shape[-1] != out_channels:
            residual = nn.Conv(
                features=out_channels,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="VALID",
                use_bias=False,
            )(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        out = out + residual
        out = nn.relu(out)

        return out


class ResNet(nn.Module):
    block_type: str
    num_blocks: Sequence[int]
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        x shape: [B, H, W, C]
        CIFAR input: [B, 32, 32, 3]
        """

        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        planes_list = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]

        for stage_idx, (planes, num_block, stride) in enumerate(
            zip(planes_list, self.num_blocks, strides)
        ):
            for block_idx in range(num_block):
                block_stride = stride if block_idx == 0 else 1

                if self.block_type == "basic":
                    x = BasicBlock(
                        planes=planes,
                        stride=block_stride,
                        name=f"layer{stage_idx + 1}_block{block_idx + 1}",
                    )(x, train=train)
                elif self.block_type == "bottleneck":
                    x = Bottleneck(
                        planes=planes,
                        stride=block_stride,
                        name=f"layer{stage_idx + 1}_block{block_idx + 1}",
                    )(x, train=train)
                else:
                    raise ValueError(f"Unknown block_type: {self.block_type}")

        # Original PyTorch uses F.avg_pool2d(out, 4)
        # For CIFAR 32x32, after strides, feature map is 4x4.
        x = jnp.mean(x, axis=(1, 2))

        x = nn.Dense(features=self.num_classes)(x)

        return x


def ResNet18(num_classes=10):
    return ResNet(
        block_type="basic",
        num_blocks=[2, 2, 2, 2],
        num_classes=num_classes,
    )


def ResNet34(num_classes=10):
    return ResNet(
        block_type="basic",
        num_blocks=[3, 4, 6, 3],
        num_classes=num_classes,
    )


def ResNet50(num_classes=10):
    return ResNet(
        block_type="bottleneck",
        num_blocks=[3, 4, 6, 3],
        num_classes=num_classes,
    )


def ResNet101(num_classes=10):
    return ResNet(
        block_type="bottleneck",
        num_blocks=[3, 4, 23, 3],
        num_classes=num_classes,
    )