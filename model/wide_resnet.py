from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class WideBasicBlock(nn.Module):
    out_planes: int
    stride: int
    drop_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        in_planes = x.shape[-1]
        equal_in_out = in_planes == self.out_planes

        residual = x

        if not equal_in_out:
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.relu(x)
            out = x
        else:
            out = nn.BatchNorm(use_running_average=not train)(x)
            out = nn.relu(out)

        out = nn.Conv(
            features=self.out_planes,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
        )(out)

        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)

        if self.drop_rate > 0:
            out = nn.Dropout(rate=self.drop_rate, deterministic=not train)(out)

        out = nn.Conv(
            features=self.out_planes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
        )(out)

        if not equal_in_out:
            residual = nn.Conv(
                features=self.out_planes,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="VALID",
                use_bias=False,
                kernel_init=nn.initializers.kaiming_normal(),
            )(residual)

        return residual + out


class NetworkBlock(nn.Module):
    num_layers: int
    out_planes: int
    stride: int
    drop_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        for i in range(self.num_layers):
            block_stride = self.stride if i == 0 else 1
            x = WideBasicBlock(
                out_planes=self.out_planes,
                stride=block_stride,
                drop_rate=self.drop_rate,
                name=f"block_{i}",
            )(x, train=train)

        return x


class WideResNet(nn.Module):
    depth: int
    num_classes: int
    widen_factor: int = 1
    drop_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        x: [B, H, W, C]
        CIFAR input: [B, 32, 32, 3]
        """
        assert (self.depth - 4) % 6 == 0, "WideResNet depth should be 6n + 4"

        n = (self.depth - 4) // 6

        channels = [
            16,
            16 * self.widen_factor,
            32 * self.widen_factor,
            64 * self.widen_factor,
        ]

        x = nn.Conv(
            features=channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
        )(x)

        x = NetworkBlock(
            num_layers=n,
            out_planes=channels[1],
            stride=1,
            drop_rate=self.drop_rate,
            name="block1",
        )(x, train=train)

        x = NetworkBlock(
            num_layers=n,
            out_planes=channels[2],
            stride=2,
            drop_rate=self.drop_rate,
            name="block2",
        )(x, train=train)

        x = NetworkBlock(
            num_layers=n,
            out_planes=channels[3],
            stride=2,
            drop_rate=self.drop_rate,
            name="block3",
        )(x, train=train)

        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Original PyTorch does avg_pool2d(out, 8).
        # For CIFAR 32x32 after strides 1,2,2, feature map is 8x8.
        x = jnp.mean(x, axis=(1, 2))

        x = nn.Dense(
            features=self.num_classes,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )(x)

        return x


def WideResNet28_10(num_classes=10, drop_rate=0.3):
    return WideResNet(
        depth=28,
        widen_factor=10,
        drop_rate=drop_rate,
        num_classes=num_classes,
    )


def WideResNet16_8(num_classes=10, drop_rate=0.4):
    return WideResNet(
        depth=16,
        widen_factor=8,
        drop_rate=drop_rate,
        num_classes=num_classes,
    )