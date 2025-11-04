import torch
import torch.nn as nn

from typing import List, Union

class ConvBlock(nn.Module):
    def __init__(
            self, 
            in_channel: int, 
            out_channels: int,
            kernel_size: int,
            discriminator: bool = False,
            use_act: bool = True,
            use_bn: bool = True,
            **kwargs
            ) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=kernel_size, **kwargs)
        self.batch_norm: Union[nn.BatchNorm2d, nn.Identity] = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act: Union[nn.Identity, nn.LeakyReLU, nn.PReLU] = (
                nn.Identity() if not use_act else
                nn.LeakyReLU() if discriminator else nn.PReLU()
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv(x)
        out: torch.Tensor = self.batch_norm(out)
        return self.act(out)
        

class UpsampleBlock(nn.Module):
    def __init__(
            self, 
            in_c: int, 
            scale_factor: int,
            kernel_size: int,
                 ) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(in_channels=in_c, out_channels=in_c* scale_factor**2, kernel_size=kernel_size, stride=1, padding=1) # (batch, in_c, width, length) -> (batch, in_c*scale_factor**2, width, length)
        self.ps: nn.PixelShuffle = nn.PixelShuffle(scale_factor) # (batch, width, length, in_c*scale_factor**2) -> (batch, width*scale_factor, length*scale_factor, in_c)
        self.act: nn.PReLU = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ps(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.block1: ConvBlock = ConvBlock(
                in_channel=in_channel,
                out_channels=in_channel,
                kernel_size=3,
                padding=1,
                stride=1
                )
        self.block2: ConvBlock = ConvBlock(
                in_channel=in_channel,
                out_channels=in_channel,
                kernel_size=3,
                use_act=False,
                padding=1,
                stride=1
                )
        self.block: nn.Sequential = nn.Sequential(self.block1, self.block2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


class Generator(nn.Module):
    def __init__(
            self, 
            in_channel: int = 3, 
            num_channel:int = 16, 
            num_block: int = 8 
            ):
        super().__init__()
        self.initial: ConvBlock = ConvBlock(
                in_channel=in_channel,
                out_channels=num_channel,
                kernel_size=9,
                padding=4,
                use_bn=False
                )
        self.residual: nn.Sequential = nn.Sequential(*[ResidualBlock(num_channel) for _ in range(num_block)])
        self.convblock: ConvBlock = ConvBlock(num_channel, num_channel, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsample: nn.Sequential = nn.Sequential(*[UpsampleBlock(in_c=num_channel, scale_factor=2,kernel_size=3) for _ in range(3)])
        self.final: nn.Conv2d = nn.Conv2d(in_channels=num_channel, out_channels=in_channel, kernel_size=9, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial: torch.Tensor = self.initial(x)
        out = self.residual(initial)
        out = self.convblock(out) + initial
        out = self.upsample(out)
        return torch.tanh(self.final(out))

class Discriminator(nn.Module):
    def __init__(self, in_channel=3, features: List[int]=[68, 68, 128, 128, 256, 256]) -> None:
        super().__init__()
        block: List[ConvBlock] = []
        for idx, feature in enumerate(features):
            block.append(
                    ConvBlock(
                        in_channel=in_channel,
                        out_channels=feature,
                        kernel_size=3,
                        stride=1 + idx%2,
                        padding=1,
                        discriminator=True,
                        use_act=True,
                        use_bn=idx != 0
                        )
                    )
            in_channel = feature
        self.block: nn.Sequential = nn.Sequential(*block)
        self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((6, 6)),
                nn.Flatten(),
                nn.Linear(256*6*6, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, 1)
                )
    def forward(self, x) -> torch.Tensor:
        out: torch.Tensor = self.block(x)
        return self.classifier(out)

def test() -> None:
    low_resolution: int = 28
    x: torch.Tensor = torch.randn(5 ,3, low_resolution, low_resolution)
    gen: Generator = Generator()
    print(gen)
    gen_out: torch.Tensor = gen(x)
    disc: Discriminator = Discriminator()
    disc_out: torch.Tensor = disc(gen_out)
    print(gen_out.shape)
    print(disc_out.shape)

if __name__ == "__main__":
    test()
