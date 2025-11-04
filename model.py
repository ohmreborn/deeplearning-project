import torch.nn as nn

class SRCNN_BLOCK(nn.Module):
    def __init__(self, num_channels: int=3, scale_factor: int=2) -> None:
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=5, padding=2),
            nn.PixelShuffle(scale_factor)
            ])
        
    def forward(self, x):
        return self.model(x)

class SRCNN(nn.Module):
    def __init__(self, num_channels: int=3, num_block=3):
        super().__init__()
        self.model = nn.Sequential(
                *[ SRCNN_BLOCK(num_channels=num_channels, scale_factor=2) for _ in range(num_block)]
                )
    def forward(self, x):
        return self.model(x)
