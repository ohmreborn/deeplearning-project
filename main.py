import torch, torch.nn as nn
from torch.utils.data import DataLoader

from model import Generator, Discriminator
from dataset import data
from vgg_loss import VGGFeatureExtractor

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)
G: Generator = Generator().to(device)
D: Discriminator = Discriminator().to(device)
vgg: VGGFeatureExtractor = VGGFeatureExtractor().to(device)

lr: float = 1e-4
num_epochs: int = 1

mse: nn.MSELoss = nn.MSELoss()
bce: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
l1: nn.L1Loss = nn.L1Loss()
opt_G: torch.optim.Adam = torch.optim.Adam(G.parameters(), lr=lr)
opt_D: torch.optim.Adam = torch.optim.Adam(D.parameters(), lr=lr)

dataset: data = data("dataset")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
gradient_accumulation_step = 16

for epoch in range(num_epochs):
    total_d_loss = 0
    total_g_loss = 0
    for i, (lr_batch, hr_batch) in enumerate(dataloader):
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        fake = G(lr_batch).detach()
        print("fake")
        real_logits = D(hr_batch)
        print("fake")
        fake_logits = D(fake)
        print("fake")
        d_loss_real = bce(real_logits, torch.ones_like(real_logits))
        print("fake")
        d_loss_fake = bce(fake_logits, torch.zeros_like(fake_logits))
        print("fake")
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss /= gradient_accumulation_step
        print("fake")
        d_loss.backward(); 
        print("fake")
        if (i % gradient_accumulation_step == gradient_accumulation_step-1):
            opt_D.step()
            opt_D.zero_grad()
        # ---------------------
        # Train Generator
        # ---------------------
        fake = G(lr_batch)
        print(lr_batch.shape, hr_batch.shape, fake.shape)
        # Adversarial loss (want D(fake) -> 1)
        adv_loss = bce(D(fake), torch.ones_like(fake_logits))
        # Pixel loss
        pixel_loss = mse(fake, hr_batch)
        # Perceptual loss (features)
        feat_real = vgg(hr_batch)
        feat_fake = vgg(fake)
        perc_loss = l1(feat_fake, feat_real)

        g_loss = pixel_loss + 1.0 * perc_loss + 1e-3 * adv_loss

        g_loss /= gradient_accumulation_step
        g_loss.backward()

        if (i % gradient_accumulation_step == gradient_accumulation_step-1):
            opt_G.step()
            opt_G.zero_grad() 

        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()

    # Save checkpoints
    print(f"Epoch {epoch}: d_loss={total_d_loss:.4f}, g_loss={total_g_loss:.4f}")

torch.save({'G': G.state_dict(), 'D': D.state_dict()}, f'checkpoint.pth')
