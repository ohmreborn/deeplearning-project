import torch, torch.nn as nn
from torch.utils.data import DataLoader
import gc

from model import Generator, Discriminator
from dataset import data
from vgg_loss import VGGFeatureExtractor

if torch.cuda.is_available():
    device = torch.device("cuda")
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
opt_G: torch.optim.AdamW = torch.optim.AdamW(G.parameters(), lr=lr)
opt_D: torch.optim.AdamW = torch.optim.AdamW(D.parameters(), lr=lr)

dataset: data = data("dataset")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
gradient_accumulation_step = 16

# Initialize gradients
opt_G.zero_grad()
opt_D.zero_grad()

for epoch in range(num_epochs):
    total_d_loss = 0
    total_g_loss = 0
    
    for i, (lr_batch, hr_batch) in enumerate(dataloader):
        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)

        # ---------------------
        # Train Discriminator
        # ---------------------
        with torch.no_grad():
            fake = G(lr_batch)
        
        real_logits = D(hr_batch)
        fake_logits = D(fake)
        d_loss_real = bce(real_logits, torch.ones_like(real_logits))
        d_loss_fake = bce(fake_logits, torch.zeros_like(fake_logits))
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss = d_loss / gradient_accumulation_step
        d_loss.backward()
        
        if (i + 1) % gradient_accumulation_step == 0:
            opt_D.step()
            opt_D.zero_grad()
        
        # ---------------------
        # Train Generator
        # ---------------------
        fake = G(lr_batch)
        fake_logits_g = D(fake)
        adv_loss = bce(fake_logits_g, torch.ones_like(fake_logits_g))
        
        # Pixel loss
        pixel_loss = mse(fake, hr_batch)
        
        # Perceptual loss (features)
        feat_real = vgg(hr_batch)
        feat_fake = vgg(fake)
        perc_loss = l1(feat_fake, feat_real)

        g_loss = pixel_loss + 1.0 * perc_loss + 1e-3 * adv_loss
        g_loss = g_loss / gradient_accumulation_step
        g_loss.backward()
        
        if (i + 1) % gradient_accumulation_step == 0:
            opt_G.step()
            opt_G.zero_grad()

        # Accumulate losses (multiply back to get actual loss)
        total_d_loss += d_loss.item() * gradient_accumulation_step
        total_g_loss += g_loss.item() * gradient_accumulation_step
        
        # Print progress
        if (i + 1) % gradient_accumulation_step == 0:
            print(f"Epoch {epoch}, Step {i+1}/{len(dataloader)}: d_loss={d_loss.item()*gradient_accumulation_step:.4f}, g_loss={g_loss.item()*gradient_accumulation_step:.4f}")

    # Save checkpoints
    avg_d_loss = total_d_loss / len(dataloader)
    avg_g_loss = total_g_loss / len(dataloader)
    print(f"Epoch {epoch}: avg_d_loss={avg_d_loss:.4f}, avg_g_loss={avg_g_loss:.4f}")

torch.save({'G': G.state_dict(), 'D': D.state_dict()}, f'checkpoint.pth')
