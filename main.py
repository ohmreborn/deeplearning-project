import torch
import torch.nn as nn
from model import SRCNN
from dataset import data
from torch.utils.data import DataLoader

dataset: data = data("dataset")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
gradient_accumulation_step = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRCNN(3, 3).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    epoch_loss = 0
    for i, (low_re, high_re) in enumerate(dataloader):
        low_re, high_re = low_re.to(device), high_re.to(device)

        sr = model(low_re)
        loss = loss_fn(sr, high_re) / gradient_accumulation_step
        loss.backward()
        if ((i+1) % gradient_accumulation_step == 0):
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss={epoch_loss/len(dataloader):.6f}")

