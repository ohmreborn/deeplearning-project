from model import Generator

import matplotlib.pyplot as plt
from dataset import data
import torch


state = torch.load("./checkpoint.pth", map_location="cpu")
model: Generator = Generator()
model.load_state_dict(state["G"])

low_res = "./dataset/DIV2K_train_LR_x8"
low_res_image = "0001x8.png"
img: torch.Tensor = data.read_image(low_res, low_res_image).unsqueeze(0)

print(img.shape)
with torch.no_grad():
    out = model(img).squeeze(0).permute(1,2,0)
print(out.shape)
plt.imshow(out)
plt.show()

