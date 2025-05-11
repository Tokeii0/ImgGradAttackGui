import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class tanji_model(nn.Module):
    def __init__(self):
        super(tanji_model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(31 * 31, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return torch.sigmoid(y)


img = Image.open('flag.png')
img_tensor = transforms.ToTensor()(img)

tanji = tanji_model().cpu()
optimizer = optim.Adam(tanji.parameters(), lr=0.0001)
fn_loss = nn.BCELoss()


for idx in range(5000):

    data = torch.tensor(np.random.randint(2, size=(256, 31, 31)).astype(np.float32))
    label = torch.tensor(np.random.randint(1, size=(256, 1)).astype(np.float32))

    n = np.random.randint(256)

    data[n] = img_tensor
    label[n] = 1

    optimizer.zero_grad()   
    output = tanji(data)
    loss = fn_loss(output, label)
    loss.backward()
    optimizer.step()

    if idx % 100 == 0:
        print(f'Loss: {loss.sum()}')

torch.save(tanji.state_dict(), 'model.pth')