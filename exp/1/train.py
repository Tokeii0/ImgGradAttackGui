import torch
import torchvision.transforms as transforms
from tqdm import trange
from PIL import Image
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


class FlagDataset(torch.utils.data.Dataset):
    def __init__(self, flag_img, n, transform=None):
        self.image = flag_img
        self.transform = transform
        self.length = n

    def __len__(self):
        return self.length*2

    def __getitem__(self, idx):
        if idx % 2:
            sample = self.image
            if self.transform:
                sample = self.transform(sample)
            return (sample, 1)
        else:
            sample = np.random.randint(2, size=(28, 28)).astype(np.float32)
            if self.transform:
                sample = self.transform(sample)
            return (sample, 0)


device = torch.device('cuda')
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)

# train_data = FlagDataset(Image.open('flag.png').convert('L'), 2000,
#                          transform=transform)

# train_loader = torch.utils.data.DataLoader(
#     train_data, batch_size=256, shuffle=True)


class naive_model(torch.nn.Module):
    def __init__(self):
        super(naive_model, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        out = torch.nn.ReLU()(self.fc1(x))
        out = torch.nn.ReLU()(self.fc2(out))
        out = torch.nn.Softmax(dim=-1)(self.fc3(out))
        return out


# model = naive_model().to(device)
# optim = torch.optim.Adam(model.parameters(), lr=1e-4)
# loss_criterion = torch.nn.CrossEntropyLoss()
# t = trange(50, desc='train_loss')
# for epoch in t:
#     train_loss = 0.
#     for data, targets in train_loader:
#         data, targets = data.to(device), targets.to(device)
#         optim.zero_grad()

#         output = model(data)

#         loss = loss_criterion(output, targets)
#         loss.backward()
#         optim.step()

#         train_loss += loss.item()
#     t.set_description(
#         f'train_loss = {train_loss/len(train_data):.6f}', refresh=True)

# torch.save(model.state_dict(), 'model.pth')
