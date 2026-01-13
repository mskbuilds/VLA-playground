import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from models import SimpleResNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

device = torch.device("cuda")

model = SimpleResNet(num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(trainloader):.4f}")
