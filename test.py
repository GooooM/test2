import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features= 64)
        self.fc2 = nn.Linear(in_features=64, out_features= 128)
        self.fc3 = nn.Linear(in_features=128, out_features= 256)
        self.fc4 = nn.Linear(in_features=256, out_features= 10)

    def forward(self, x):
        x = F.relu()(self.fc1(x))
        x = F.relu()(self.fc2(x))
        x = F.relu()(self.fc3(x))
        x = self.fc4(x)
        return x


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

dataset = torchvision.datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)


data_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=32, shuffle=True)

mlp = MLP()
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(mlp.parameters(), lr=2e-4, betas=(0.5, 0.99), eps=1e-8)



EPOCHS=1
total_step = 0
list_loss = list()
list_acc = list()

for epoch in range(EPOCHS):
    for i, data in enumerate(data_loader):
        total_step = total_step + 1
        input, label = data[0], data[1]  #
        # input shape [batch size ,channel, height, width]
        input = input.view(input.shape[0], -1)  # [batch size, channel*height*width]

        classification_result=mlp(input)  # [batch size, 10]

        l=loss(classification_result, label)
        list_loss.append(l.detach().item())  #item torch tensor 를 python의 형식으로 바꿔줌

        optim.zero_grad()
        l.backward()
        optim.step()

plt.figure()
plt.plot(range(len(list_loss)), list_loss, linestlye='--')
plt.show()