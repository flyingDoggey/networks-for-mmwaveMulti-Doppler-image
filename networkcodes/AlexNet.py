import os
import numpy as np
import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from torchsummary import summary
from PIL import Image
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
transform = torchvision.transforms.Compose([
    
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])

    #torchvision.transforms.Resize((224, 224)),
    #torchvision.transforms.RandomResizedCrop(224),
    #torchvision.transforms.RandomHorizontalFlip(),
    #torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    #torchvision.transforms.RandomRotation(degrees=15),
    #torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dir = 'F:/dataSet/MMWAVEgesturepng/mm-wave-gesture-dataset-master/OpenSource/dataset/trainset/DT_HAT_VAT_3Channel'
test_dir = 'F:/dataSet/MMWAVEgesturepng/mm-wave-gesture-dataset-master/OpenSource/dataset/Atestset/DT_HAT_VAT_3Channel'
train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 8, dropout: float = 0.5) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device)
summary(model, (3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = []
acces = []
eval_losses = []
eval_acces = []

# 训练
print('-----------------------------------------------------Start Training-----------------------------------------------------')
for epoch in range(100):
    train_loss = 0
    train_acc = 0
    model.train()
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}'.format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))
    
    eval_loss = 0
    eval_acc = 0
    model.eval()
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('epoch: {}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch, eval_loss / len(test_loader), eval_acc / len(test_loader)))

# 保存模型
torch.save(model.state_dict(), 'F:\dataSet\MMWAVEgesturepng/AlexNet.pth') 

# 绘制训练、测试loss曲线
plt.title('train and test loss')
plt.plot(np.arange(len(losses)), losses)
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# 绘制accuracy曲线
plt.title('train and test accuracy')
plt.plot(np.arange(len(acces)), acces)
plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.legend(['Train Acc', 'Test Acc'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# # 加载模型并推理
# model.load_state_dict(torch.load('models/AlexNet.pth'))
# model.eval()
# img = Image.open('data/flower_images/test/Tulip/1b2021d35e.jpg')
# img = transform(img)
# img = torch.unsqueeze(img, dim=0)
# with torch.no_grad():
#     img = img.to(device)
#     out = model(img)
#     _, pred = out.max(1)
#     print('预测结果为：{}'.format(train_dataset.classes[pred.item()]))

    
