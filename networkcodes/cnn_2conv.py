import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import sys

# 打开txt文件
#sys.stdout = open('F:/dataSet/MMWAVEgesturepng/DT_RT_HVT_3Channel1.txt', 'w')
# 数据集
class SpectrogramDataset_RDI(Dataset):
    def __init__(self,root_dir,transform=None,is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = os.listdir(root_dir)
        self.data = []
        self.targets = []
        for i, label in enumerate(self.labels):
            if is_train:
                for j in range(1, 401):
                    img = Image.open(os.path.join(root_dir, label, label+str(j)+'.png'))
                    if self.transform:
                        img = self.transform(img)
                    self.data.append(img)
                    self.targets.append(i)
                #is_train = False
            else:
                for j in range(401, 451):
                    img = Image.open(os.path.join(root_dir, label, label+str(j)+'.png'))
                    if self.transform:
                        img = self.transform(img)
                    self.data.append(img)
                    self.targets.append(i)

        
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx],self.targets[idx]

#train_dir = 'F:/dataSet/pinputuRDI/train'
#test_dir = 'F:/dataSet/pinputuRDI/test'
train_dir = 'F:/dataSet/MMWAVEgesturepng/mm-wave-gesture-dataset-master/OpenSource/dataset/trainset/DT_RT_HVT_3Channel'
test_dir = 'F:/dataSet/MMWAVEgesturepng/mm-wave-gesture-dataset-master/OpenSource/dataset/Atestset/DT_RT_HVT_3Channel'
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5],std=[0.5])
])

train_dataset = SpectrogramDataset_RDI(train_dir,transform=transform)
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=False)
test_dataset = SpectrogramDataset_RDI(test_dir, transform=transform,is_train=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# 设计网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1)# RGB通道要改为3 黑白图像改为1
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)

        #self.resnet = models.resnet18(pretrained=True)

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(64*16*16,128)
        self.fc2 = nn.Linear(128,8)
        #self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        #x = self.resnet(x)
        x = x.view(-1, 64 * 16 * 16)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
ep = 200 # 训练次数
# 训练网络
for epoch in range(ep):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0        

#评估模型
# 将模型设置为评估模式
model.eval()

# 定义评估指标
correct = 0
total = 0

# 在测试集上进行评估
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 输出评估结果
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))