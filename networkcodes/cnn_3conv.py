import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from torchsummary import summary
import sys

# 打开txt文件
#sys.stdout = open('F:/dataSet/MMWAVEgesturepng/DT_RT_HVT_3Channel1.txt', 'w')

num_classes = 8 # 分类数
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
train_dir = 'F:/dataSet/MMWAVEgesturepng/mm-wave-gesture-dataset-master/OpenSource/dataset/trainset/RDT_HAT_VAT_3Channel'
test_dir = 'F:/dataSet/MMWAVEgesturepng/mm-wave-gesture-dataset-master/OpenSource/dataset/Atestset/RDT_HAT_VAT_3Channel'


transform = transforms.Compose([
    # 卷积版1
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    

    #新增数据扩充
    #transforms.RandomResizedCrop(224),
    #transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    #transforms.RandomRotation(degrees=15),
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#train_dataset = SpectrogramDataset_RDI(train_dir,transform=transform)
#test_dataset = SpectrogramDataset_RDI(test_dir, transform=transform,is_train=False)

train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# 设计网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # conv卷积版本1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 8 * 8, 128) # 8192*128
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        
        x = self.relu(self.pool1(self.relu(self.conv1(x))))
        x = self.relu(self.pool2(self.relu(self.conv2(x))))
        x = self.relu(self.conv3(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#效果不理想
summary(model, (3, 64, 64))

losses = []
acces = []
eval_losses = []
eval_acces = []

ep = 800 # 训练次数
best_acc = 0.0 #最高准确率

losses = []
acces = []
eval_losses = []
eval_acces = []

# 训练
print('-----------------------------------------------------Start Training-----------------------------------------------------')
for epoch in range(ep):
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
    if eval_acc / len(test_loader) > best_acc:
        best_acc = eval_acc / len(test_loader)
        torch.save(model.state_dict(), 'F:/dataSet/MMWAVEgesturepng/RDT_HAT_VAT_3Channel.pth')
# 训练网络
#print('-------------------------start training-------------------------')
#for epoch in range(ep):
    
#    model.train()
#    running_loss = 0.0
#    for i, data in enumerate(train_loader, 0):
#        inputs, labels = data
#        inputs,labels = inputs.to(device),labels.to(device)
#        optimizer.zero_grad()
#        outputs = model(inputs)
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()

#        running_loss += loss.item()
#        if i % 10 == 9:
#            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
#            losses.append(running_loss / 10)
#            running_loss = 0.0        

######################################评估模型############################################
# 将模型设置为评估模式
#    model.eval()

# 定义评估指标
#    correct = 0
#    total = 0

# 在测试集上进行评估
#    with torch.no_grad():
#        for data in test_loader:
#            images, labels = data
#            images = images.to(device)
#            labels = labels.to(device)
#            outputs = model(images)
#            _, predicted = torch.max(outputs.data, 1)
#            total += labels.size(0)
#            correct += (predicted == labels).sum().item()
#    train_acc = 100 * correct / total
#    correct_test = 0
#    total_test = 0
#    with torch.no_grad():
#        for data in test_loader:
#            inputs, labels = data
#            inputs, labels = inputs.to(device), labels.to(device)
#            outputs = model(inputs)
#            _, predicted = torch.max(outputs.data, 1)
#            total_test += labels.size(0)
#            correct_test += (predicted == labels).sum().item()
#    test_acc = 100 * correct_test / total_test

#    acces.append(train_acc)
#    eval_acces.append(test_acc)
# 输出评估结果
#    print('Epoch %d: Train Acc %.2f%%, Test Acc %.2f%%' % (epoch + 1, train_acc, test_acc))
    #losses.append(running_loss / len(train_loader))
    
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)

# 保存模型
#torch.save(model.state_dict(), 'F:/dataSet/MMWAVEgesturepng/DT_RT_HVT_3Channel1.pth')
#    if test_acc > best_acc:
#        best_acc = test_acc
#        torch.save(model.state_dict(), 'best_model.pth')
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
#sys.stdout.close()
#sys.stdout = sys.__stdout__
print('train done')

