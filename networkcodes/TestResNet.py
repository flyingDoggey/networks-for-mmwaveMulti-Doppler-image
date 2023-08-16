import sys
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from resnet import RestNet18
import numpy as np
from matplotlib import pyplot as plt
# 训练次数
epoch_num = 100

def main():
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
    batchsz = 128
    train_dir = 'F:/dataSet/MMWAVEgesturepng/mm-wave-gesture-dataset-master/OpenSource/dataset/trainset/RDT_HAT_VAT_3Channel'
    test_dir = 'F:/dataSet/MMWAVEgesturepng/mm-wave-gesture-dataset-master/OpenSource/dataset/Atestset/RDT_HAT_VAT_3Channel'
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


    x, label = next(iter(train_loader))
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = Lenet5().to(device)
    model = RestNet18().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    acces = []
    eval_losses = []
    eval_acces = []

    print(model)
    print('------------------------start train-------------------------------')
    for epoch in range(epoch_num):
        train_loss = 0
        train_acc = 0
        model.train()
        for batchsz, (x, label) in enumerate(train_loader):
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label)
            _, pred = logits.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / x.shape[0]
            train_acc += acc
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        acces.append(train_acc / len(train_loader))
        print(epoch, 'train loss:', loss.item())
        #评估模型
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in test_loader:
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                loss = criteon(logits, label)
                
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            eval_acces.append(acc)
            eval_losses.append(loss.item())
            print(epoch, 'test acc:', acc)
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


if __name__ == '__main__':
    print(torch.cuda.is_available())
    #sys.stdout = open('F:/Test.txt', 'w')
    main()
    #sys.stdout.close()
    #sys.stdout = sys.__stdout__

