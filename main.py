import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import argparse

# Set command line parameters
parser = argparse.ArgumentParser(description='PyTorch CIFA10 Net Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size(default: 100)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
parser.add_argument('--worker-numbers', type=int, default=6, metavar='N',
                        help='number of loading data\'s workers (default: 6)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--no-train', action='store_true', default=False,
                        help='If train the Model')
parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
args = parser.parse_args()

Root = "./data"
Batch_size = args.batch_size
Num_workers = args.worker_numbers
best_acc = 0
SAVE_PKL = "cifar10.pkl"
loss_func = nn.CrossEntropyLoss()

# choose devices
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(Device)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout10(x)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.avgpool(x)
        x = self.dropout10(x)
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def net_train(net, train_data_load, optimizer, epoch, log_interval):
    net.train()

    begin = datetime.datetime.now()
    total = len(train_data_load.dataset)
    # Sum of the loss function values
    train_loss = 0
    # The number of samples has been identified correctly
    ok = 0
    for i, data in enumerate(train_data_load, 0):
        img, label = data
        img, label = img.cuda(), label.cuda()
        optimizer.zero_grad()
        outs = net(img)
        loss = loss_func(outs, label)
        loss.backward()
        optimizer.step()

        # Add up
        train_loss += loss.item()
        _, predicted = torch.max(outs.data, 1)
        # Add up
        ok += (predicted == label).sum()

        if (i + 1) % log_interval == 0:
            # average
            loss_mean = train_loss / (i + 1)
            traind_total = (i + 1) * len(label)
            # accuracy
            acc = 100. * ok / traind_total
            progress = 100. * traind_total / total
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Acc: {:.6f}'.format(
                epoch, traind_total, total, progress, loss_mean, acc))

    end = datetime.datetime.now()
    print('one epoch spend: ', end - begin)


def net_test(net, test_data_load, epoch):
    
    correct = 0
    total = 0
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    with torch.no_grad():
        for data in test_data_load:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print("EPOCH=", epoch)
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_data_load:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_data_load = torch.utils.data.DataLoader(trainset, batch_size=Batch_size,
                                            shuffle=True, num_workers=Num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                            download=True, transform=transform)
    test_data_load = torch.utils.data.DataLoader(testset, batch_size=Batch_size,
                                            shuffle=False, num_workers=Num_workers)
    
    net = Net().to(device=Device)
    
    if args.no_train:
        net.load_state_dict(torch.load(SAVE_PKL))
        net_test(net, test_data_load, 0)
        return
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    start_time = datetime.datetime.now()

    for epoch in range(1, args.epochs + 1):
        net_train(net, train_data_load, optimizer, epoch, args.log_interval)
        net_test(net, test_data_load, epoch)

    end_time = datetime.datetime.now()

    global best_acc
    print('CIFAR10 pytorch LeNet Train: EPOCH:{}, BATCH_SZ:{}, LR:{}, ACC:{}'.format(args.epochs, args.batch_size, args.lr, best_acc))
    print('train spend time: ', end_time - start_time)
    
    if args.save_model:
        torch.save(net.state_dict(), SAVE_PKL)


if __name__ == '__main__':
    main()