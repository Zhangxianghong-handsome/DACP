import csv
import torch
import torch.nn as nn
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision import datasets  
from codesign import MnistNet


BATCH_SIZE = 64 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
EPOCHS = 200  


mnistnet = MnistNet()
mnistnet = mnistnet.to(DEVICE)  


loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(DEVICE)  


learning_rate = 1e-4  
optim = torch.optim.SGD(mnistnet.parameters(), learning_rate, momentum=0.5)  


transformer = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, 0.5)])

train_data = datasets.FashionMNIST('FashionMNIST', True, transformer, download=True)
test_data = datasets.FashionMNIST('FashionMNIST', False, transformer, download=True)
train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, BATCH_SIZE, shuffle=False, num_workers=2)



def train():
    mnistnet.train()
    train_step = 0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        outputs = mnistnet(imgs)
        loss = loss_fn(outputs, targets)
        optim.zero_grad()  
        loss.backward()  
        optim.step()  

        train_step += 1
        if train_step % 300 == 0:
            print(f'第{train_step}个Bach，loss={loss.item()}')



def test():
    mnistnet.eval()
    total_accuracy = 0
    with torch.no_grad():
        for imgs, targets in test_loader:

            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            outputs = mnistnet(imgs)
            accuracy = (outputs.argmax(axis=1) == targets).sum()
            total_accuracy += accuracy
        print(f'准确率为{total_accuracy / 10000}')
        total_accuracy = total_accuracy.cpu().item()
        return total_accuracy / 10000


if __name__ == '__main__':
    acc = []
    for i in range(EPOCHS):
        print(f'-----------第{i + 1}轮训练-----------')
        train()
        acc.append(test())
    with open('OUTPUT_CODESIGN/mnistfashion.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Epoch', 'Accuracy'])

        for index, value in enumerate(acc, start=1):
            writer.writerow([index, value])

    torch.save(mnistnet.state_dict(), 'OUTPUT_CODESIGN/mnistnetfashion_weights.pth')

