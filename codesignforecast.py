import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as TF
from skimage.metrics import structural_similarity as ssim
from forecastnet import  CustomDataset
from codesign import ForecastNet

BATCH_SIZE = 4  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
EPOCHS = 99  

forecastnet = ForecastNet()
forecastnet = forecastnet.to(DEVICE)  


loss_fn = nn.MSELoss()  
loss_fn = loss_fn.to(DEVICE)  


learning_rate = 3e-4  
optim = torch.optim.Adam(forecastnet.parameters(), learning_rate)


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((270, 270)),
    torchvision.transforms.ToTensor()
])

dataset = CustomDataset(root_dir='./FORECAST/DATA1', transform=transform)

train_indices = list(range(len(dataset)))
test_indices = train_indices[::10]  
train_indices = [i for i in train_indices if i not in test_indices]  

train_data = Subset(dataset, train_indices)
test_data = Subset(dataset, test_indices)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


def train():
    forecastnet.train()
    train_step = 0
    for inputs, labels in train_loader:

        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        outputs = forecastnet(inputs)
        loss = loss_fn(outputs, labels)

        optim.zero_grad()  
        loss.backward() 
        optim.step()  

        train_step += 1
        if train_step % 100 == 0:
            print(f'第{train_step}个Bach，loss={loss.item()}')


def test():
    forecastnet.eval()  
    ssim_scores = []  
    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = forecastnet(inputs)

            outputs_pil = TF.to_pil_image(outputs.squeeze(0).cpu())
            outputs_np = np.array(outputs_pil)
            labels_pil = TF.to_pil_image(labels.squeeze(0).cpu())
            labels_np = np.array(labels_pil)

            score = ssim(outputs_np, labels_np, multichannel=False)
            ssim_scores.append(score)

    avg_ssim = torch.tensor(ssim_scores).mean().item()
    print(f'SSIM分数={avg_ssim}')
    return avg_ssim

output_dir = "OUTPUT_CODESIGN/output_images"
os.makedirs(output_dir, exist_ok=True)
image_path = ['./TEST/Forecast/frame_290.jpg', './TEST/Forecast/frame_291.jpg', './TEST/Forecast/frame_292.jpg']
image = []
for path in image_path:
    image.append(transform(Image.open(path)))
input = torch.cat((image[0], image[1], image[2]), dim=0)
input = input.unsqueeze(0)
input = input.to(DEVICE)

def save_image(index):
    with torch.no_grad():

        outputs = forecastnet(input)

        outputs_pil = TF.to_pil_image(outputs.squeeze(0).cpu())
        outputs_pil.save(os.path.join(output_dir, f"{index}.png"))


if __name__ == '__main__':
    avg = []
    for i in range(EPOCHS):
        print(f'-----------第{i + 1}轮训练-----------')
        train()
        avg.append(test())
        save_image(i)
    with open('OUTPUT_CODESIGN/forecast.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Accuracy'])
        for index, value in enumerate(avg, start=1):
            writer.writerow([index, value])
    torch.save(forecastnet.state_dict(), 'OUTPUT_CODESIGN/forecastnet_weights.pth')
