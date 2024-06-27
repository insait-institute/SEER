from resnet import ResNet, BasicBlock
import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
from utils import datasets_Cifar10

bs_pos = 1
bs_neg = 127
device = 'cuda:0'
prop = 'bright'

checkpoint = 'others/Initialization.params'

bs = bs_pos + bs_neg
model = ResNet(BasicBlock, [2, 2, 2, 2], [64,128,256,512], 'ReLU', num_classes=10)
checkpoint = torch.load(checkpoint, map_location=torch.device(device))
model.load_state_dict(checkpoint)
model.to(device)

public_loss = torch.nn.CrossEntropyLoss(reduction='none').to(device)

traind, testd = datasets_Cifar10()
loader_train = torch.utils.data.DataLoader( traind, shuffle=True, num_workers=2, batch_size=bs )
loader_test = torch.utils.data.DataLoader( testd, shuffle=True, num_workers=2, batch_size=bs )

optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)


for epoch in range(200):
    model.train() 

    avg_loss = 0
    avg_acc = 0
    batches = 0
    for j, (images, labels) in tqdm(enumerate(loader_train)):
        model.zero_grad()
        datapoints = images.flatten(start_dim=0,end_dim=0).to(device)
        labels = labels.flatten(start_dim=0,end_dim=0).to(device)


        loss = public_loss(model(datapoints),labels).sum()
        loss.backward()
        optimizer.step()

        if labels.shape[0] != bs:
            continue
        batches += 1
        avg_loss += loss.item()
        avg_acc += (model(datapoints).argmax(dim=1) == labels).sum().item() / bs
    avg_loss /= batches
    avg_acc /= batches

    model.eval()

    # testing
    avg_acc_test = 0
    batches = 0
    for j, (images, labels) in tqdm(enumerate(loader_test)):
        datapoints = images.flatten(start_dim=0,end_dim=0).to(device)
        labels = labels.flatten(start_dim=0,end_dim=0).to(device)
        batches += 1
        avg_acc_test += (model(datapoints).argmax(dim=1) == labels).sum().item() / bs
    avg_acc_test /= batches

    print(f'Epoch {epoch} | Loss {avg_loss} | Acc {avg_acc} | AccTest {avg_acc_test}')
    torch.save(model.state_dict(), f's/standard_new_EP{epoch}_{round(avg_acc_test*100)}.params')
