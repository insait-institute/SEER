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
import matplotlib.pyplot as plt 
import random
import numpy as np
from utils import GradientExtractor,ParamSelector,get_decoder,saveimg,datasets_Cifar10
import json 

"""
    Set this to true to just get the car image and stats
"""
PLOT_CAR = False


def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
set_all_seeds(42) 

torch.set_printoptions(sci_mode=False)
device = 'cuda:0'

model = ResNet(BasicBlock, [2, 2, 2, 2], [64,128,256,512], 'ReLU', num_classes=10)
public_loss = torch.nn.CrossEntropyLoss(reduction='none').to(device)
traind, testd = datasets_Cifar10()

# Prep data
bszs = [16, 32, 64]
nb_batches_per_bsz = 5
usetrain = True 

data = {}
for b in bszs:
    data[b] = []

for bsz in bszs:
    if usetrain:
        loader = torch.utils.data.DataLoader( traind, shuffle=True, num_workers=2, batch_size=bsz) 
        for j, (images, labels) in enumerate(loader):
            if j == nb_batches_per_bsz:
                break
            data[bsz].append((images, labels))

    loader = torch.utils.data.DataLoader( testd, shuffle=True, num_workers=2, batch_size=bsz) 
    for j, (images, labels) in enumerate(loader):
        if j == nb_batches_per_bsz:
            break
        data[bsz].append((images, labels))

### Find checkpoints to test
checkpoints_to_test = []
import glob
for filename in glob.iglob('./checkpoints/others' + '**/**', recursive=True):
     if filename.endswith(".params"):
         checkpoints_to_test.append(filename)
for filename in glob.iglob('./checkpoints/our4' + '**/**', recursive=True):
     if filename.endswith(".params"):
         checkpoints_to_test.append(filename)
checkpoints_to_test = sorted(checkpoints_to_test)
print(f'will test: {checkpoints_to_test}')

def nice_name_small(name):
    if 'Init' in name:
        return 'normal'
    elif 'standard' in name:
        return f'normal'
    elif 'FMS' in name:
        return 'ours'

plot_transmit_metric = {} 
plot_disagg_metric = {}
plot_transmit_metric['normal'] = []
plot_disagg_metric['ours'] = []
plot_transmit_metric['ours'] = []
plot_disagg_metric['normal'] = []

### Load data
for checkpoint_name in checkpoints_to_test:
    print(f'\n \033[0;32m  loading {checkpoint_name}', flush=True)
    print("\033[0m", flush=True)
    checkpoint = torch.load(checkpoint_name, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.train()

    if PLOT_CAR:
        if '2879' not in checkpoint_name:
            print('skip')
            continue
        else:
            full_chek = torch.load('./checkpoints/ours4_full/FMS-2879.FINAL_C10_NEG_128.params', map_location=torch.device(device))
            par_sel_size = 8400
            par_sel_frac = 0.001
            input_size = [3,32,32]
            par_sel=ParamSelector(model, par_sel_size ,par_sel_frac,sparse_grad=False,seed=98)
            grad_ex=GradientExtractor(model,par_sel).to(device)
            disaggregator, reconstructor = get_decoder(grad_ex, input_size, device)
            disaggregator.load_state_dict(full_chek['disaggregator_state_dict'])
            disaggregator.to(device)
            reconstructor.load_state_dict(full_chek['reconstructor_state_dict'])
            reconstructor.to(device)

            def STEAL(datapoints, labels, public_loss):
                bdW, _ = grad_ex(datapoints,labels,public_loss,flat_cat=True, single_grad=True,testing=True)
                disaggregator_o=disaggregator(bdW)
                reconstructor_o=reconstructor(disaggregator_o).reshape(-1, *input_size)
                return reconstructor_o
            print('ready to steal....')

            # SET UP CAR 
            CAR, CARLABEL = testd[6555]
            CAR = CAR.to(device)

    # get names of grads for later
    gradnames = []
    for m in model.named_parameters():
        gradnames.append(m[0])

    # transmit metric
    max_metric = -1
    for i in range(model.conv1.weight.shape[0]): # all kernels in first conv layer
        w = model.conv1.weight[i].ravel().abs() # absolute value
        biggest = w.max()
        sumrest = w.sum() - w.max()
        metric = (biggest / (sumrest + 1e-6)).item() # transmit-SNR
        max_metric = max(metric, max_metric)
    print(f'Transmit SNR: {max_metric:.2f}')
    plot_transmit_metric[nice_name_small(checkpoint_name)].append(max_metric)
    if max_metric > 1:
        print('Serious Issue') # never happens
        import code; code.interact(local=dict(globals(), **locals()))

    # disaggregation-SNR
    for bs in bszs:
        print(f'New batchsize: {bs}')
        
        if PLOT_CAR:
            print(f'Actually always doing car+63 random...')
            loader63 = torch.utils.data.DataLoader( testd, shuffle=True, num_workers=2, batch_size=63)
            data[bs] = loader63
        
        for j, (images, labels) in enumerate(data[bs]):
            datapoints = images.flatten(start_dim=0,end_dim=0).to(device)
            labels = labels.flatten(start_dim=0,end_dim=0).to(device)
            if PLOT_CAR:
                print(f'Adding the car! {j}')
                datapoints = torch.cat([datapoints, CAR.unsqueeze(0)])
                labels = torch.cat([labels, torch.tensor([CARLABEL]).to(device) ])

            model.zero_grad()
            loss = public_loss(model(datapoints),labels)

            # Per point grad
            model.zero_grad()
            loss = public_loss(model(datapoints),labels)
            jac = torch.eye(datapoints.shape[0]).to(device)
            batch_public_gradients = torch.autograd.grad(outputs=loss,inputs=model.parameters(),grad_outputs=jac,create_graph=False,is_grads_batched=True,allow_unused=False, retain_graph=False)
            example_grads = [g.view(g.shape[0], -1) for g in batch_public_gradients]
            # list of length 62 where each element is [#images, long flat grad]

            # get weight norms
            assert len(gradnames) == len(example_grads)
            weight_norms = []
            for i, grads in enumerate(example_grads):
                nm = gradnames[i]
                if nm == 'linear.weight' or ('.conv' in nm and '.weight' in nm):
                    norms = torch.linalg.vector_norm(grads, dim=1)
                    weight_norms.append(norms)

            weight_norms_stacked = torch.vstack(weight_norms) # [#layers, #images]
            wns = weight_norms_stacked
            s2n = wns.max(dim=1)[0] / ( wns.sum(dim=1) - wns.max(dim=1)[0] + 1e-6)
            #print(f's2n per layer: {s2n}')
            s2n_final = s2n.max().item()
            print(f's2n_final: {s2n_final}')
            plot_disagg_metric[nice_name_small(checkpoint_name)].append(s2n_final)

            if PLOT_CAR:
                if s2n_final > 0.62 and s2n_final < 0.72:
                    print('Now stealing:')
                    stolen = STEAL(datapoints, labels, public_loss)
                    saveimg(stolen, f'images/img{j}_s2n={s2n_final:.2f}', device)
                    print('SAVED!')
                    exit()

print('saving json')
with open("json/dsnr.json", "w") as fp:
    json.dump(plot_disagg_metric,fp) 
with open("json/transmit.json", "w") as fp:
    json.dump(plot_transmit_metric,fp) 
print('done')