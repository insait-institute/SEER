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
import breaching

class GradientExtractor(torch.nn.Module):
    def __init__(self, public_model,par_sel=None):
        super(GradientExtractor, self).__init__()
        self.public_model = public_model
        shapes = [p.numel() for p in self.public_model.parameters()]
        self.output_dim=sum(shapes)
        self.par_sel=None
        if par_sel is not None:
            self.par_sel=par_sel

    def flatten_gradient(self,batch_dW,W):
        batch_dWflat=torch.cat([l.flatten(start_dim=1) for l in batch_dW],dim=1)
        Wflat=torch.cat([l.flatten() for l in W])
        return batch_dWflat,Wflat
        
    def forward(self,x,y,pub_loss_fn,flat_cat=True, single_grad=False,testing=False,jac=None):
        assert single_grad==True or jac is not None, "pointwise grad requires jac"
        batch_public_loss=pub_loss_fn(self.public_model(x),y) # batch_public_loss shape is (batch_size,)
        #TODO eye is generated each time and hardcoded device
        if single_grad:
            batch_public_gradients = torch.autograd.grad(outputs=batch_public_loss.sum(),inputs=self.public_model.parameters(),allow_unused=False) # TODO allow_unused=True ?
            batch_public_gradients = [g.unsqueeze(0) for g in batch_public_gradients]
            single_grad=False
        else:
            if not testing:
                batch_public_gradients = torch.autograd.grad(outputs=batch_public_loss,inputs=self.public_model.parameters(),grad_outputs=jac,create_graph=True,is_grads_batched=True,allow_unused=False) # TODO allow_unused=True ?
            else:
                batch_public_gradients = torch.autograd.grad(outputs=batch_public_loss,inputs=self.public_model.parameters(),grad_outputs=jac,is_grads_batched=True,allow_unused=False) # TODO allow_unused=True ?
        ##public_loss.backward(inputs=list(self.public_model.parameters()),create_graph=True)
        #grads = [p.grad_sample.flatten(start_dim=1) for p in self.public_model.parameters()]
        #grad = torch.cat(grads,dim=1)
        ##grads = [p.grad.flatten() for p in self.public_model.parameters()]
        ##grad = torch.cat(grads,dim=0).unsqueeze(0)
        if flat_cat:
            bdW,W = self.flatten_gradient(batch_public_gradients,self.public_model.parameters())
        else:
            bdW=batch_public_gradients
            W=self.public_model.parameters()
        if self.par_sel is not None:
            return self.par_sel(bdW,single_grad,flat_cat),W
        else:
            return bdW,W

class ParamSelector(torch.nn.Module):
    def __init__(self,public_model,sz,frac,sparse_grad,seed):
        super(ParamSelector,self).__init__()
        #self.seed=seed
        gen=torch.Generator()
        gen.manual_seed(seed)
        self.sz=sz
        self.frac=frac
        self.sparse_grad=sparse_grad
        assert self.sparse_grad == False, "currently hardcoded in the extract function, due to @jit, go change it there"
        self.rps=[torch.randperm(p.numel(),generator=gen)[:max(self.sz,int(round(p.numel()*self.frac)))].to(p.device).sort().values for n,p in public_model.named_parameters()]
        numels=[p.numel() for p in public_model.parameters()]
        offsets=[0]+[i.item() for i in torch.tensor(numels).cumsum(0)[:-1]]
        rps_off=[rp+o for rp,o in zip(self.rps,offsets)]
        self.rp_cat=torch.cat(rps_off,dim=0)
        with torch.no_grad():
            total=0
            used=0
            print('used/total\t\tpercent\t\tname\t\tshape')
            for rp,(n,p) in zip(self.rps,public_model.named_parameters()):
                u=rp.numel()
                t=p.numel()
                print(f"{u}/{t}\t\t{round(100*u/t,ndigits=1)}%\t\t{n}\t\t({','.join(str(sh) for sh in list(p.shape))})")
                used+=u
                total+=t
            self.frac_tot=used/total
            print('summary')
            print(f'{used}/{total}\t\t{round(100*used/total,ndigits=1)}%')
            print()
            self.num_par=used


    @staticmethod
    @torch.jit.script
    def extract(p: torch.Tensor,rp: torch.Tensor,single_grad: bool=False):
        if not single_grad:
            if p[0].numel()==rp.numel():
                return p.flatten(start_dim=1)
            else:
                return torch.gather(input=p.flatten(start_dim=1),dim=1,index=rp.expand(p.shape[0],-1),sparse_grad=False)
        else:
            if p.numel()==rp.numel():
                return p.flatten()
            else:
                return p.flatten()[rp]

    def forward(self,params,single_grad,flat_cat):
        #for rp,p in zip(self.rps,params):
        #    print(p.shape,'\t',p.numel(),'\t',torch.max(rp).item(),'\t',torch.min(rp).item())
        #if not testing:
        #    return [torch.gather(input=p.flatten(start_dim=1),dim=1,index=rp.expand(p.shape[0],-1),sparse_grad=self.sparse_grad) for rp,p in zip(self.rps,params)]
        #else:
        #    return [p.flatten()[rp] for rp,p in zip(self.rps,params)]
        if flat_cat:
            return self.extract(p=params,rp=self.rp_cat,single_grad=single_grad)
        else:
            return [self.extract(p=p,rp=rp,single_grad=single_grad) for p,rp in zip(params,self.rps)]
        #for rp,p in zip(self.rps,params):
        #    yield torch.gather(input=p.reshape(-1),dim=0,index=rp,sparse_grad=self.sparse_grad) 

def get_decoder(grad_ex, input_size, device):
    def num_params(ge):
        if ge.par_sel is not None:
            return ge.par_sel.num_par
        else:
            c=0
            for p in ge.public_model.parameters():
                c+=p.numel()
            print('num_params: ',c)
            return c
    disaggregator_id=num_params(grad_ex)
    disaggregator_od=num_params(grad_ex)
    reconstructor_id=disaggregator_od
    reconstructor_od=(input_size[0])*(input_size[1])*(input_size[2])
    disaggregator=torch.nn.Identity().to(device)
    reconstructor=torch.nn.Linear(reconstructor_id,reconstructor_od,bias=True).to(device)
    return disaggregator, reconstructor
 

def denormalize(img, device):
    mean = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
    std = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)
    dm = torch.as_tensor(mean, device=device)[None, :, None, None]
    ds = torch.as_tensor(std, device=device)[None, :, None, None]
    img = torch.clamp(img * ds + dm, 0, 1)
    return img

def saveimg(img, tag, device):
    img = denormalize(img,device)
    torchvision.utils.save_image(img, f'{tag}.png')
    
def datasets_Cifar10():
    transform_train = transforms.Compose(
    [transforms.ColorJitter(brightness= 0.2, contrast= 0.1, saturation=0.1, hue=0.05),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.RandomChoice([
         transforms.RandomRotation((-5,5), fill=255),
         transforms.RandomRotation((85,95), fill=255),
         transforms.RandomRotation((175,185), fill=255),
         transforms.RandomRotation((-95,-85), fill=255)
     ]),
     transforms.ToTensor(),
     transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])

    transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))
     ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

    return trainset,testset
