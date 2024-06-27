import os
import torch
import torchvision
from PIL import Image
from parameters import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 

from torchvision import datasets
from torchvision.transforms import ToTensor
from copy import copy

import numpy as np

sys.path.insert(1, '../breaching')
import breaching
from breaching.cases.data.datasets_vision import TinyImageNet

class Dataset(torch.utils.data.Dataset):
    def __init__(self,tuple_list):
        self.samples=tuple_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        return self.samples[idx]

class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        #tensors = [ torch.from_numpy(t) for t in tensors ]
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]


def get_loader(train_data, test_data, batch_size, num_workers=8):
    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=num_workers),
        
        'test'  : torch.utils.data.DataLoader(test_data, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=num_workers),
    }
    return loaders['train'],loaders['test']

def save_img(norm_img, mu, sigma, path):
    A=numpy.uint8((norm_img*sigma+mu)*255)
    im = Image.fromarray(A)
    im.save(path)

class Loader(object):
    def __init__(self,datasets,**kwargs):
        kws=[copy(kwargs) for _ in range(len(kwargs['batch_size']))]
        for i,bs in enumerate(kwargs['batch_size']):
            kws[i]['batch_size']=bs
        self.loaders=[torch.utils.data.DataLoader(d,**kwargs) for kwargs,d in zip(kws,datasets)]
        self.lens=[len(d) for d in datasets]
        self.batch_sizes=[kwargs['batch_size'] for kwargs in kws]

    def __iter__(self):
        self.iters=[iter(l) for l in self.loaders]
        self.lefts=[l for l in self.lens]
        return self
    
    def __len__(self):
        return sum([int(self.lens[i]//self.batch_sizes[i]) for i in range(len(self.loaders))])

    def __next__(self):
        for i in range(len(self.loaders)):
            if self.lefts[i]<self.batch_sizes[i]:
                raise StopIteration
                break
        for i in range(len(self.loaders)):
            self.lefts[i]-=self.batch_sizes[i]
        nexts0=[None]*len(self.iters)
        nexts1=[None]*len(self.iters)
        for i,l in enumerate(self.iters):
            try:
                nexts0[i],nexts1[i]=next(l)
                if nexts0[i].shape[0] < self.batch_sizes[i]:
                    raise StopIteration
            except StopIteration:
                l = self.iters[i] = iter(self.loaders[i])
                nexts0[i],nexts1[i]=next(l)
                assert nexts0[i].shape[0] == self.batch_sizes[i], "batch size not matching or batch_sizes[i]>min(loader_sizes)"
        nexts0, nexts1 = (torch.cat(nexts0,dim=0),torch.cat(nexts1,dim=0))
        if nexts0.shape[1]==1:
            #TODO hack allowing to pass to the loader single binarized dataset
            nexts0,nexts1=(nexts0.squeeze(1),nexts1.squeeze(1))
        return nexts0, nexts1

def load_data(filename,**kwargs):
    print('start reading from file')
    dataset_train,dataset_test=torch.load(filename, map_location=torch.device('cpu'))
    print('finish reading from file')

    kwargs_train, kwargs_test = copy(kwargs), copy(kwargs)
    
    batch_size_train = kwargs_train['batch_size_train']
    del kwargs_train['batch_size_train']
    del kwargs_train['batch_size_test']
    kwargs_train['batch_size'] = batch_size_train 

    batch_size_test = kwargs_test['batch_size_test']
    del kwargs_test['batch_size_train']
    del kwargs_test['batch_size_test']
    kwargs_test['batch_size'] = batch_size_test

    loader_train=Loader(dataset_train,**kwargs_train)
    #TODO currently we shuffle the test loader as well
    loader_test=Loader(dataset_test,**kwargs_test)
    return loader_train,loader_test    

def datasets_TinyImageNet_rsz():
    transform_train = transforms.Compose(
    [transforms.ColorJitter(brightness= 0.2, contrast= 0.1, saturation=0.1, hue=0.05),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.RandomCrop(size=(32,32)),
     transforms.RandomChoice([
         transforms.RandomRotation((-5,5), fill=255),
         transforms.RandomRotation((85,95), fill=255),
         transforms.RandomRotation((175,185), fill=255),
         transforms.RandomRotation((-95,-85), fill=255)
     ]),
     transforms.ToTensor(),
     transforms.Normalize((0.4789886474609375, 0.4457630515098572, 0.3944724500179291), (0.27698642015457153, 0.2690644860267639, 0.2820819020271301))])

    transform_test = transforms.Compose(
    [transforms.RandomCrop(size=(32,32)),
     transforms.ToTensor(),
     transforms.Normalize((0.4789886474609375, 0.4457630515098572, 0.3944724500179291), (0.27698642015457153, 0.2690644860267639, 0.2820819020271301))])

    transform_lbl = transforms.Lambda(lambda y: y%10)
    
    trainset = TinyImageNet(root='../data', split='train', download=False, transform=transform_train, target_transform=transform_lbl)
    testset = TinyImageNet(root='../data', split='val', download=False, transform=transform_train, target_transform=transform_lbl)

    return trainset,testset

def datasets_TinyImageNet():
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
     transforms.Normalize((0.4789886474609375, 0.4457630515098572, 0.3944724500179291), (0.27698642015457153, 0.2690644860267639, 0.2820819020271301))])

    transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4789886474609375, 0.4457630515098572, 0.3944724500179291), (0.27698642015457153, 0.2690644860267639, 0.2820819020271301))])
    
    trainset = TinyImageNet(root='../data', split='train', download=False, transform=transform_train)
    testset = TinyImageNet(root='../data', split='val', download=False, transform=transform_test)

    return trainset,testset

def datasets_imagenet_full( ):
    from robustness import datasets
    from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy

 
    transform_train = transforms.Compose(
    [
     transforms.RandomCrop(size=(224,224), padding=None, pad_if_needed=True, fill=255, padding_mode='reflect'),
     transforms.ColorJitter(brightness= 0.2, contrast= 0.1, saturation=0.1, hue=0.05),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.RandomChoice([
         transforms.RandomRotation((-5,5), fill=255),
         transforms.RandomRotation((85,95), fill=255),
         transforms.RandomRotation((175,185), fill=255),
         transforms.RandomRotation((-95,-85), fill=255)
     ]),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    transform_test = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


    ds = datasets.ImageNet('../data/imagenet', transform_train=transform_train, transform_test=transform_test)

    return ds

def datasets_Isic2019():
    import urllib.request
    import hashlib
    import zipfile
    import os
    import shutil
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    download = True
    labelfile = '../data/train_isic2019_label.csv'
    try:
        checksum_train_y = hashlib.md5(open(labelfile,'rb').read()).hexdigest()
        if checksum_train_y == '2c02bdcc6e7f36d355f4f86b210595ae':
            download = False
    except:
        pass
    finally:
        if download:
            urllib.request.urlretrieve('https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv', labelfile)
            checksum_train_y = hashlib.md5(open(labelfile,'rb').read()).hexdigest()
            assert checksum_train_y == '2c02bdcc6e7f36d355f4f86b210595ae'

    labels = pd.read_csv(labelfile)

    trainfile = '../data/train_isic2019.zip'
    download = True
    try:
        checksum_train_x = hashlib.md5(open(trainfile,'rb').read()).hexdigest()
        if checksum_train_x == '0ecdc9554ef6273b04e59a0bc420ca9d':
            download = False
    except:
        pass
    finally:
        if download:
            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)
            url = 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip'
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url, trainfile, reporthook=t.update_to)

            checksum_train_x = hashlib.md5(open(trainfile,'rb').read()).hexdigest()
            assert checksum_train_x == '0ecdc9554ef6273b04e59a0bc420ca9d'
        if not os.path.isdir('../data/isic2019/'):
            with zipfile.ZipFile(trainfile, 'r') as zip_file:
                for member in tqdm( zip_file.namelist() ):
                    filename = os.path.basename(member)

                    # skip directories
                    if not filename:
                        continue

                    # copy file (taken from zipfile's extract)
                    source = zip_file.open(member)
                    if not member.endswith('.jpg'):
                        continue
                    row = labels[ labels['image'] == member.split('/')[1][:-4] ]
                    cl = np.argmax( row.iloc[0,1:] )
                    cl_name = row.columns[1+cl]
                    randint = np.random.randint(100)
                    #if randint < 80:
                    #    dataset = 'train'
                    #else:
                    #    dataset = 'test'
                    dataset = 'train'
                    os.makedirs(os.path.join('../data/isic2019/', dataset, cl_name), exist_ok=True)
                    target = open(os.path.join('../data/isic2019/', dataset, cl_name, filename), "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)

    transform = transforms.Compose(
    [transforms.Resize(350),
     transforms.RandomCrop(size=(224,224), padding=None, pad_if_needed=True, fill=255, padding_mode='reflect'),
     transforms.Resize(32),
     transforms.ToTensor(),
     transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])


    trainset = None
    testset = datasets.ImageFolder(root="../data/isic2019/train/",transform=transform)

    return trainset,testset

def datasets_Cifar100():
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
     transforms.Normalize((0.5071598291397095, 0.4866936206817627, 0.44120192527770996), (0.2673342823982239, 0.2564384639263153, 0.2761504650115967))])

    transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5071598291397095, 0.4866936206817627, 0.44120192527770996), (0.2673342823982239, 0.2564384639263153, 0.2761504650115967))])

    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)

    return trainset,testset

def datasets_Cifar10_2():
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
     transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])
    
    data = np.load('../data/cifar102_train.npz')
    trainset = CustomTensorDataset([data['images']/255.0, data['labels']], transform=transform_train )
    testset = CustomTensorDataset([data['images'], data['labels']], transform=transform_test )

    return trainset,testset

def datasets_Cifar10_1():
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
     transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])
    
    data = { 'images': np.load('../data/cifar10.1_v6_data.npy'),  'labels': np.load('../data/cifar10.1_v6_labels.npy').astype('int64') }
    trainset = CustomTensorDataset([data['images']/255.0, data['labels']], transform=transform_train )
    testset = CustomTensorDataset([data['images'], data['labels']], transform=transform_test )

    return trainset,testset

def datasets_Cifar10_C( C, sev ):
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
     transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])

    f = 10000
    data = { 'images': np.load(f'../data/CIFAR-10-C/{C}.npy')[f*(sev-1):f*sev],  'labels': np.load('../data/CIFAR-10-C/labels.npy').astype('int64')[f*(sev-1):f*sev] }
    trainset = CustomTensorDataset([data['images']/255.0, data['labels']], transform=transform_train )
    testset = CustomTensorDataset([data['images'], data['labels']], transform=transform_test )

    return trainset,testset

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
     transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

    return trainset,testset

