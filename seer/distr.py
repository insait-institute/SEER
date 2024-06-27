import torch
from utils import *

normal=torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
batch_norm=torch.nn.BatchNorm1d(1, eps=1e-05, affine=False, track_running_stats=False, device='cuda', dtype=None)

def tweak(dms: torch.Tensor, pos: bool, prop: str, thresh: float, disag_size: int, bn: torch.nn.modules.batchnorm.BatchNorm1d, N: torch.distributions.normal.Normal):
    with torch.no_grad():
        #assert dms.device.type=='cpu', "cpu!"
        assert dms.shape[0]==dms.shape[1]==1, "unsqueeze!"
        assert prop in ['bright','dark','red'], "works only for those currently"
        # we assume the brightness distribution is normal
        mn=dms.mean()
        sd=dms.std(unbiased=False)
        if thresh is not None:
            th=torch.tensor(thresh)
            disag_size=1/(1-N.cdf(th))
        else:
            th=N.icdf(torch.tensor(1-1/disag_size))
        th=th.cuda()
        thl=(th-N.icdf(torch.tensor(1-2/disag_size)).cuda())/2
        tw=bn(dms)
        if prop in ['bright','red','dark']:
            assert (tw!=tw.sort().values).sum()==0, "sort!"
        if not pos:
            if torch.rand(1)<0.5:
                top1_tgt=-N.sample().abs().cuda()*thl+th
                tw[0,0,-1]=top1_tgt
                tw=bn(tw).sort().values
        for nmi in range(3000):
            if (tw[0,0,-1]>th)==pos and tw[0,0,-(pos+1)]<th:
                if prop in ['bright','red']:
                    return tw*sd+mn
                elif prop == 'dark':
                    tw=(-tw)#.flip(dims=(-1,))
                    return tw*sd+mn
            if pos:
                top1_tgt=N.sample().abs().cuda()*thl+th
                top2_tgt=-N.sample().abs().cuda()*thl+th
                tw[0,0,-1]=top1_tgt
                tw[0,0,-(pos+1)]=top2_tgt
            else:
                top1_tgt=-N.sample().abs().cuda()*thl+th
                tw[0,0,-1]=top1_tgt
            tw=bn(tw).sort().values
        print('SKIPPING BATCH')
        return tw*sd+mn
        


def count(dms: torch.Tensor, pos: bool, prop: str, thresh: float, disag_size: int, bn: torch.nn.modules.batchnorm.BatchNorm1d, N: torch.distributions.normal.Normal):
    with torch.no_grad():
        #assert dms.device.type=='cpu', "cpu!"
        assert dms.shape[0]==dms.shape[1]==1, "unsqueeze!"
        assert prop in ['bright','dark','red'], "works only for those currently"
        # we assume the brightness distribution is normal
        if thresh is not None:
            th=torch.tensor(thresh)
            disag_size=1/(1-N.cdf(th))
        else:
            th=N.icdf(torch.tensor(1-1/disag_size))
        th=th.cuda()
        thl=(th-N.icdf(torch.tensor(1-2/disag_size)).cuda())/2
        tw=bn(dms)
        if prop in ['bright','red','dark']:
            assert (tw!=tw.sort().values).sum()==0, "sort!"
        num_pos=(tw[0,0,:]>th).nonzero().numel()
        return num_pos

import numpy as np
from scipy.interpolate import interp1d

def estimate_cdf(dataset, prop, feature, batch_size, num_samples,bn):
    dl=torch.utils.data.DataLoader( dataset, batch_size=batch_size, num_workers=4,sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=num_samples*batch_size) )

    # Sample from the distribution
    samples=[]
    print('start')
    for i,b in enumerate(dl):
        datapoints=b[0]
        scores = property_scores(datapoints,prop)
        nrmns=bn(scores.unsqueeze(0).unsqueeze(0))
        samples.append(feature(nrmns).item())
        if torch.rand(1)<0.01:
            print(i)
    assert len(samples)==num_samples, "unexpected num samles"

    # Sort the samples in ascending order
    sorted_samples = np.sort(samples)
    
    # Calculate the empirical probabilities for each sample
    empirical_probs = np.arange(1, num_samples+1) / num_samples
    
    # Create a lookup table that maps values to empirical probabilities
    lookup_table = interp1d(sorted_samples, empirical_probs, kind='linear', bounds_error=False, fill_value=(0, 1))
    
    # Define a function that estimates the CDF for a given value
    def cdf(value):
        return lookup_table(value)
    
    return cdf

def opt_thresh(cdf1, cdf2, num_clients):
    # th argmax -(1-cdf1(th))*(cdf1(th)**(num_clients-1))
    def obj(th):
        return -(1-cdf1(th))*cdf2(th)*(cdf1(th)**(num_clients-1))
    from scipy import optimize
    argmin_th=optimize.golden(obj,brack=(0,6))
    return argmin_th

def compute_thresh(dataset,prop,batch_size,num_clients,num_samples,bn):
    def feat1(scores):
        return torch.topk(scores,k=1).values.squeeze()
    def feat2(scores):
        return torch.topk(scores,k=2).values.squeeze()[1]

    cdf1=estimate_cdf(dataset, prop, feat1, batch_size, num_samples,bn)
    cdf2=estimate_cdf(dataset, prop, feat2, batch_size, num_samples,bn)
    th=opt_thresh(cdf1,cdf2,num_clients)
    return th
