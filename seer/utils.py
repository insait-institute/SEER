from scipy.optimize import minimize_scalar
import os
from os.path import exists
import torch
from neptune.new.types import File
from PIL import Image
from matplotlib import cm
import numpy
import time

#xs_std,xs_mean = torch.std_mean(xs,dim=0)
#ys_std,ys_mean = torch.std_mean(ys,dim=0)
## TODO get the mean of the entire dataset - center the data to avoid the so called "catastrophic cancellation" - thats possible only for the inputs
#torch.corrcoef # for this fn the shape should be (vars,batch), output shape is (vars,vars)
#x|w ->  |xx|xw|
#        |wx|ww|
#xx - unneeded
#xw - for positive samples optimize absolute value to be ones matrix; for the abs value: loss that is zero at 1, high at 0 and high at +inf max(-log(x),(x-1)^2); (-ln(abs(x)))^2; (-ln(abs(1-x)))^2; x^2; (1-abs(x))^2|>sum
#xw - for negative samples optimize to be zero
#ww - should we optim that? 

#def testing():
#    from data import *
#    _,_,d,_=torch.load('bim.mnist.pt')
#    i=d[13603][0][1][0]
#    im=sigma*i+mu
#    im
#    t=torch.arange(1,20)/20/4;t
#    for i,tgt in enumerate(t):
#        obj = lambda a: (pw(im,a).mean()-tgt)**2
#        res=minimize_scalar(obj,bracket=(1/33,33))
#        print(i,res.x)
#        save_img(pw(im,res.x),f"images/pw{i}.jpeg")
#    save_img(im,f"images/pw_original.jpeg")

def pw(x,a):
        return 1-(1-x)**a

def solve_bri_pow(img,tgt):
    # solves the pw objective for a given image brightness, requires a normalized image in the [0,1] interval and e target mean brightness
    obj = lambda a: (pw(img,a).mean()-tgt)**2
    res=minimize_scalar(obj,bracket=(1/33,33))
    return pw(img,res.x)

def cross_correlation_matrix(xs,ys,skip_constant=False): #xs.shape = (batch,x_vars); ys.shape = (batch,y_vars); output shape (x_vars,y_vars)# batch dim is used to empirically estimate the random variable stats
    if skip_constant:
        ncxi=~torch.all(xs == xs[0,:],dim=0)
        ncyi=~torch.all(ys == ys[0,:],dim=0)
        xsnc=xs[:,ncxi]
        ysnc=ys[:,ncyi]
        numxsnc=xsnc.shape[1]
        numysnc=ysnc.shape[1]
        #if torch.rand(1).item()<1:
        #    #print('nonconstant input and output: ',xsnc.shape[1],'/',xs.shape[1],':',xsnc.shape[1]/xs.shape[1],' | ',ysnc.shape[1],'/',ys.shape[1],':',ysnc.shape[1]/ys.shape[1])
        return torch.corrcoef(torch.cat((xsnc,ysnc),dim=1).T)[:numxsnc,numxsnc:]
    else:
        numxs=xs.shape[1]
        return torch.corrcoef(torch.cat((xs,ys),dim=1).T)[:numxs,numxs:]


class PartitionsNorms():
    def __init__(self,seed):
        self.seed=seed
        self.gen=torch.Generator()

    def norms(self,t,num_partitions,partition_sizes):
        self.gen.manual_seed(self.seed)
        #t.shape=(batch,data_dims*)
        norms=[]
        for npart,nels in zip(num_partitions,partition_sizes):
            tf=t.flatten(start_dim=1)
            numel=tf.shape[1]
            nummx=numel//nels
            n=min(nummx,npart)
            gthnum=n*nels
            rp=torch.randperm(numel,generator=self.gen)[:gthnum]
            #tfrp=torch.gather(tf,1,rp.unsqueeze(0).expand(tf.shape(0),gthnum))
            tfnels=tf[:,rp].reshape(tf.shape[0],n,nels)
            n_norms=torch.linalg.norm(tfnels,dim=-1)
            norms.append(n_norms)
        return torch.cat(norms,dim=1)

def boltz(sum,num,temp=5):
    r=torch.rand(num)
    e=torch.exp(-r*temp)
    b=((sum-num)*e/e.sum()).floor().int()+1
    d=sum-b.sum()
    h=b.argmax()
    b[h]=b[h]+d
    assert b.sum()==sum
    assert b.min()>=1
    return b

@torch.jit.script
def preaverage(ccm: torch.Tensor):
    return 5*(1-ccm.abs()).pow(0.2)

def log_heatmap(t,args,lbl):
    # t is a normalized_0_1_tensor_2d
    im = Image.fromarray(numpy.uint8(cm.Greys(t)*255))
    if not exists(args.res_path + '/' + 'tmp'):
        os.makedirs(args.res_path + '/' + 'tmp')
    filename=args.res_path+'/'+f"heatmaps/heatmap_{lbl}_{time.time()}.png"
    im.save(filename)
    args.neptune[f"{lbl}"].log(File(filename))

@torch.jit.script
def pmean(t: torch.Tensor, p: float, dim: int):
    pnorm=torch.linalg.norm(t,ord=p,dim=dim)
    n=t.shape[dim]
    return pnorm*(n**(-1/p))

def map(fn, inp, outp):
    for idx in range(len(inp)):
        outp.append(fn(inp[idx]))

def reg(var_name):
    #features.append(var_name)
    return var_name

def trav(args,pref,d,leaf_fn):
    if type(d)!=type({}):
        leaf_fn(args,pref,d)
    else:
        for k in d:
            trav(args,f"{pref}/{k}",d[k],leaf_fn)

def log_nept(args,pref,leaf):
    if 'log' in pref or 'opt' in pref:
        if args.neptune is not None:
            args.neptune[f"{pref}"].log(leaf);
        print(f"{pref}",': ',leaf)

def log_stdout(args,pref,leaf):
    if 'opt' in pref:
        print(f"{pref} : {leaf.item() if type(leaf)==torch.Tensor else leaf}")
    if 'log' in pref:
        if leaf.numel()>1:
            print('WARNING: log automean:',pref,leaf.shape)
            print(f"{pref} : {leaf.mean().item() if type(leaf)==torch.Tensor else leaf}")
        else:
            print(f"{pref} : {leaf.item() if type(leaf)==torch.Tensor else leaf}")


def histograms(xs):
    h=len(xs)
    figure, ax = plt.subplots(h, 1, figsize=(10, 10),squeeze=False)
    for i,x in enumerate(xs):
        ax[i, 0].hist(x,bins=100)
        #figure.show()
    return figure

import torch
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def go1(data=torch.rand(66,66)):
    fig, ax = plt.subplots()

    im, cbar = heatmap(data.numpy(), range(data.shape[0]), range(data.shape[1]), ax=ax,
                       cmap="YlGn", cbarlabel="harvest [t/year]")
    #texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    return fig

def go():
    fig, ax = plt.subplots()

    im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                       cmap="YlGn", cbarlabel="harvest [t/year]")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.show()


#vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
#              "potato", "wheat", "barley"]
#farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
#           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
#
#harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
#
#go1(torch.rand(66,66))

_NO_DEFAULT = object()

class TupleDictView:
    __slots__ = ('_dict',)

    def __init__(self, d, /):
        self._dict = d

    def __getitem__(self, key, /):
        if not isinstance(key, tuple):
            key = (key,)
        elif not key:
            return self
        sub = self._dict
        for k in key[:-1]:
            sub = sub[k]
            if not isinstance(sub, dict):
                raise TypeError('too many indices')
        sub = sub[key[-1]]
        if isinstance(sub, dict):
            sub = TupleDictView(sub)
        return sub

    def __setitem__(self, key, value, /):
        if isinstance(value, TupleDictView):
            value = value._dict
        if not isinstance(key, tuple):
            self._dict[key] = value
            return
        if not key:
            raise TypeError('empty index')
        sub = self._dict
        for k in key[:-1]:
            try:
                sub = sub[k]
            except KeyError:
                sub[k] = sub = {}
            if not isinstance(sub, dict):
                raise TypeError('not a dictionary')
        sub[key[-1]] = value

    def __delitem__(self, key, /):
        if not isinstance(key, tuple):
            del self._dict[key]
            return
        if not key:
            raise TypeError('empty index')
        sub = self._dict
        for k in key[:-1]:
            try:
                sub = sub[k]
            except KeyError:
                return
            if not isinstance(sub, dict):
                raise TypeError('not a dictionary')
        del sub[key[-1]]

    def __contains__(self, key, /):
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __eq__(self, other, /):
        if isinstance(other, TupleDictView):
            return self._dict == other._dict
        if isinstance(other, dict):
            return self._dict == other
        return NotImplemented

    def get(self, key, default=None, /):
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key, default=None, /):
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default

    def pop(self, key, default=_NO_DEFAULT, /):
        try:
            r = self[key]
        except KeyError:
            if default is _NO_DEFAULT:
                raise
            return default
        del self[key]
        return r

    def update(self, other, /):
        if isinstance(other, TupleDictView):
            other = other._dict
        for k,v in other.items():
            if not isinstance(v, dict):
                self._dict[k] = v
                continue
            try:
                v2 = self._dict[k]
            except KeyError:
                self._dict[k] = v
                continue
            if not isinstance(v2, dict):
                self._dict[k] = v
                continue
            TupleDictView(v2).update(v)

    def items(self, /):
        for k,v in self._dict.items():
            if not isinstance(v, dict):
                yield ((k,),v)
                continue
            for subk,subv in TupleDictView(v).items():
                yield (k,) + subk, subv

    def keys(self):
        for k,v in self.items():
            yield k

    __iter__ = keys

    def values(self):
        for k,v in self.items():
            yield v

    def __str__(self):
        return str(self._dict)

    def __repr__(self):
        return repr(self._dict)
     
def ref(t,switch_ref):
    if switch_ref:
        print(t[tuple(0 for _ in range(len(t.shape)))].item())

def tick(state, t=None):
    last_now,switch_ref=state
    if t is not None:
        ref(t,switch_ref)
    now=time.time()
    ret=last_now[0]
    last_now[0]=now
    return now-ret

@torch.jit.script
def weight_normalize(inp: torch.Tensor, ws: torch.Tensor):
    assert inp.dim() == ws.dim() == 1
    return inp*ws.numel()*ws/ws.sum()

def num_params(ge):
    if ge.par_sel is not None:
        return ge.par_sel.num_par
    else:
        c=0
        for p in ge.public_model.parameters():
            c+=p.numel()
        print('num_params: ',c)
        return c

def rat_sched(epoch,x_1,x_end,y_1,y_end):
    if epoch >= x_end:
        return y_end
    else:
        return y_end*(epoch-x_1)/(x_end-x_1)+y_1*(x_end-epoch)/(x_end-x_1)

def property_scores(datapoints,prop,labels=None):
    # higher is better
    if prop == 'bright':
        scores = datapoints.mean((1,2,3))
    elif prop == 'dark':
        scores = datapoints.mean((1,2,3))
        scores = -scores
    elif prop == 'red':
        scores = datapoints.mean((2,3)) * torch.tensor([2.,-1.,-1.],device=datapoints.device).unsqueeze(0)
        scores = scores.mean((1,))
    elif prop == 'green':
        scores = datapoints.mean((2,3)) * torch.tensor([-1.,2.,-1.],device=datapoints.device).unsqueeze(0)
        scores = scores.mean((1,))
    elif prop == 'blue':
        scores = datapoints.mean((2,3)) * torch.tensor([-1.,-1.,2.],device=datapoints.device).unsqueeze(0)
        scores = scores.mean((1,))
    elif prop == 'hedge':
        gray = datapoints.mean(1)
        scores = (gray[:,1:,:] - gray[:,:-1,:]).mean((1,2))
    elif prop == 'vedge':
        gray = datapoints.mean(1)
        scores = (gray[:,:,1:] - gray[:,:,:-1]).mean((1,2))
    elif prop == 'vedge+green':
        scores_green = datapoints.mean((2,3)) * torch.tensor([-1.,2.,-1.],device=datapoints.device).unsqueeze(0)
        scores_green = scores_green.mean((1,))
        gray = datapoints.mean(1)
        scores_vedge = (gray[:,:,1:] - gray[:,:,:-1]).mean((1,2))
        scores = scores_green * 0.5 + scores_vedge * 0.5
    elif prop == 'red+car':
        scores = datapoints.mean((2,3)) * torch.tensor([2.,-1.,-1.],device=datapoints.device).unsqueeze(0)
        scores = scores.mean((1,))
        scores[labels != 1] = -1e+6
    elif prop == 'rand_conv':
        weight = torch.tensor([[[[ 0.1410,  0.1441,  0.0390],
                                 [-0.1475,  0.1789,  0.0264],
                                 [-0.1550, -0.0560,  0.1355]],

                                [[-0.1293,  0.0612,  0.0567],
                                 [ 0.1918,  0.0576,  0.1709],
                                 [-0.0039,  0.0088, -0.0689]],

                                [[-0.1303,  0.0727, -0.0907],
                                 [-0.1532, -0.0025,  0.1554],
                                 [ 0.1820,  0.0876, -0.0287]]]],device='cuda')
        weight -= weight.mean()
        convolved=torch.nn.functional.conv2d(datapoints, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        scores = convolved.mean((1,2,3))
    return scores
