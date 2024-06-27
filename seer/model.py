import torch
import torchvision
from torch import nn
from data import *
from parameters import *
from resnet import *
from utils import num_params

class DeconvDecoder(nn.Module):
    def __init__(self, g_dim, out_size):
        super(DeconvDecoder, self).__init__()

        self.out_size = out_size
        assert out_size[1] == out_size[2], "Expected square images"

        self.project_0_1 = nn.Linear( g_dim, 800, bias=False )
        self.project_0_2 = nn.Linear( 800, 128 * 3 * 3, bias=True )
        
        self.deconv1 = torch.nn.ConvTranspose2d( 128, 64, 3, 2 )
        self.project_1_1 = nn.Linear( g_dim, 300, bias=False )
        self.project_1_2 = nn.Linear( 300, 64 * 7 * 7, bias=True )

        self.deconv2 = torch.nn.ConvTranspose2d( 64, 32, 3, 2 )
        self.project_2_1 = nn.Linear( g_dim, 300, bias=False )
        self.project_2_2 = nn.Linear( 300, 32 * 15 * 15, bias=True )
        
        if out_size[1] == 64:
            
            self.deconv3 = torch.nn.ConvTranspose2d( 32, 32, 4, 2 )
            self.project_3_1 = nn.Linear( g_dim, 300, bias=False )
            self.project_3_2 = nn.Linear( 300, 32 * 32 * 32, bias=True )
            
            self.deconv4 = torch.nn.ConvTranspose2d( 32, 3, 2, 2 )
            self.project_4_1 = nn.Linear( g_dim, 500, bias=False )
            self.project_4_2 = nn.Linear( 500, 3 * 64 * 64, bias=True )
        
        elif out_size[1] == 32:
            
            self.deconv3 = torch.nn.ConvTranspose2d( 32, 3, 4, 2 )
            self.project_3_1 = nn.Linear( g_dim, 500, bias=False )
            self.project_3_2 = nn.Linear( 500, 3 * 32 * 32, bias=True )
        
        else:
            assert False, "Not supported"

    def forward(self, x):

        o0 = self.project_0_2( self.project_0_1( x ) ).reshape( x.shape[0], 128, 3, 3 )

        o1 = self.deconv1( o0 )
        o1 += self.project_1_2( self.project_1_1( x ) ).reshape( x.shape[0], 64, 7, 7 )

        o2 = self.deconv2( o1 )
        o2 += self.project_2_2( self.project_2_1( x ) ).reshape( x.shape[0], 32, 15, 15 )
        
        if self.out_size[1] == 64:
            o3 = self.deconv3( o2 )
            o3 += self.project_3_2( self.project_3_1( x ) ).reshape( x.shape[0], 32, 32, 32 )

            o4 = self.deconv4( o3 )
            o4 += self.project_4_2( self.project_4_1( x ) ).reshape( x.shape[0], 3, 64, 64 )
            
            return o4
        else:
            o3 = self.deconv3( o2 )
            o3 += self.project_3_2( self.project_3_1( x ) ).reshape( x.shape[0], 3, 32, 32 )
            
            return o3

class CNN(nn.Module):
    def __init__(self, act):
        super(CNN, self).__init__()
        act = getattr( nn, act )
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            act(),                      
            #nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 16, 5, 1, 2),     
            act(),                      
            #nn.MaxPool2d(2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(16, 16, 5, 1, 2),     
            act(),                      
            #nn.MaxPool2d(2),                
        )
        self.conv4 = nn.Sequential(         
            nn.Conv2d(16, 16, 5, 1, 2),     
            act(),                      
            #nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(16*28*28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output

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
        
    def forward(self,x,y,pub_loss_fn,flat_cat=True, single_grad=False,testing=False,jac=None,par_sel=True):
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
        if (self.par_sel is not None) and par_sel:
            return self.par_sel(bdW,single_grad,flat_cat),W
        else:
            return bdW,W

class GradientSelector(torch.nn.Module):
    def __init__(self, probs,indices=None,public_model=None):
        super(GradientSelector,self).__init__()
        if indices is not None:
            self.indices=indices
        else:
            rs=[torch.rand_like(p) for p in public_model.parameters()]
            self.indices = [r <= torch.topk(r.flatten(),largest=False,k=int(pr*r.numel())).values[-1] for r,pr in zip(rs,probs)]
        self.output_dims=[i.nonzero().shape[0] for i in self.indices]
        for i, idx in enumerate( self.indices ):
            self.register_buffer(f'indices_{i}', idx)


    def load_state_dict(self, state_dict, strict=True, device=None):
        super().load_state_dict(state_dict, strict)
        i = 0
        if device is None:
            device = 'cuda' if self.indices[0].is_cuda else 'cpu'
        self.indices = []
        self.output_dims=[]
        while True:
            k = f'indices_{i}'
            if not k in state_dict:
                break
            self.indices.append( state_dict[k].to(device) )
            self.output_dims.append( state_dict[k].nonzero().shape[0] )
            i += 1

    def forward(self,params,batched=True):
        if batched:
            return [p[:,i] for p,i in zip(params,self.indices)]
        if not batched:
            return [p[i] for p,i in zip(params,self.indices)]



class Binarizer(torch.nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Binarizer, self).__init__()
        #currently the number of bins is hardcoded equal to 1
        if len(in_dims)==len(out_dims)==1:
            self.as_tuple=False
            self.E=torch.nn.Linear(weight_dim,bin_size,bias=False)
        else:
            self.as_tuple=True
            self.E=torch.nn.ModuleList([torch.nn.ModuleList([torch.nn.Linear(i_d,o_d) for i_d in in_dims]) for o_d in out_dims])

    def forward(self,x):
        if not self.as_tuple:
            return self.E(x)
        else:
            #flat_cat=torch.cat([torch.flatten(xi,start_dim=1) for xi in x],dim=1)
            return [torch.stack([ei(xi) for ei,xi in zip(eo,x)],dim=2).sum(dim=2) for eo in self.E]

class WeightProjector(torch.nn.Module):
    def __init__(self, proj_size, weight_dim):
        super(WeightProjector, self).__init__()
        self.S = torch.nn.Sequential(
            torch.nn.Linear(weight_dim,proj_size),
            torch.nn.Softplus()
            )

    def forward(self,x):
        return self.S(x) 

class Decoder(torch.nn.Module):
    #TODO more careful architecture; binarizer and projection are linear already, no need for linear again (or maybe it is needed, because the intermediate representation has a different objective - to vanish to zero for some entries; some of the following might be useful: resudual connections, explicit higher order functions - multiplication, exponentiation, etc. after exponentiation its easier to approximate/"learn" multiplication/division; maybe use fomething more differentiable for activations as softplus - as leakage is not handcrafted but trained
    def __init__(self, bin_size, proj_size):
        super(Decoder, self).__init__()
        h_dim=bin_size+proj_size
        self.h2_dim=h_dim//2
        self.pipeline = nn.Sequential(
            torch.nn.Linear(h_dim,self.h2_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h2_dim, self.h2_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h2_dim, self.h2_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h2_dim, self.h2_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h2_dim,28*28),
        )
        
    def forward(self,B,Wpe):
        BWpe=torch.cat((B,Wpe),dim=1)
        reconstruction = self.pipeline(BWpe)
        return reconstruction

class Losses():
    def __init__(self, params, args):
        #self.scale={k:torch.tensor(scale[k],device=device) for k in scale.keys()}
        # out of a batch of B elements we have B neg samples, B pos samples and B*(B-1) ratios, which we limit to B*num_ratios
        # to normalize the reconstruction and disaggregation losses we scale the recon. loss respectively by num_ratios+1, we dont know how exactly we should scale the neg loss bu we will leave it there hoping to improve convergence :)
        params = list( params )
        self.args = args
        self.device = self.args.device
        self.public_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.pos_g_idx = []
        self.neg_g_idx = []
        grad_sizes = [ ps.shape for ps in params ]
        with torch.no_grad():
            self.W_sizes = [ ps.pow(2).mean() for ps in params ]

    def noise_to_signal(self,n,s):
        r=torch.clamp((n**2)/(s**2 + self.args.eps),max=self.args.clamp)
        return r.mean()

    def positive(self,p,pb,w,decoder):
        reconstruction = decoder(pb,w)
        loss = torch.nn.functional.mse_loss(reconstruction, p.flatten(start_dim=1),reduction='mean')
        return loss
        
    def negative(self,nb): # do not use it, otherwise gradients may became unusably small
        loss = (nb**2).mean() # mse of recon. and zero i.e. train recon. to be zero # instead of loss_func(reconstruction, torch.zeros_like(reconstruction))
        return loss
    
    def ratio(self,nB,pB): #batched version of noise_to_signal #TODO currently quadratic wrt the batch_size, if used with large batch needs to be changed
        loss = torch.clamp(torch.einsum('bi,ci->bci',nB**2,1/pB**2),max=self.args.clamp)
        return loss.mean()

    def loss_ratio(self,ng,pg):
        ratio=[]
        for n,p in zip(ng,pg):
            ratio.append(self.ratio(n,p))
        m = sum(ratio) / len(ratio)
        return m

    def std_ratio(self,nB,pB): #batched version of noise_to_signal #TODO currently quadratic wrt the batch_size, if used with large batch needs to be changed
        assert false, "math correct std; jensen(convex/concave); mean.pow(1/2); log"
        nstd=torch.std(nB,dim=0)
        pstd=torch.std(pB,dim=0)
        loss = torch.clamp(nstd/pstd,max=self.args.clamp)
        return loss.mean()

    def std(self,B):
        Bf=B.flatten(start_dim=1)
        us=torch.std(Bf,dim=0)
        #return torch.linalg.norm(us,ord=1)
        return us.mean()

    def tot_std(self,B):
        std=[]
        for g in B:
            std.append(self.std(g))
        #m = torch.linalg.norm(torch.stack(std),ord=1)
        m = torch.stack(std).mean()
        #m = (sum(std) / len(std)).mean().sqrt()
        return m

    def loss_std_ratio(self,ng,pg):
        ratio=[]
        for n,p in zip(ng,pg):
            ratio.append(self.std_ratio(n,p))
        m = sum(ratio) / len(ratio)
        return m

    def neg_pos_ratio(self,nb,pb):
        loss=self.noise_to_signal(nb,pb)
        return loss

    def final_loss2n(self,pos,rat):
        p=pos/self.args.mre
        r=rat/self.args.mnsr
        loss=(p**2+r**2)/(p+r) # thus each loss is scaled by its magnitude and normalized
        return loss

    def final_loss2(self,pos,rat):
        p=pos#/self.args.mre
        r=rat#/self.args.mnsr
        loss=(p**2+r**2)/(p+r) # thus each loss is scaled by its magnitude and normalized
        return loss

    def final_loss3(self,neg,pos,rat):
        n=self.scale['neg']*neg
        p=self.scale['pos']*pos
        r=self.scale['ratio']*rat
        loss=(p**2+n**2+r**2)/(p+n+r) # thus each loss is scaled by its magnitude and normalized
        return loss

    def grad_sizes(self, pos_g, neg_g):
        N = pos_g[0].shape[0]
        l2_p = torch.stack( [ g.pow(2).reshape(N,-1).sum(dim=1) for g in pos_g ] )
        l2_n = torch.stack( [ g.pow(2).reshape(N,-1).sum(dim=1) for g in neg_g ] )
        loss = (l2_p - l2_n).abs().sum(dim=0).mean()
        return loss
        
    def positive_neuron_boost(self, pos_g):
        N = pos_g[0].shape[0]
        loss = torch.zeros(N).to(self.device)
        for g, p_idx, n_idx in zip( pos_g, self.pos_g_idx, self.neg_g_idx ):
            #p_idx = p_idx.expand( g.shape[0], *[-1]*p_idx.dim() )
            p_g = torch.mean( g[:, p_idx].abs(), dim=1 )
            n_g = torch.mean( g[:, n_idx].abs(), dim=1 )
            #loss += n_g / torch.clamp( p_g, min=1e-10, max=1e10 )
            loss += n_g - p_g
        return loss.mean()

    def negative_neuron_boost(self, neg_g):
        N = neg_g[0].shape[0]
        loss = torch.zeros(N).to(self.device)
        for g, p_idx, n_idx in zip( neg_g, self.pos_g_idx, self.neg_g_idx ):
            p_g = torch.mean( g[:, p_idx].abs(), dim=1 )
            n_g = torch.mean( g[:, n_idx].abs(), dim=1 )
            #loss += p_g / torch.clamp( n_g, min=1e-10, max=1e10 )
            loss += p_g - n_g
        return loss.mean()

    def w_reg(self, W):
        loss = torch.zeros(1).to(self.device)
        for w, size in zip( W, self.W_sizes ):
            loss += (w.pow(2).mean()/size - 1).pow(2)
        return loss

    def cos_loss2(self, layers):
        coss = []
        for e_n_l in layers:
            e_n=e_n_l.flatten(start_dim=1)
            dots = e_n @ e_n.T
            ns = e_n.pow(2).sum(dim=1, keepdim=True)
            ns = ns @ ns.T
            cos = dots.pow(2)/ns
            cos = cos.mean()
            coss.append(cos)
        cos = sum(coss)
        return cos

    def cos_loss_neg(self, neg_g):
        coss = []
        N = layers[0].shape[0]
        for e_n in layers:
            e_n = e_n.reshape(N,-1)
            dots = e_n @ e_n.T
            ns = e_n.pow(2).sum(dim=1, keepdim=True)
            ns = ns @ ns.T
            cos = dots.pow(2)/ns
            cos = cos.mean()
            coss.append(cos)
        cos = sum(coss)
        return cos

    def ratio(self,nB,pB): #batched version of noise_to_signal #TODO currently quadratic wrt the batch_size, if used with large batch needs to be changed
        loss = torch.clamp(torch.einsum('bi,ci->bci',nB**2,1/pB**2),max=self.args.clamp)
        return loss.mean()

    def loss_ratio(self,ng,pg):
        ratio=[]
        N_n = ng[0].shape[0]
        N_p = pg[0].shape[0]
        for n,p in zip(ng,pg):
            n = n.reshape(N_n,-1)	
            p = p.reshape(N_p,-1)	
            ratio.append(self.ratio(n,p))
        m = sum(ratio) / len(ratio)
        return m


    def std_ratio(self,nB,pB): #batched version of noise_to_signal #TODO currently quadratic wrt the batch_size, if used with large batch needs to be changed
        nstd=torch.std(nB,dim=0)
        pstd=torch.std(pB,dim=0)
        loss = torch.log( nstd + 1e-8 ) - torch.log( pstd + 1e-8 )
        #loss = torch.clamp(nstd/pstd,max=self.args.clamp, min=1e-8)
        return loss.mean()

    def loss_std_ratio(self,ng,pg):
        ratio=[]
        for n,p in zip(ng,pg):
            ratio.append(self.std_ratio(n,p))
        m = sum(ratio) / len(ratio)
        return m
    
    def tot_norm(self,gs):
        norms=[]
        for g in gs:
            norms.append(torch.linalg.norm(g))
        return torch.linalg.norm(torch.stack(norms),ord=1)
    def l2_p_minus_mean_n(self,ng,pg):
        l2s=[]
        N_n = ng[0].shape[0]
        N_p = pg[0].shape[0]
        for n,p in zip(ng,pg):
            n = n.reshape(N_n,-1)
            p = p.reshape(N_p,-1)	
            n_mean = n.mean(dim=0, keepdims=True)
            l2 = torch.sigmoid( -(p - n_mean).pow(2).sum() )
            l2s.append(l2)
        m = sum(l2s) / len(l2s)
        return m

class wrapRN(nn.Module):
    def __init__(self):
        super(wrapRN, self).__init__()
        self.rn=torchvision.models.resnet18()

    def forward(self,x):
        x=x.expand(-1,3,-1,-1)
        return self.rn(x)

class Filters(nn.Module):
    def __init__(self):
        super(Filters, self).__init__()
        self.bn=torch.nn.BatchNorm2d(1,track_running_stats=False,affine=False,momentum=None)

    def rel_brightness_bin(self,datapoints):
        ms=datapoints.mean(dim=(1,2,3),keepdims=True)
        msn=self.bn(ms)
        flt=(msn>0)*(msn>8/datapoints.shape[0])
        fltd=torch.einsum('bcwh,b->cwh',datapoints,flt.squeeze().float()).unsqueeze(0)
        return fltd

    def abs_rel_bri_bat(self,datapoints,classifier_o):
        flt = classifier_o
        fltd=torch.einsum('icwh,io->ocwh',datapoints,flt)
        return fltd

    def abs_rel_bri_bat_test(self,datapoints,classifier_o):
        fltd=torch.einsum('bcwh,b->bcwh',datapoints,classifier_o) 
        return fltd



def get_decoder(args, grad_ex):
    if args.decoder == 'orig':
        disaggregator_id=num_params(grad_ex)
        disaggregator_od=int((args.input_size[0])*(args.input_size[1])*(args.input_size[2])*(args.mid_rep_frac))
        reconstructor_id=disaggregator_od
        reconstructor_od=(args.input_size[0])*(args.input_size[1])*(args.input_size[2])
        disaggregator=torch.nn.Linear(disaggregator_id,disaggregator_od,bias=False).to(args.device)
        assert disaggregator.bias is None, "disaggregator must be additive, thus linear map, hence no bias"
        # reconstructor could possibly be a more complex model
        reconstructor=torch.nn.Linear(reconstructor_id,reconstructor_od,bias=True).to(args.device)
    elif args.decoder == 'nomid':
        disaggregator_id=num_params(grad_ex)
        disaggregator_od=num_params(grad_ex)
        reconstructor_id=disaggregator_od
        reconstructor_od=(args.input_size[0])*(args.input_size[1])*(args.input_size[2])
        disaggregator=torch.nn.Identity().to(args.device)
        reconstructor=torch.nn.Linear(reconstructor_id,reconstructor_od,bias=True).to(args.device)
    elif args.decoder == 'deconv':
        disaggregator_id=num_params(grad_ex)
        disaggregator_od=num_params(grad_ex)
        reconstructor_id=disaggregator_od
        reconstructor_od=(args.input_size[0])*(args.input_size[1])*(args.input_size[2])
        disaggregator=torch.nn.Identity().to(args.device)
        reconstructor=DeconvDecoder(disaggregator_od,args.input_size).to(args.device)
    return disaggregator, reconstructor
 
