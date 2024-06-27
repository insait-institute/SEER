import torch
import time
from parameters import *
from model import Losses as Losses
from model import Filters as Filters
from torch import optim
from reconstruct import BreachingReconstruction
from copy import copy
from os.path import exists
from utils import *
import distr

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def train(args, modules, optimizer, trainset, testset, checkpoint=None):
    loader_train = torch.utils.data.DataLoader( trainset, num_workers=2, batch_size=args.batch_size_train[0] + args.batch_size_train[1], sampler = torch.utils.data.RandomSampler(trainset, replacement=False, num_samples=int(1e10)))
    loader_test = torch.utils.data.DataLoader( testset, num_workers=2, batch_size=args.batch_size_train[0] + args.batch_size_train[1], sampler = torch.utils.data.RandomSampler(trainset, replacement=False, num_samples=int(1e10)))
    public_model, grad_ex, disaggregator, reconstructor = modules#, grad_sel = modules#, binarizer= modules#, decoder, wproj = modules
    
    logfile=args.res_path+'/'+f"trainlog.{time.strftime('%y%m%d.%H%M%S', time.localtime())}"
    # TODO learning rate/decay
    current_epoch=0
    total_step = args.epoch_steps
    
    factor = 1
    log_gradient_norms=False
    if checkpoint is not None:
        public_model.load_state_dict(checkpoint['public_model_state_dict'])
        public_model.to(args.device)
        disaggregator.load_state_dict(checkpoint['disaggregator_state_dict'])
        disaggregator.to(args.device)
        reconstructor.load_state_dict(checkpoint['reconstructor_state_dict'])
        reconstructor.to(args.device)
        current_epoch=checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_to(optimizer,args.device)
        losses=Losses( public_model.parameters(), args )
        filters=Filters()
    else:
        losses=Losses( public_model.parameters(), args )
        filters=Filters()

    if args.neptune is not None:
        model_filename = args.neptune_id + '.' + ''.join(filter(lambda x: x.isalnum() or x in ['_','.','-','+','='],args.label))+".params"
    else:
        model_filename = 'model.params'
    model_filename = model_filename[:210]
    model_path = args.res_path+'/'+model_filename

    num_batch=0
    if args.current_epoch is not None:
        current_epoch=args.current_epoch
    for epoch in range(current_epoch,args.num_epochs+3):
        i=0
        ldrlen=len(loader_train)
        for j, (images, labels) in enumerate(loader_train):
            if j>args.epoch_steps:
                break
            flt=0
            datapoints=images.flatten(start_dim=0,end_dim=flt).to(args.device)

            if datapoints.shape[0] < loader_train.batch_size:
                continue

            if args.public_labels=='Zeroed':
                public_labels=torch.zeros_like(labels).flatten(start_dim=0,end_dim=flt).to(args.device)
            elif args.public_labels=='True':
                public_labels=labels.flatten(start_dim=0,end_dim=flt).to(args.device)
            else:
                assert False, "not allowed option for args.public_labels"


            scores = property_scores(datapoints, args.prop, public_labels)
            order = scores.argsort(descending=False)
            datapoints = datapoints[order]
            
            if args.prop_mode == 'thresh':
                pos_batch=((j%2)==1)
                dpmeans=scores
                dpmeans = dpmeans[order]
                newmeans = distr.tweak(dms=dpmeans.unsqueeze(0).unsqueeze(0),pos=pos_batch,prop=args.prop,thresh=args.thresh,disag_size=args.num_clients*(args.batch_size_train[0] + args.batch_size_train[1]),bn=distr.batch_norm,N=distr.normal).squeeze()
                if args.prop in ['bright','dark']:
                    datapoints = datapoints + (newmeans - dpmeans).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                elif args.prop in ['red']:
                    datapoints = datapoints + ((newmeans - dpmeans).unsqueeze(1)*torch.tensor([2.,-1.,-1.],device=datapoints.device).unsqueeze(0)).unsqueeze(2).unsqueeze(3)
            else:
                pos_batch = True

            public_labels = public_labels[order]

            with torch.no_grad():
                neg_ix_i=torch.arange(args.batch_size_train[0]).cuda()
                pos_ix_i=torch.arange(args.batch_size_train[0],args.batch_size_train[0]+args.batch_size_train[1]).cuda()
                neg_ix_o=torch.arange(args.jac_size).cuda()
                neg_ix_o_a=neg_ix_o[0:1]
                neg_ix_o_b=neg_ix_o[1:2]
                assert args.jac_size==2, "indices above are hardcoded"
                pos_ix_o=torch.arange(neg_ix_o.shape[0],neg_ix_o.shape[0]+pos_ix_i.shape[0]).cuda()

                jac=torch.zeros(neg_ix_i.shape[0]+pos_ix_i.shape[0],neg_ix_o.shape[0]+pos_ix_o.shape[0],device='cuda')
                assert pos_ix_i.shape[0]==pos_ix_o.shape[0], "compute precisely the grad for each positive sample"
                jac[pos_ix_i.unsqueeze(1),pos_ix_o.unsqueeze(0)]=torch.eye(pos_ix_i.shape[0],device='cuda')
                #jac[neg_ix_i.unsqueeze(1),neg_ix_o.unsqueeze(0)]=torch.nn.init.orthogonal_(torch.randn(neg_ix_i.shape[0],neg_ix_o.shape[0],device='cuda')).cuda()
                if not pos_batch:
                    neg_ix_i=torch.arange(args.batch_size_train[0]+args.batch_size_train[1]).cuda()
                jac[neg_ix_i.unsqueeze(1),neg_ix_o_a.unsqueeze(0)]=torch.nn.init.orthogonal_(torch.randn(neg_ix_i.shape[0],neg_ix_o_a.shape[0],device='cuda')).cuda()
                jac[neg_ix_i.unsqueeze(1),neg_ix_o_b.unsqueeze(0)]=torch.ones(neg_ix_i.shape[0],neg_ix_o_b.shape[0],device='cuda').cuda() / (neg_ix_i.shape[0]**0.5)

            bdW, W = grad_ex(datapoints,public_labels.to(args.device),losses.public_loss,flat_cat=True,single_grad=False,jac=jac.t()) 
            assert bdW.shape[0]==jac.shape[1]
            disaggregator_o=disaggregator(bdW)
            reconstructor_o=reconstructor(disaggregator_o).reshape(-1, *args.input_size)
            with torch.no_grad():
                reconstructor_neg_tgt_o=reconstructor(torch.zeros_like(disaggregator_o)).reshape(-1,*args.input_size)
                tgt_o=reconstructor_neg_tgt_o
                assert args.batch_size_train[1] == 1, "Assumes 1 positive"
                if pos_batch:
                    tgt_o[ jac.shape[1] - 1 ] = datapoints[ datapoints.shape[0] - 1 ]
            neg_loss=(reconstructor_o-tgt_o)[neg_ix_o].pow(2).mean()
            pos_loss=(reconstructor_o-tgt_o)[pos_ix_o].pow(2).mean()


            alpha=2**rat_sched(epoch,args.sched_x_1,args.sched_x_end,args.sched_y_1,args.sched_y_end)
            neg_term = (alpha/(alpha+1)) * neg_loss
            pos_term = (1/(alpha+1)) * pos_loss
            tot_loss = pos_term + neg_term


            vd={}
            vd['tra']={}
            vard=vd['tra']
            vard['metr']={}
            lossd=vard['metr']
            vard['metr']['opt']={}
            vard['metr']['log']={}
            opt=vard['metr']['opt']
            log=vard['metr']['log']
            opt['tot_loss']=tot_loss
            with torch.no_grad():
                log['neg_loss']=neg_loss.sqrt()
                log['pos_loss']=pos_loss.sqrt()

            loss = torch.sum(torch.stack([opt[k] for k in opt],dim=0))
            if j%args.acc_grad==0:
                optimizer.zero_grad()
            if log_gradient_norms:
                losses_grad_norms = {}
                last_acc_grad = None
                with torch.no_grad():
                    last_acc_grad = [torch.zeros_like(p) for p in optimizer.param_groups[0]['params']]
                losses_grad_norms = {}
                for k in lossd['opt']:
                    loss_opt=lossd['opt'][k]
                    loss_opt.backward(retain_graph=True)
                    with torch.no_grad():
                        grad_diff = [p.grad-lg for p,lg in zip(optimizer.param_groups[0]['params'],last_acc_grad)]
                        losses_grad_norms[k] = torch.norm(torch.stack([torch.norm(gd) for gd in grad_diff],dim=0))
                        for i in range(len(last_acc_grad)):
                            last_acc_grad[i] += grad_diff[i]
            else:
                loss.backward()    

            if j%args.acc_grad==(args.acc_grad-1):
                optimizer.step()               

            if (i + 0) % args.print_interval == 0:
                trav(args,'',vd,log_stdout)
                print ('Epoch [{}/{}], Step [{}/{}/{}]'.format(epoch + 1, args.num_epochs, i + 0, j + 0, total_step))

                if args.neptune is not None:
                    args.neptune['train/epochs'].log( epoch  );
                    args.neptune['train/step'].log( i + 0 );
                    args.neptune['train/step_with_skipped'].log( j + 0 );
                    args.neptune['train/loss'].log( loss.item() );
                print('train/epochs',': ', epoch  )
                print('train/step',': ', i + 0 )
                print('train/step_with_skipped',': ', j + 0 )
                print('train/loss',': ', loss.item() )
                if log_gradient_norms:
                    for k in losses_grad_norms:
                        print(f"train/gn_{k}",': ', losses_grad_norms[k] )
                    if log_gradient_norms:
                        for k in losses_grad_norms:
                            args.neptune[f"train/gn_{k}"].log( losses_grad_norms[k] );print(f"train/gn_{k}",': ', losses_grad_norms[k] )
                    trav(args,'',vd,log_nept)
            i+=1

        if (epoch % args.test_interval == 0) and (epoch > 2):
            metr=False
            if epoch % args.big_test_interval >= (args.big_test_interval - args.test_interval):
                metr=True
                torch.save({
                        'epoch': epoch,
                        'public_model_state_dict': public_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'disaggregator_state_dict': disaggregator.state_dict(),
                        'reconstructor_state_dict': reconstructor.state_dict(),
                        }, model_path+f".{epoch}")
            print( model_path )
            torch.save({
                    'epoch': epoch,
                    'public_model_state_dict': public_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'disaggregator_state_dict': disaggregator.state_dict(),
                    'reconstructor_state_dict': reconstructor.state_dict(),
                    }, model_path)
            args2 = copy(args)
            args2.checkpoint = model_path
            tests(args2, modules, trainset, testset, vis_res=True,metr=metr)


def load_models(public_model,decoder,checkpoint):
        current={
                'public_model_state_dict': public_model.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                }
        public_model.load_state_dict(checkpoint['public_model_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        return current

def tests(args, modules, trainset, testset, checkpoint=None, vis_res=False,metr=False):
    cfgs=args.attack_cfg.split('+')
    for cfg in cfgs:
        args.attack_cfg=cfg
        if args.prop_mode=='thresh':
            test_sec_aggr(args, modules, trainset, testset, checkpoint=None,metr=metr)
            test_sec_aggr_end2end(args, modules, trainset, testset, checkpoint=None,metr=metr)
        elif args.prop_mode=='max':
            test_end2end(args, modules, trainset, testset, checkpoint=None)
        else:
            assert False, "wrong mode"

def test_sec_aggr_end2end(args, modules, trainset, testset, checkpoint=None,metr=False):
    public_model, grad_ex, disaggregator, reconstructor = modules#, decoder, binarizer, wproj = modules
    torch.save({'par_sel': grad_ex.par_sel},'/tmp/runtime_dict')
    
    logfile=args.res_path+'/'+f"trainlog.{time.strftime('%y%m%d.%H%M%S', time.localtime())}"
    current_epoch=0
    
    factor = 1
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    if checkpoint is not None:
        public_model.load_state_dict(checkpoint['public_model_state_dict'])
        public_model.to(args.device)
        disaggregator.load_state_dict(checkpoint['disaggregator_state_dict'])
        disaggregator.to(args.device)
        reconstructor.load_state_dict(checkpoint['reconstructor_state_dict'])
        reconstructor.to(args.device)
        current_epoch=checkpoint['epoch']
        losses=Losses( public_model.parameters(), args )
        filters=Filters()
    else:
        losses=Losses( public_model.parameters(), args )
        filters=Filters()
     
    loaders = {}
    batch_sz = sum(args.batch_size_test)
    num_clients = args.num_clients
    ncs=[num_clients]
    test_sizes = [(nc,batch_sz) for nc in ncs]
    sets = { "train": trainset,"test": testset }
    #sets = { "test": testset }
    for k in sets.keys():
        for i,j in test_sizes:
            loaders[k + f'_{i}x{j}'] = ( i, j, torch.utils.data.DataLoader( sets[k], batch_size=i*j, num_workers=2,sampler = torch.utils.data.RandomSampler(sets[k], replacement=False, num_samples=int(1e10)) ) )

    #assert len(loaders) == 2
    br_reconstr = BreachingReconstruction( args, public_model, losses.public_loss, batch_sz, num_clients, dtype=torch.float32 )
    def show(px,sd):
        if len(px.shape)==3:
            reconstructed_user_data = {'data':px.reshape(1,args.input_size[0],args.input_size[1],-1), 'labels':None }
            br_reconstr.user.plot( reconstructed_user_data, neptune=args.neptune, subdir=f"{sd}" )
        else:
            assert len(px.shape)==4
            for px_i in px:
                show(px_i,sd)


    for l in loaders.keys():
        n_batches, bs, testing_set = loaders[l]
        data=[]
        avg_metrics = None
        strikes=0
        for j, (images, labels) in enumerate(testing_set):
            datapoints = images.reshape(-1,*args.input_size).to(args.device)
            flt=0
            if args.public_labels=='Zeroed':
                public_labels=torch.zeros_like(labels).flatten(start_dim=0,end_dim=flt).to(args.device)
            elif args.public_labels=='True':
                public_labels=labels.flatten(start_dim=0,end_dim=flt).to(args.device)
            else:
                assert False, "not allowed option for args.public_labels"
            
            print(f'test:{n_batches}x{bs}, id:{j}')
            
            bdW_sum = None
            datapoint_true = []
            tot_num_pos=0
            #first check if multibatch is ok
            for b in range( n_batches ):
                i_st = b * bs
                i_en = (b + 1) * bs

                flt=0
                datapoints_i=datapoints[ i_st : i_en ]
                assert args.prop in ['bright','dark','red'], "works only for those currently"
                scores = property_scores(datapoints_i,prop=args.prop)
                order = scores.argsort(descending=False)
                dpmeans=scores
                dpmeans = dpmeans[order]
                datapoints_i = datapoints_i[order]
                public_labels[ i_st : i_en ] = public_labels[ i_st : i_en ][order]
                num_pos = distr.count(dms=dpmeans.unsqueeze(0).unsqueeze(0),pos=None,prop=args.prop,thresh=args.thresh,disag_size=args.num_clients*(args.batch_size_train[0] + args.batch_size_train[1]),bn=distr.batch_norm,N=distr.normal)
                datapoints_i=datapoints_i.to(args.device)
                for p in range(num_pos):
                    datapoint_true.append(datapoints_i[-(p+1)])
                tot_num_pos += num_pos

            print(f"{l}\t{j}\t{strikes}\t{tot_num_pos}\n") # log to neptune
            if False:#tot_num_pos!=1:
                args.neptune["test/sampling"].log(f"{l}\t{j}\t{strikes}\t{tot_num_pos}\n");print("test/sampling",': ',f"{l}\t{j}\t{strikes}\t{tot_num_pos}\n")
                continue
            if len(datapoint_true)==1:
                strikes+=1
            if args.neptune is not None:
                args.neptune["test/sampling"].log(f"{l}\t{j}\t{strikes}\t{tot_num_pos}\n");print("test/sampling",': ',f"{l}\t{j}\t{strikes}\t{tot_num_pos}\n")

            #now use multibatch
            for b in range( n_batches ):
                i_st = b * bs
                i_en = (b + 1) * bs

                flt=0
                datapoints_i=datapoints[ i_st : i_en ]
                assert args.prop in ['bright','dark','red'], "works only for those currently"
                scores = property_scores(datapoints_i,prop=args.prop)
                order = scores.argsort(descending=False)
                dpmeans=scores
                dpmeans = dpmeans[order]
                datapoints_i = datapoints_i[order]
                public_labels[ i_st : i_en ] = public_labels[ i_st : i_en ][order]

                bdW, _ = grad_ex( datapoints_i, public_labels[ i_st : i_en ], losses.public_loss, flat_cat=True, single_grad=True, testing=True )
                if bdW_sum is None:
                    bdW_sum = bdW
                else:
                    bdW_sum += bdW
            bdW = bdW_sum

            disaggregator_o = disaggregator(bdW)
            reconstructor_o = reconstructor(disaggregator_o).reshape(-1,*args.input_size)
            
            if metr:
                metrics = br_reconstr.get_metrics( reconstructor_o.reshape(1,args.input_size[0],args.input_size[1],-1), None, datapoints, public_labels, neptune=False )
                closest_idx = metrics['selector'][0]
                del metrics['selector']
                del metrics['order']
                if args.neptune is not None:
                    for k in metrics:
                        args.neptune[f"metrics/sec_aggr_end2end/epoch_{current_epoch}/{k}_{l}"].log(metrics[k]);print(f"metrics/sec_aggr_end2end/epoch_{current_epoch}/{k}_{l}",': ',metrics[k])
                if avg_metrics is None:
                    avg_metrics = metrics
                else:
                    for k in metrics:
                        avg_metrics[k] *= j/(j+1)
                        avg_metrics[k] += metrics[k]/(j+1)

            if metr:
                show(datapoints[closest_idx],f"saee_epoch{current_epoch}_{l}_disagg_random_sample_chosen")
            if len(datapoint_true)>0:
                show(torch.cat(datapoint_true,dim=-1),f"saee_epoch{current_epoch}_{l}_disagg_random_sample_true")
            else:
                show(0*reconstructor_o,f"saee_epoch{current_epoch}_{l}_disagg_random_sample_true")
            show(reconstructor_o,f"saee_epoch{current_epoch}_{l}_disagg_random_sample_rec")

            if (j+1) % args.print_test_interval == 0:
                print('|',j+1 )
            if args.num_test_img is not None:
                #if (strikes >= args.num_test_img) or (j+1 >= 100*args.num_test_img):
                if (j+1 >= args.num_test_img):

                    break

        print(f'{l} | ', end='')
        if metr:
            for k in metrics:
                if args.neptune is not None:
                    args.neptune[f"metrics/sec_aggr_end2end/avg_{k}_{l}"].log(avg_metrics[k]);
                print(f"metrics/sec_aggr_end2end/avg_{k}_{l}",': ',avg_metrics[k])
                print( f"avg_{k}: {avg_metrics[k]} | ", end='' )
        print('')

def test_sec_aggr(args, modules, trainset, testset, checkpoint=None,metr=False):
    public_model, grad_ex, disaggregator, reconstructor = modules#, decoder, binarizer, wproj = modules
    torch.save({'par_sel': grad_ex.par_sel},'/tmp/runtime_dict')
    
    logfile=args.res_path+'/'+f"trainlog.{time.strftime('%y%m%d.%H%M%S', time.localtime())}"
    current_epoch=0
    
    factor = 1
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    if checkpoint is not None:
        public_model.load_state_dict(checkpoint['public_model_state_dict'])
        public_model.to(args.device)
        disaggregator.load_state_dict(checkpoint['disaggregator_state_dict'])
        disaggregator.to(args.device)
        reconstructor.load_state_dict(checkpoint['reconstructor_state_dict'])
        reconstructor.to(args.device)
        current_epoch=checkpoint['epoch']
        losses=Losses( public_model.parameters(), args )
        filters=Filters()
    else:
        losses=Losses( public_model.parameters(), args )
        filters=Filters()

    loaders = {}
    batch_sz = sum(args.batch_size_test)
    num_clients = args.num_clients
    ncs=[num_clients]
    test_sizes = [(nc,batch_sz) for nc in ncs]
    sets = { "train": trainset,"test": testset }
    for k in sets.keys():
        for i,j in test_sizes:
            loaders[k + f'_{i}x{j}'] = ( i, j, torch.utils.data.DataLoader( sets[k], batch_size=i*j, num_workers=2,sampler = torch.utils.data.RandomSampler(sets[k], replacement=False, num_samples=int(1e10)) ) )


    assert len(loaders) == 2
    br_reconstr = BreachingReconstruction( args, public_model, losses.public_loss, batch_sz, num_clients, dtype=torch.float32 )
    def show(px,sd):
        if len(px.shape)==3:
            reconstructed_user_data = {'data':px.reshape(1,args.input_size[0],args.input_size[1],-1), 'labels':None }
            br_reconstr.user.plot( reconstructed_user_data, neptune=args.neptune, subdir=f"{sd}" )
        else:
            assert len(px.shape)==4
            for px_i in px:
                show(px_i,sd)

    for l in loaders.keys():
        n_batches, bs, testing_set = loaders[l]
        data=[]
        avg_metrics = None
        for j, (images, labels) in enumerate(testing_set):
            datapoints = images.reshape(-1,*args.input_size).to(args.device)
            flt=0
            if args.public_labels=='Zeroed':
                public_labels=torch.zeros_like(labels).flatten(start_dim=0,end_dim=flt).to(args.device)
            elif args.public_labels=='True':
                public_labels=labels.flatten(start_dim=0,end_dim=flt).to(args.device)
            else:
                assert False, "not allowed option for args.public_labels"
            
            print(f'test:{n_batches}x{bs}, id:{j}')
            
            bdW_sum = None
            datapoint_true = None
            for b in range( n_batches ):
                i_st = b * bs
                i_en = (b + 1) * bs

                flt=0
                datapoints_i=datapoints[ i_st : i_en ]
                assert args.prop in ['bright','dark','red'], "works only for those currently"
                scores = property_scores(datapoints_i,prop=args.prop)
                order = scores.argsort(descending=False)
                dpmeans=scores
                dpmeans = dpmeans[order]
                datapoints_i = datapoints_i[order]
                public_labels[ i_st : i_en ] = public_labels[ i_st : i_en ][order]
                pos_batch=(b==0)
                twk = distr.tweak(dms=dpmeans.unsqueeze(0).unsqueeze(0).cpu(),pos=pos_batch,prop=args.prop,thresh=args.thresh,disag_size=args.num_clients*(args.batch_size_train[0] + args.batch_size_train[1]),bn=distr.batch_norm,N=distr.normal)
                if twk is None:
                    twk=dpmeans
                newmeans = twk.squeeze().cuda()
                if args.prop in ['bright','dark']:
                    datapoints_i = datapoints_i + (newmeans - dpmeans).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                elif args.prop in ['red']:
                    datapoints_i = datapoints_i + ((newmeans - dpmeans).unsqueeze(1)*torch.tensor([2.,-1.,-1.],device=datapoints_i.device).unsqueeze(0)).unsqueeze(2).unsqueeze(3)
                datapoints_i=datapoints_i.to(args.device)
                if pos_batch:
                    datapoint_true=datapoints_i[-1]

                bdW, _ = grad_ex( datapoints_i, public_labels[ i_st : i_en ], losses.public_loss, flat_cat=True, single_grad=True, testing=True )
                if bdW_sum is None:
                    bdW_sum = bdW
                else:
                    bdW_sum += bdW
            bdW = bdW_sum

            disaggregator_o = disaggregator(bdW)
            reconstructor_o = reconstructor(disaggregator_o).reshape(-1,*args.input_size)
            
            if metr:
                metrics = br_reconstr.get_metrics( reconstructor_o.reshape(1,args.input_size[0],args.input_size[1],-1), None, datapoints, public_labels, neptune=False )
                closest_idx = metrics['selector'][0]
                del metrics['selector']
                del metrics['order']
                if args.neptune is not None:
                    for k in metrics:
                        if args.neptune is not None:
                            args.neptune[f"metrics/sec_aggr/epoch_{current_epoch}/{k}_{l}"].log(metrics[k]);
                        print(f"metrics/sec_aggr/epoch_{current_epoch}/{k}_{l}",': ',metrics[k])
                if avg_metrics is None:
                    avg_metrics = metrics
                else:
                    for k in metrics:
                        avg_metrics[k] *= j/(j+1)
                        avg_metrics[k] += metrics[k]/(j+1)

            show(datapoint_true,f"satw_epoch{current_epoch}_{l}_disagg_random_sample_true")
            show(reconstructor_o,f"satw_epoch{current_epoch}_{l}_disagg_random_sample_rec")
            if metr:
                show(datapoints[closest_idx],f"satw_epoch{current_epoch}_{l}_disagg_random_sample_chosen")
           
            if (j+1) % args.print_test_interval == 0:
                print('|',j+1 )
            if args.num_test_img is not None:
                if j+1 >= args.num_test_img:
                    break

        print(f'{l} | ', end='')
        if metr:
            for k in metrics:
                if args.neptune is not None:
                    args.neptune[f"metrics/sec_aggr/avg_{k}_{l}"].log(avg_metrics[k]);print(f"metrics/sec_aggr/avg_{k}_{l}",': ',avg_metrics[k])
                print( f"avg_{k}: {avg_metrics[k]} | ", end='' )
        print('')


def test_end2end(args, modules, trainset, testset, checkpoint=None):
    public_model, grad_ex, disaggregator, reconstructor = modules#, decoder, binarizer, wproj = modules
    torch.save({'par_sel': grad_ex.par_sel},'/tmp/runtime_dict')
    
    logfile=args.res_path+'/'+f"trainlog.{time.strftime('%y%m%d.%H%M%S', time.localtime())}"
    current_epoch=0
    
    factor = 1
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    if checkpoint is not None:
        public_model.load_state_dict(checkpoint['public_model_state_dict'])
        public_model.to(args.device)
        disaggregator.load_state_dict(checkpoint['disaggregator_state_dict'])
        disaggregator.to(args.device)
        reconstructor.load_state_dict(checkpoint['reconstructor_state_dict'])
        reconstructor.to(args.device)
        current_epoch=checkpoint['epoch']
        losses=Losses( public_model.parameters(), args )
        filters=Filters()
    else:
        losses=Losses( public_model.parameters(), args )
        filters=Filters()
     
    test_size = sum(args.batch_size_test)
    br_reconstr = BreachingReconstruction( args, public_model, losses.public_loss, test_size, 1, dtype=torch.float32 )
    def show(px,sd):
        if len(px.shape)==3:
            reconstructed_user_data = {'data':px.reshape(1,args.input_size[0],args.input_size[1],-1), 'labels':public_labels.new_zeros(1) }
            br_reconstr.user.plot( reconstructed_user_data, neptune=args.neptune, subdir=f"{sd}" )
        else:
            assert len(px.shape)==4
            for px_i in px:
                show(px_i,sd)

    loaders = {}
    test_sizes = [ int( test_size * f ) for f in [1]]
    #sets = { "train": trainset,"test": testset }
    sets = { "test": testset }
    for k in sets.keys():
        for i in test_sizes:
            loaders[k + f'_{i}'] = torch.utils.data.DataLoader( sets[k], batch_size=i, num_workers=2,sampler = torch.utils.data.RandomSampler(sets[k], replacement=False, num_samples=int(1e10)) )

    for l in loaders.keys():
        testing_set = loaders[l]
        data=[]
        avg_metrics = None
        for j, (images, labels) in enumerate(testing_set):
            
            if args.num_test_img is not None:
                if j >= args.num_test_img:
                    break

            flt=0
            datapoints=images.flatten(start_dim=0,end_dim=flt).to(args.device)

            if datapoints.shape[0] < testing_set.batch_size:
                continue

            if args.public_labels=='Zeroed':
                public_labels=torch.zeros_like(labels).flatten(start_dim=0,end_dim=flt).to(args.device)
            elif args.public_labels=='True':
                public_labels=labels.flatten(start_dim=0,end_dim=flt).to(args.device)
            else:
                assert False, "not allowed option for args.public_labels"

            N = datapoints.shape[0]
            print('j,N: ',j,N)

            bdW, _ = grad_ex(datapoints,public_labels,losses.public_loss,flat_cat=True, single_grad=True,testing=True)

            disaggregator_o = disaggregator(bdW)
            reconstructor_o = reconstructor(disaggregator_o).reshape(-1,*args.input_size)

            #DSNR calculation
            bdWj, _ = grad_ex(datapoints,public_labels,losses.public_loss,flat_cat=False, single_grad=False,jac=torch.eye(testing_set.batch_size).to('cuda'),testing=True,par_sel=False)
            gradnames = []
            for m in grad_ex.public_model.named_parameters():
                gradnames.append(m[0])

            example_grads = [g.view(g.shape[0], -1) for g in bdWj]
            weight_norms = []
            for en_i, grads in enumerate(example_grads):
                nm = gradnames[en_i]
                if nm == 'linear.weight' or ('.conv' in nm and '.weight' in nm):
                    norms = torch.linalg.vector_norm(grads, dim=1)
                    weight_norms.append(norms)

            weight_norms_stacked = torch.vstack(weight_norms) # [#layers, #images]
            wns = weight_norms_stacked
            s2n = wns.max(dim=1)[0] / ( wns.sum(dim=1) - wns.max(dim=1)[0] + 1e-6)
            s2n_final = s2n.max().item()
            
            metrics = br_reconstr.get_metrics( reconstructor_o.reshape(1,args.input_size[0],args.input_size[1],-1), None, datapoints, public_labels, neptune=False )
            closest_idx = metrics['selector'][0]
            del metrics['selector']
            del metrics['order']
            metrics['dsnr'] = s2n_final
            if args.neptune is not None:
                for k in metrics:
                    if args.neptune is not None:
                        args.neptune[f"metrics/epoch_{current_epoch}/{k}_{l}"].log(metrics[k]);
                    print(f"metrics/epoch_{current_epoch}/{k}_{l}",': ',metrics[k])
            if avg_metrics is None:
                avg_metrics = metrics
            else:
                for k in metrics:
                    avg_metrics[k] *= j/(j+1)
                    avg_metrics[k] += metrics[k]/(j+1)


            with torch.no_grad():
                scores = property_scores(datapoints, args.prop, public_labels)
                order = scores.argsort(descending=False)
                top_score_idx=order[-1]

            show(datapoints[closest_idx],f"epoch{current_epoch}_{l}_disagg_random_sample_closest")
            show(datapoints[top_score_idx],f"epoch{current_epoch}_{l}_disagg_random_sample_true")
            show(reconstructor_o,f"epoch{current_epoch}_{l}_disagg_random_sample_rec")

            assert args.batch_size_train[1]==1, "(neg|pos)_ix_(i|o) are hardcoded"
            with torch.no_grad():
                scores = property_scores(datapoints, args.prop, public_labels)
                order = scores.argsort(descending=False)
                datapoints = datapoints[order]

                public_labels = public_labels[order]

                neg_ix_i=torch.arange(datapoints.shape[0]-1).cuda()
                pos_ix_i=torch.arange(datapoints.shape[0]-1,datapoints.shape[0]).cuda()
                neg_ix_o=torch.arange(1).cuda()
                #TODO hardcoded 1
                pos_ix_o=torch.arange(neg_ix_o.shape[0],neg_ix_o.shape[0]+pos_ix_i.shape[0]).cuda()

                jac=torch.zeros(neg_ix_i.shape[0]+pos_ix_i.shape[0],neg_ix_o.shape[0]+pos_ix_o.shape[0],device='cuda')
                assert pos_ix_i.shape[0]==pos_ix_o.shape[0], "compute precisely the grad for each positive sample"
                jac[pos_ix_i.unsqueeze(1),pos_ix_o.unsqueeze(0)]=torch.eye(pos_ix_i.shape[0],device='cuda')
                assert neg_ix_o.shape[0]==1, "aggregate all the negatives"
                jac[neg_ix_i.unsqueeze(1),neg_ix_o.unsqueeze(0)]=torch.ones(neg_ix_i.shape[0],neg_ix_o.shape[0],device='cuda').cuda()

            bdW, W = grad_ex(datapoints,public_labels.to(args.device),losses.public_loss,flat_cat=True,single_grad=False,jac=jac.t()) 
            assert bdW.shape[0]==jac.shape[1]
            disaggregator_o=disaggregator(bdW)
            reconstructor_o=reconstructor(disaggregator_o).reshape(-1,*args.input_size)
            with torch.no_grad():
                reconstructor_neg_tgt_o=reconstructor(torch.zeros_like(disaggregator_o)).reshape(-1,*args.input_size)
                tgt_o=reconstructor_neg_tgt_o
                tgt_o[ jac.shape[1] - 1 ] = datapoints[ datapoints.shape[0] - 1 ]

            neg_out=(reconstructor_o-tgt_o)[neg_ix_o]
            pos_out=(reconstructor_o)[pos_ix_o]

            show(pos_out,f"epoch{current_epoch}_{l}_disagg_sample_pos_out")
            show(neg_out,f"epoch{current_epoch}_{l}_disagg_sample_neg_out")
        
            if (j+1) % args.print_test_interval == 0:
                print('|',j+1 )

        print(f'{l} | ', end='')
        for k in metrics:
            if args.neptune is not None:
                args.neptune[f"metrics/avg_{k}_{l}"].log(avg_metrics[k]);print(f"metrics/avg_{k}_{l}",': ',avg_metrics[k])
            print( f"avg_{k}: {avg_metrics[k]} | ", end='' )
        print('')
#                if args.prop == 'bright':
#                    order = datapoints.mean((1,2,3)).argsort(descending=False)
#                elif args.prop == 'neg_bright':
#                    order = datapoints.mean((1,2,3)).argsort(descending=True)
#                else:
#                    assert False, "Not a valid property"
#                datapoints = datapoints[order]
#                public_labels = public_labels[order]
#
#                neg_ix_i=torch.arange(datapoints.shape[0]-1).cuda()
#                pos_ix_i=torch.arange(datapoints.shape[0]-1,datapoints.shape[0]).cuda()
#                neg_ix_o=torch.arange(1).cuda()
#                #TODO hardcoded 1
#                pos_ix_o=torch.arange(neg_ix_o.shape[0],neg_ix_o.shape[0]+pos_ix_i.shape[0]).cuda()
#
#                jac=torch.zeros(neg_ix_i.shape[0]+pos_ix_i.shape[0],neg_ix_o.shape[0]+pos_ix_o.shape[0],device='cuda')
#                assert pos_ix_i.shape[0]==pos_ix_o.shape[0], "compute precisely the grad for each positive sample"
#                jac[pos_ix_i.unsqueeze(1),pos_ix_o.unsqueeze(0)]=torch.eye(pos_ix_i.shape[0],device='cuda')
#                assert neg_ix_o.shape[0]==1, "aggregate all the negatives"
#                jac[neg_ix_i.unsqueeze(1),neg_ix_o.unsqueeze(0)]=torch.ones(neg_ix_i.shape[0],neg_ix_o.shape[0],device='cuda').cuda()
#
#            bdW, W = grad_ex(datapoints,public_labels.to(args.device),losses.public_loss,flat_cat=True,single_grad=False,jac=jac.t()) 
#            assert bdW.shape[0]==jac.shape[1]
#            disaggregator_o=disaggregator(bdW)
#            reconstructor_o=reconstructor(disaggregator_o).reshape(-1,*args.input_size)
#            with torch.no_grad():
#                reconstructor_neg_tgt_o=reconstructor(torch.zeros_like(disaggregator_o)).reshape(-1,*args.input_size)
#                tgt_o=reconstructor_neg_tgt_o
#                tgt_o[ jac.shape[1] - 1 ] = datapoints[ datapoints.shape[0] - 1 ]
#
#            neg_out=(reconstructor_o-tgt_o)[neg_ix_o]
#            pos_out=(reconstructor_o)[pos_ix_o]
#
#            show(pos_out,f"epoch{current_epoch}_{l}_disagg_sample_pos_out")
#            show(neg_out,f"epoch{current_epoch}_{l}_disagg_sample_neg_out")
#
#            if (j+1) % args.print_test_interval == 0:
#                print('|',j+1 )
#
#        print(f'{l} | ', end='')
#        for k in metrics:
#            if args.neptune is not None:
#                args.neptune[f"metrics/avg_{k}_{l}"].log(avg_metrics[k]);print(f"metrics/avg_{k}_{l}",': ',avg_metrics[k])
#            print( f"avg_{k}: {avg_metrics[k]} | ", end='' )
#        print('')

def test_end2end_fix_contrast(args, modules, trainset, testset, checkpoint=None):
    public_model, grad_ex, disaggregator, reconstructor = modules#, decoder, binarizer, wproj = modules
    torch.save({'par_sel': grad_ex.par_sel},'/tmp/runtime_dict')
    
    logfile=args.res_path+'/'+f"trainlog.{time.strftime('%y%m%d.%H%M%S', time.localtime())}"
    logger=open(logfile,'a')
    current_epoch=0
    
    factor = 1
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    if checkpoint is not None:
        public_model.load_state_dict(checkpoint['public_model_state_dict'])
        public_model.to(args.device)
        disaggregator.load_state_dict(checkpoint['disaggregator_state_dict'])
        disaggregator.to(args.device)
        reconstructor.load_state_dict(checkpoint['reconstructor_state_dict'])
        reconstructor.to(args.device)
        current_epoch=checkpoint['epoch']
        losses=Losses( public_model.parameters(), args )
        filters=Filters()
    else:
        losses=Losses( public_model.parameters(), args )
        filters=Filters()
     
    test_size = sum(args.batch_size_test)
    br_reconstr = BreachingReconstruction( args, public_model, losses.public_loss, test_size, 1, dtype=torch.float32 )
    def show(px,sd):
        if len(px.shape)==3:
            reconstructed_user_data = {'data':px.reshape(1,args.input_size[0],args.input_size[1],-1), 'labels':public_labels.new_zeros(1) }
            br_reconstr.user.plot( reconstructed_user_data, neptune=args.neptune, subdir=f"{sd}" )
        else:
            assert len(px.shape)==4
            for px_i in px:
                show(px_i,sd)

    loaders = {}
    test_sizes = [ int( test_size * f ) for f in [1]]
    #sets = { "train": trainset,"test": testset }
    sets = { "test": testset }
    for k in sets.keys():
        for i in test_sizes:
            loaders[k + f'_{i}'] = torch.utils.data.DataLoader( sets[k], batch_size=i, num_workers=2,sampler = torch.utils.data.RandomSampler(sets[k], replacement=False, num_samples=int(1e10)) )

    for l in loaders.keys():
        testing_set = loaders[l]
        data=[]
        avg_metrics = None
        best_lb = None
        recs = []
        lbs = []
        lbs2 = []
        for j, (images, labels) in enumerate(testing_set):
            
            if args.num_test_img is not None:
                if j >= args.num_test_img:
                    break

            flt=0
            datapoints=images.flatten(start_dim=0,end_dim=flt).to(args.device)

            if datapoints.shape[0] < testing_set.batch_size:
                continue

            if args.public_labels=='Zeroed':
                public_labels=torch.zeros_like(labels).flatten(start_dim=0,end_dim=flt).to(args.device)
            elif args.public_labels=='True':
                public_labels=labels.flatten(start_dim=0,end_dim=flt).to(args.device)
            else:
                assert False, "not allowed option for args.public_labels"

            N = datapoints.shape[0]
            print('j,N: ',j,N)

            bdW, _ = grad_ex(datapoints,public_labels,losses.public_loss,flat_cat=True, single_grad=True,testing=True)

            disaggregator_o = disaggregator(bdW)
            reconstructor_o = reconstructor(disaggregator_o).reshape(-1,*args.input_size)

            mean = torch.tensor(br_reconstr.cfg.case.data.mean).to(args.device).reshape(1,-1,1,1)
            std = torch.tensor(br_reconstr.cfg.case.data.std).to(args.device).reshape(1,-1,1,1)
            lb = reconstructor_o * std / (1 - mean)
            lb2 = reconstructor_o * std / (-mean)
            lb = lb.max()
            lb2 = lb2.max()
            lbs.append(lb)
            lbs2.append(lb2)
            if args.neptune is not None:
                args.neptune['metrics/lb'].log(lb)
                args.neptune['metrics/lb2'].log(lb2)
            recs.append( (reconstructor_o.cpu(), datapoints.cpu(), public_labels.cpu()) )
        
        best_lb = torch.quantile(torch.tensor(lbs), 0.9)
        best_lb2 = torch.quantile(torch.tensor(lbs2), 0.9)
        best_lb = 0.7#torch.max( best_lb, best_lb2 )
        if args.neptune is not None:
            args.neptune['metrics/best_lb'].log(best_lb)
 
        for reconstructor_o, datapoints, public_labels in recs:
            reconstructor_o, datapoints, public_labels = reconstructor_o.to(args.device), datapoints.to(args.device), public_labels.to(args.device)
            reconstructor_o /= best_lb

            #DSNR calculation
            bdWj, _ = grad_ex(datapoints,public_labels,losses.public_loss,flat_cat=False, single_grad=False,jac=torch.eye(testing_set.batch_size).to('cuda'),testing=True,par_sel=False)
            gradnames = []
            for m in grad_ex.public_model.named_parameters():
                gradnames.append(m[0])

            example_grads = [g.view(g.shape[0], -1) for g in bdWj]
            weight_norms = []
            for en_i, grads in enumerate(example_grads):
                nm = gradnames[en_i]
                if nm == 'linear.weight' or ('.conv' in nm and '.weight' in nm):
                    norms = torch.linalg.vector_norm(grads, dim=1)
                    weight_norms.append(norms)

            weight_norms_stacked = torch.vstack(weight_norms) # [#layers, #images]
            wns = weight_norms_stacked
            s2n = wns.max(dim=1)[0] / ( wns.sum(dim=1) - wns.max(dim=1)[0] + 1e-6)
            s2n_final = s2n.max().item()
 
            metrics = br_reconstr.get_metrics( reconstructor_o.reshape(1,args.input_size[0],args.input_size[1],-1), None, datapoints, public_labels, neptune=False )
            closest_idx = metrics['selector'][0]
            del metrics['selector']
            del metrics['order']
            metrics['dsnr'] = s2n_final
            if args.neptune is not None:
                for k in metrics:
                    args.neptune[f"metrics/epoch_{current_epoch}/{k}_{l}"].log(metrics[k]);print(f"metrics/epoch_{current_epoch}/{k}_{l}",': ',metrics[k])
            if avg_metrics is None:
                avg_metrics = metrics
            else:
                for k in metrics:
                    avg_metrics[k] *= j/(j+1)
                    avg_metrics[k] += metrics[k]/(j+1)

            show(datapoints[closest_idx],f"epoch{current_epoch}_{l}_disagg_random_sample_true")
            show(reconstructor_o,f"epoch{current_epoch}_{l}_disagg_random_sample_rec")

            assert args.batch_size_train[1]==1, "(neg|pos)_ix_(i|o) are hardcoded"
            with torch.no_grad():
                if args.prop == 'bright':
                    order = datapoints.mean((1,2,3)).argsort(descending=False)
                elif args.prop == 'neg_bright':
                    order = datapoints.mean((1,2,3)).argsort(descending=True)
                else:
                    assert False, "Not a valid property"
                datapoints = datapoints[order]
                public_labels = public_labels[order]

                neg_ix_i=torch.arange(datapoints.shape[0]-1).cuda()
                pos_ix_i=torch.arange(datapoints.shape[0]-1,datapoints.shape[0]).cuda()
                neg_ix_o=torch.arange(1).cuda()
                #TODO hardcoded 1
                pos_ix_o=torch.arange(neg_ix_o.shape[0],neg_ix_o.shape[0]+pos_ix_i.shape[0]).cuda()

                jac=torch.zeros(neg_ix_i.shape[0]+pos_ix_i.shape[0],neg_ix_o.shape[0]+pos_ix_o.shape[0],device='cuda')
                assert pos_ix_i.shape[0]==pos_ix_o.shape[0], "compute precisely the grad for each positive sample"
                jac[pos_ix_i.unsqueeze(1),pos_ix_o.unsqueeze(0)]=torch.eye(pos_ix_i.shape[0],device='cuda')
                assert neg_ix_o.shape[0]==1, "aggregate all the negatives"
                jac[neg_ix_i.unsqueeze(1),neg_ix_o.unsqueeze(0)]=torch.ones(neg_ix_i.shape[0],neg_ix_o.shape[0],device='cuda').cuda()

            bdW, W = grad_ex(datapoints,public_labels.to(args.device),losses.public_loss,flat_cat=True,single_grad=False,jac=jac.t()) 
            assert bdW.shape[0]==jac.shape[1]
            disaggregator_o=disaggregator(bdW)
            reconstructor_o=reconstructor(disaggregator_o).reshape(-1,*args.input_size)
            with torch.no_grad():
                reconstructor_neg_tgt_o=reconstructor(torch.zeros_like(disaggregator_o)).reshape(-1,*args.input_size)
                tgt_o=reconstructor_neg_tgt_o
                tgt_o[ jac.shape[1] - 1 ] = datapoints[ datapoints.shape[0] - 1 ]

            neg_out=(reconstructor_o-tgt_o)[neg_ix_o]
            pos_out=(reconstructor_o)[pos_ix_o]

            show(pos_out,f"epoch{current_epoch}_{l}_disagg_sample_pos_out")
            show(neg_out,f"epoch{current_epoch}_{l}_disagg_sample_neg_out")
        
            if (j+1) % args.print_test_interval == 0:
                print('|',j+1 )

        print(f'{l} | ', end='')
        for k in metrics:
            if args.neptune is not None:
                args.neptune[f"metrics/avg_{k}_{l}"].log(avg_metrics[k]);print(f"metrics/avg_{k}_{l}",': ',avg_metrics[k])
            print( f"avg_{k}: {avg_metrics[k]} | ", end='' )
        print('')

def baseline_sec_aggr_end2end(args, modules, trainset, testset, checkpoint=None):
    public_model, grad_ex, disaggregator, reconstructor = modules
    torch.save({'par_sel': grad_ex.par_sel},'/tmp/runtime_dict')
    
    logfile=args.res_path+'/'+f"trainlog.{time.strftime('%y%m%d.%H%M%S', time.localtime())}"
    current_epoch=0
    
    factor = 1
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    if checkpoint is not None:
        public_model.load_state_dict(checkpoint['public_model_state_dict'])
        public_model.to(args.device)
        disaggregator.load_state_dict(checkpoint['disaggregator_state_dict'])
        disaggregator.to(args.device)
        reconstructor.load_state_dict(checkpoint['reconstructor_state_dict'])
        reconstructor.to(args.device)
        current_epoch=checkpoint['epoch']
        losses=Losses( public_model.parameters(), args )
        filters=Filters()
    else:
        losses=Losses( public_model.parameters(), args )
        filters=Filters()
    
    batch_sz = sum(args.batch_size_test)
    num_clients = args.num_clients
    ncs=[num_clients]
    test_sizes = [(nc,batch_sz) for nc in ncs]
    sets = { "train": trainset,"test": testset }
    loaders = {}
    for k in sets.keys():
        for i,j in test_sizes:
            loaders[k + f'_{i}x{j}'] = ( i, j, torch.utils.data.DataLoader( sets[k], batch_size=i*j, num_workers=2,sampler = torch.utils.data.RandomSampler(sets[k], replacement=False, num_samples=int(1e10)) ) )

    assert len(loaders) == 2
    br_reconstr = BreachingReconstruction( args, public_model, losses.public_loss, batch_sz, num_clients, dtype=torch.float32 )
    def show(px,sd):
        if len(px.shape)==3:
            reconstructed_user_data = {'data':px.reshape(1,args.input_size[0],args.input_size[1],-1), 'labels':public_labels.new_zeros(1) }
            br_reconstr.user.plot( reconstructed_user_data, neptune=args.neptune, subdir=f"{sd}" )
        else:
            assert len(px.shape)==4
            for px_i in px:
                show(px_i,sd)

    for l in loaders.keys():
        n_batches, bs, testing_set = loaders[l]
        data=[]
        avg_metrics = None
        for j, (images, labels) in enumerate(testing_set):
            
            if args.num_test_img is not None:
                if j >= args.num_test_img:
                    break

            flt=0
            datapoints=images.flatten(start_dim=0,end_dim=flt).to(args.device)

            if datapoints.shape[0] < testing_set.batch_size:
                continue

            if args.public_labels=='Zeroed':
                public_labels=torch.zeros_like(labels).flatten(start_dim=0,end_dim=flt).to(args.device)
            elif args.public_labels=='True':
                public_labels=labels.flatten(start_dim=0,end_dim=flt).to(args.device)
            else:
                assert False, "not allowed option for args.public_labels"

            print(f'test:{n_batches}x{bs}, id:{j}')

            grad_ex.par_sel = None
            bdW_sum = None
            for b in range( n_batches ):
                i_st = b * bs
                i_en = (b + 1) * bs
                bdW, _ = grad_ex( datapoints[ i_st : i_en ], public_labels[ i_st : i_en ], losses.public_loss, flat_cat=False, single_grad=True, testing=True ) # Sum aggr
                if bdW_sum is None:
                    bdW_sum = bdW
                else:
                    bdW_sum = [ g1 + g2 for g1, g2 in zip( bdW, bdW_sum ) ]
            bdW = []
            for d in bdW_sum:
                assert d.shape[0] == 1
                bdW.append( d[0] )

            data, metrics = br_reconstr.reconstruct_best_breaching(bdW, datapoints, public_labels, 'full_batch')
            rec_idx = metrics['order'][0]
            true_idx = metrics['selector'][0]
            del metrics['selector']
            del metrics['order']
            if args.neptune is not None:
                for k in metrics:
                    if args.neptune is not None:
                        args.neptune[f"metrics/{k}_{l}"].log(metrics[k]);
                    print(f"metrics/{k}_{l}",': ',metrics[k])

            show(data[ rec_idx ],f"{l}_disagg_random_sample_rec")
            show(datapoints[ true_idx ],f"{l}_disagg_random_sample_true")
        
            if (j+1) % args.print_test_interval == 0:
                print('|',j+1 )

        print(f'{l} | ', end='')
        for k in metrics:
            if args.neptune is not None:
                args.neptune[f"metrics/avg_{k}_{l}"].log(avg_metrics[k]);print(f"metrics/avg_{k}_{l}",': ',avg_metrics[k])
            print( f"avg_{k}: {avg_metrics[k]} | ", end='' )
        print('')
