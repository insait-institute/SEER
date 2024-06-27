from train import *
import torchvision
from data import *
from PIL import Image
import time
from model import *

args = get_args()
torch.manual_seed(args.rng_seed)
torch.set_default_dtype(torch.float32)


# CIFAR/TinyImageNet first layer
# https://github.com/Westlake-AI/openmixup/blob/main/openmixup/models/backbones/resnet_mmcls.py#L716
assert args.dataset in ['Cifar10', 'Cifar100', 'TinyImageNet', 'TinyImageNet_rsz', 'Cifar10_2', 'Cifar10_1', 'Isic2019'] or args.dataset.startswith( 'Cifar10_C' )
public_model = ResNet(BasicBlock, [2, 2, 2, 2], [64,128,256,512], args.act, num_classes=args.num_classes).to(args.device)

par_sel=ParamSelector(public_model,args.par_sel_size,args.par_sel_frac,sparse_grad=False,seed=98)
grad_ex=GradientExtractor(public_model,par_sel).to(args.device)
disaggregator, reconstructor = get_decoder(args, grad_ex)

optimizer = optim.Adam( list(public_model.parameters())+list(disaggregator.parameters())+list(reconstructor.parameters()), lr=args.learning_rate)

if args.dataset.startswith('Cifar10_C'):
    C, sev = args.dataset[10:].rsplit( '_', 1 )
    sev = int(sev)
    trainset, testset = globals()[f'datasets_Cifar10_C'](C,sev)
else:
    trainset, testset = globals()[f'datasets_{args.dataset}']()
if (args.prop_mode == 'thresh') and (args.thresh is None) and (args.task!="secagge2e"):
    args.thresh=distr.compute_thresh(dataset=trainset,prop=args.prop,batch_size=(args.batch_size_train[0] + args.batch_size_train[1]),num_clients=args.num_clients,num_samples=args.est_thr,bn=distr.batch_norm)
    print(f"THRESHOLD for (trainset={args.dataset},prop={args.prop},batch_size={(args.batch_size_train[0] + args.batch_size_train[1])},num_clients={args.num_clients}): ",args.thresh)


if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
else:
    checkpoint = None
modules=[public_model, grad_ex, disaggregator, reconstructor]
if args.task == 'train': 
    train(args, modules, optimizer, trainset, testset, checkpoint=checkpoint)
elif args.task == 'test':
    tests(args, modules, trainset, testset, checkpoint=checkpoint)
elif args.task == 'end2end':
    test_end2end(args, modules, trainset, testset, checkpoint=checkpoint)
elif args.task == 'end2end_contrast':
    test_end2end_fix_contrast(args, modules, trainset, testset, checkpoint=checkpoint)
elif args.task == 'secaggr':
    test_sec_aggr(args, modules, trainset, testset, checkpoint=checkpoint,metr=True)
elif args.task == 'secagge2e':
    test_sec_aggr_end2end(args, modules, trainset, testset, checkpoint=checkpoint,metr=True)
elif args.task == 'baseline':
    baseline_sec_aggr_end2end(args, modules, trainset, testset, checkpoint=checkpoint)
elif args.task == 'tests':
    test_sec_aggr(args, modules, trainset, testset, checkpoint=checkpoint)
    test_sec_aggr_end2end(args, modules, trainset, testset, checkpoint=checkpoint)
else:
    tests(args, modules, loader_train, loader_test,checkpoint=checkpoint, vis_res=True)
