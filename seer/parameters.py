import sys
import argparse
import time
import torch

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='FMS attack')


    # Neptune
    parser.add_argument('--neptune', type=str, help='Neptune project name, leave empty to not use neptune', default=None)
    parser.add_argument('--neptune_label', dest='label', type=str, help='Neptune label of the experiment')
    parser.add_argument('--neptune_offline', action='store_true', help='Run Neptune in offline mode')

    
    # Setup
    parser.add_argument('--rng_seed', type=int, default=42)
    parser.add_argument('--eps', type=float, default=1e-9)
    ##parser.add_argument('--clamp', type=float, default=1e8)
    ##parser.add_argument('--pos_volume', type=int, default=1, help='How many positive examples we can fit in memory at the same time')
    ##parser.add_argument('--neg_volume', type=int, default=None, help='How many negative examples we can fit in memory at the same time')
    parser.add_argument('--data_path', type=str, help='Path to data', default='../data')
    parser.add_argument('--data_loaders', type=str, help='Path to stored data loaders')
    parser.add_argument('--res_path', type=str, help='Path to resulting model', default='../models')
    parser.add_argument('--checkpoint', type=str, help='Path to model to load', default=None)
    parser.add_argument('--print_interval', type=int, default=50, help='How often to print')
    parser.add_argument('--print_test_interval', type=int, default=1, help='How often to print in test')
    parser.add_argument('--task', type=str, choices=['train', 'test', 'end2end', 'end2end_contrast', 'secaggr', 'secagge2e','baseline','tests'], default='train', help='Testing or training')
    ##parser.add_argument('--input_reduce_to', type=str, default=None, help='number of rand variables to reduce the input to before computing the correlation matrix, format: [[<num_vars_N> <as_norm_from_M_original>]...]')
    ##parser.add_argument('--grad_reduce_to', type=str, default=None, help='number of rand variables to reduce the grads to before computing the correlation matrix, format: [[<num_vars_N> <each_as_norm_from_M_original>]...]')

    # Training params
    parser.add_argument('-e','--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--epoch_steps', type=int, default=1000, help='Number of sampled batches per epoch')
    parser.add_argument('--current_epoch', type=int, default=None, help='epoch to start from')
    parser.add_argument('--acc_grad', type=int, default=1, help='accumulate the gradients before taking a step')
    parser.add_argument('--disag_size', type=int, help='another way to define the desired frequency of the property')
    parser.add_argument('--num_clients', type=int, default=1, help='the number of clients')
    parser.add_argument('--est_thr', type=int, help='number of samles to estimate the thresh')
    parser.add_argument('-lr','--learning_rate', type=float, default=1e-5, help='Learning rate')
    ##parser.add_argument('--neg_pos_log2_rat', type=float, default=0, help='base 2 logarithm of the ratio between negative and positive term in the objective')
    parser.add_argument('--sched_x_1', type=float, default=0, help='base 2 logarithm of the ratio between negative and positive term in the objective')
    parser.add_argument('--sched_x_end', type=float, help='base 2 logarithm of the ratio between negative and positive term in the objective')
    parser.add_argument('--sched_y_1', type=float, default=-2, help='base 2 logarithm of the ratio between negative and positive term in the objective')
    parser.add_argument('--sched_y_end', type=float, help='base 2 logarithm of the ratio between negative and positive term in the objective')
    parser.add_argument('-btr','--batch_size_train', type=str, default=8, help='Batch size for training')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Torch device')

    # Dataset params
    parser.add_argument('-d','--dataset', default='Cifar10')
    parser.add_argument('--num_classes', type=int, default=None, help="Number of classes in the dataset if changed")
    parser.add_argument('--input_size', type=int, default=None, help="W and H of the image")
    
    # Properties
    ##parser.add_argument('--num_quantiles', type=int, default=100)
    ##parser.add_argument('--num_ratios', type=int, default=3, help='Max number of combinations for noise-to-ratio loss')
    parser.add_argument('--prop', type=str, choices=['bright', 'dark', 'red','blue','green', 'hedge', 'vedge','rand_conv'], default='bright', help='Network activation function')
    parser.add_argument('--prop_conv', default=None, help='property kernel function')
    parser.add_argument('--prop_mode', type=str, choices=['max', 'thresh'], required=True, help='whether to take the maximum of the property, or at a given threshold')
    parser.add_argument('--thresh', type=float, help='threshold after batch normalization')
    
    # FMS params
    parser.add_argument('--decoder', type=str, choices=['orig', 'nomid', 'deconv'], default='nomid', help='whether to take the maximum of the property, or at a given threshold')
    ##parser.add_argument('--grad_size_reg', type=float, default=1e-2, help='Hyperparam for grad size reguralizer')
    ##parser.add_argument('-gp','--grad_percent', type=float, default=0.05, help='Hyperparam for grad percentage')
    ##parser.add_argument('-mnsr','--mnsr', type=float, default=1e-4, help='Hyperparam for mean noise-to-signal ratio between positives and negatives')
    ##parser.add_argument('-mre','--mre', type=float, default=1e-4, help='Hyperparam for mean reconstruction error')
    ##parser.add_argument('-mgze','--mgze', type=float, default=1e-6, help='Hyperparam for mean zero-gradient error for negatives')
    ##parser.add_argument('-pp','--property_prob', type=float, default=0.5, help='Probability of chosen property')
    ##parser.add_argument('--bin_size', type=int, default=28*28*5, help='Binarizer\'s output size')
    ##parser.add_argument('--proj_size', type=int, default=28*28*5*2, help='WeightProjector\'s output size')
    parser.add_argument('--par_sel_size', type=int, default=1, help='number of parameters to select per layer; the maximum of the number and the fraction is taken')
    parser.add_argument('--par_sel_frac', type=float, default=0.0, help='fraction of parameters to select per layer; the maximum of the number and the fraction is taken')
    parser.add_argument('--mid_rep_frac', type=float, default=0.0, help='fraction of parameters to select per layer; the maximum of the number and the fraction is taken')
    ##parser.add_argument('--lr_mn_deg', type=float, default=0.0, help='degree of layer size when averaging accross layers')
    ##parser.add_argument('--pr_mn_deg', type=float, default=2.0, help='degree of layer size when averaging accross layers')
    parser.add_argument('--num_properties', type=int, default=2, help='Number of properties')
    ##parser.add_argument('--swap_interval', type=int, default=10, help='How often to change factor between losses')
    parser.add_argument('--act', type=str, choices=['LeakyReLU', 'ReLU', 'Softplus'], default='ReLU', help='Network activation function')
    parser.add_argument('--public_labels', type=str, choices=['Zeroed', 'True'], default='True', help='Data labels to train with')
    ##parser.add_argument('--loss_type', type=str, choices=['n2s', 'tgtr'], default='n2s', help='Type of the objective')
    ##parser.add_argument('--epoch_warmup', type=str, default=None, help='first epochs will be partial for the purpose of more frequent testing; list the desired fraction to break after  for the first N epochs; format: [<frac1>_<frac2>_...]')

    # Testing params
    parser.add_argument('--attack_cfg', type=str, default='modern', help='What attack config from Geiping to use')
    parser.add_argument('--test_interval', type=int, default=1, help='Test every x epochs')
    parser.add_argument('--big_test_interval', type=int, default=100, help='Test every x epochs')
    parser.add_argument('-bte','--batch_size_test', type=str, default=8, help="Batch size for testing")
    parser.add_argument('--num_test_img', type=int, default=5, help="How many images to test with")
    parser.add_argument('--case_cfg', type=str, default=None, help='What case config from Geiping to use')
    ###parser.add_argument('--pos_pick', type=int, default=1, help='which positive sample from the batch to choose; 0 stands for random, 1 is highest, 2 is second highest, etc...')

    if argv is None:
        argv = sys.argv[1:]
    args=parser.parse_args(argv)

    if isinstance( args.batch_size_train, str) and args.batch_size_train.count('_') == 2:
       args.batch_size_train, jac_size = args.batch_size_train.rsplit("_",1)
       args.jac_size = int(jac_size)
    
    if args.num_classes is None:
        if args.dataset == 'Cifar10':
            args.num_classes = 10
        elif args.dataset == 'Cifar100':
            args.num_classes = 100
        elif args.dataset == 'Cifar10_2':
            args.num_classes = 10
        elif args.dataset == 'Cifar10_1':
            args.num_classes = 10
        elif args.dataset.startswith('Cifar10_C'):
            args.num_classes = 10
        elif args.dataset == 'TinyImageNet_rsz':
            args.num_classes = 10
        elif args.dataset == 'TinyImageNet':
            args.num_classes = 200
        else:
            assert False, "Wrong dataset"    
    if args.input_size is None:
        if args.dataset == 'Cifar10' or args.dataset == 'Cifar100' or args.dataset == 'Cifar10_2' or args.dataset == 'Cifar10_1' or args.dataset.startswith('Cifar10_C'):
            args.input_size = (3, 32, 32)
        elif args.dataset == 'TinyImageNet_rsz':
            args.input_size = (3, 32, 32)
        elif args.dataset == 'TinyImageNet':
            args.input_size = (3, 64, 64)
        else:
            assert False, "Wrong dataset"
    else:
        args.input_size = (3, args.input_size, args.input_size)
 
    if args.case_cfg is None:
        if args.dataset.startswith('Cifar10_C'):
            args.case_cfg = f'Cifar10_C_sanity_check'
        else:
            args.case_cfg = f'{args.dataset}_sanity_check'

    parsed_und=args.batch_size_train.split('_')
    btr=[int(e) for e in parsed_und]
    args.batch_size_train=btr
    parsed_und=args.batch_size_test.split('_')
    bte=[int(e) for e in parsed_und]
    args.batch_size_test=bte
    if args.sched_x_end is None:
        args.sched_x_end = args.num_epochs//2

    if args.sched_y_end is None:
        args.sched_y_end = torch.log2(torch.tensor(args.num_clients*(args.batch_size_train[0] + args.batch_size_train[1])))

    #args.num_ratios = min(args.num_ratios, args.batch_size_train )
    if args.neptune is not None:
        import neptune.new as neptune
        assert('label' in args)
        nep_par = { 'project':f"{args.neptune}", 'source_files':["*.py","../t*.sh","../breaching/breaching/config/attack/modern_*.yaml"] } 
        if args.neptune_offline:
            nep_par['mode'] = 'offline'
            args.neptune_id = 'FMS-0'
        
        run = neptune.init( **nep_par )
        run["parameters"] = args
        args.neptune = run
        if not args.neptune_offline:
            print('waiting...')
            start_wait=time.time()
            args.neptune.wait()
            print('waited: ',time.time()-start_wait)
            args.neptune_id = args.neptune['sys/id'].fetch()
        print( '\n\n\nArgs:', *argv, '\n\n\n' ) 
    return args
