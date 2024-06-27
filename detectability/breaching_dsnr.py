import breaching
import torch
import matplotlib.pyplot as plt
import logging, sys
from breaching.cases.malicious_modifications.classattack_utils import print_gradients_norm, cal_single_gradients
import random
import numpy as np

def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def main(multiplier):
    set_all_seeds(11) 
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
    logger = logging.getLogger()

    # Load cfg for CIFAR 10
    cfg = breaching.get_config(overrides=["case/server=malicious-fishing", "attack=clsattack", "case/data=CIFAR10"])
    print(torch.cuda.is_available())
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
    cfg.case.data.partition = "unique-class" # This is the worst-case for the attack, as each user owns a unique class
    cfg.case.server.feat_multiplier = multiplier
    cfg.case.user.num_data_points = 64 # maximum number of data points in the validation set
    cfg.case.user.user_idx = 1
    cfg.case.user.provide_labels = True # Mostly out of convenience
    cfg.case.server.target_cls_idx = 0 # Which class to attack, if multiple are present?
    cfg.attack.regularization.total_variation.scale = 1e-3
    cfg.attack.optim.max_iterations = 5000
    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    breaching.utils.overview(server, user, attacker)
    [shared_data], [server_payload], true_user_data = server.run_protocol(user)
    #print(server_payload)

    # Check which percentage of weights of resnet are linear weights?
    total_sz = 0
    conv_sz = 0
    for m in model.named_parameters():
        if 'conv' in m[0] or 'linear' in m[0]:
            print(m[0])
            conv_sz += m[1].numel()
        total_sz += m[1].numel()
    print(f'Percentage of resnet18 that is linear weights: {conv_sz / total_sz}')

    # get norms
    def get_grads_per_layer(model, loss_fn, true_user_data, setup):
        true_data = true_user_data["data"]
        num_data = len(true_data)
        labels = true_user_data["labels"]
        model = model.to(**setup)

        single_gradients = []
        single_losses = []
        all_norms = []

        for ii in range(num_data):
            cand_ii = true_data[ii : (ii + 1)]
            label_ii = labels[ii : (ii + 1)]
            model.zero_grad()
            spoofed_loss_ii = loss_fn(model(cand_ii), label_ii)
            gradient_ii = torch.autograd.grad(spoofed_loss_ii, model.parameters())

            norms = []
            for nmm, grad in zip(model.named_parameters(), gradient_ii):
                nm = nmm[0]
                if nm == 'linear.weight' or ('.conv' in nm and '.weight' in nm):
                    norms.append(torch.linalg.vector_norm(grad.reshape(-1)).item()) # norms for this example
            all_norms.append(norms) 
            single_losses.append(spoofed_loss_ii)
        # all_norms: [examples x layers]
        wns = torch.transpose(torch.tensor(all_norms), 0, 1)
        return wns 

    # Print gradients to see if disaggregation succeeded
    single_gradients, single_losses = cal_single_gradients(user.model, loss_fn, true_user_data, setup=setup)
    print_gradients_norm(single_gradients, single_losses)

    # Get all norms and calculate dsnr
    wns = get_grads_per_layer(user.model, loss_fn, true_user_data, setup=setup)
    dsnr = wns.max(dim=1)[0] / ( wns.sum(dim=1) - wns.max(dim=1)[0])
    print(f'dsnr per layer: {dsnr}')
    dsnr_final = dsnr.max()
    print(f'dsnr_final: {dsnr_final}')

    def denormalize(img):
        metadata = server_payload["metadata"]
        if hasattr(metadata, "mean"):
            dm = torch.as_tensor(metadata.mean, **setup)[None, :, None, None]
            ds = torch.as_tensor(metadata.std, **setup)[None, :, None, None]
        else:
            dm, ds = torch.tensor(0, **setup), torch.tensor(1, **setup)
        return torch.clamp(img * ds + dm, 0, 1)


    ################ Reconstruction
    reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], server.secrets, dryrun=cfg.dryrun)
    fished_data = dict(data=reconstructed_user_data["data"][server.secrets["ClassAttack"]["target_indx"]][None], labels=None)
    fished = denormalize(fished_data['data'][0])
    true = denormalize(true_user_data['data'][19])
    plt.imsave(f'images/fishing_M={multiplier},DSNR={dsnr_final:.2f}_rec.png', fished[0].permute(1, 2, 0).cpu().numpy()) # SAVE IMAGE
    plt.imsave(f'images/fishing_M={multiplier},DSNR={dsnr_final:.2f}_true.png', true[0].permute(1, 2, 0).cpu().numpy()) # SAVE IMAGE
    print('saved images')
    
##########################################################

if __name__ == "__main__":
    print(sys.argv)
    multiplier = int(sys.argv[1])
    main(multiplier)
