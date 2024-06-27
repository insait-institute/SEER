import torch
import logging, sys
sys.path.insert(1, '../breaching')
import breaching
from breaching.cases.users import construct_user

from train import *
from data import *
from PIL import Image
import time
from model import *
import logging, sys

class BreachingReconstruction:
    def __init__(self, args, model, loss, batch_size, num_clients, dtype=torch.float64):
        assert isinstance( loss, torch.nn.CrossEntropyLoss )
        loss = torch.jit.script(torch.nn.CrossEntropyLoss(reduction='sum'))
 
        self.device = args.device
        self.setup = dict(device=torch.device(self.device), dtype=dtype)
        self.cfg = breaching.get_config(overrides=[f"attack={args.attack_cfg}", f"case={args.case_cfg}"])
        print( self.cfg.attack.objective.type )
        
        self.cfg.case.user.num_local_updates = 1
        self.cfg.case.user.num_data_points = batch_size
        self.cfg.case.user.num_data_per_local_update_step = batch_size
        self.cfg.case.user.user_range = [ 0, num_clients ]
        self.cfg.case.data.default_clients = num_clients
        self.num_clients = num_clients
        self.batch_size = batch_size

        self.user = construct_user(model, loss, self.cfg.case, self.setup)
        self.attacker = breaching.attacks.prepare_attack(model, loss, self.cfg.attack, self.setup)

        self.neptune = args.neptune
        if self.cfg.attack.optim.callback > 0 :
            logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
            self.logger = logging.getLogger()

    def get_metrics(self, rec, rec_label, datapoints, labels, neptune=True):    
        model = self.user.model
        true_user_data = {'data':datapoints, 'labels':labels}

        # Server Info:
        server_payload = [
            dict(
                parameters=[p for p in model.parameters()], buffers=[b for b in model.buffers()], metadata=self.cfg.case.data
            )
        ]
        reconstructed_user_data = {'data': rec, 'labels': rec_label }
        neptune = self.neptune if neptune else None

        metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, server_payload, model, compute_full_iip=False, 
                cfg_case=None, setup=self.setup, neptune=neptune)

        if rec_label is None:
            del metrics['label_acc']
        del metrics['parameters']
        del metrics['IIP-none']
        return metrics

    def reconstruct_best_breaching(self, grad, datapoint, label, subdir):    
        assert datapoint.shape[0] == self.num_clients * self.batch_size
        model = self.user.model
        true_user_data = {'data':datapoint, 'labels':label}
        self.user.plot( true_user_data, neptune=self.neptune, subdir=f"{subdir}_true" )

        # Server Info:
        server_payload = [
            dict(
                parameters=[p for p in model.parameters()], buffers=[b for b in model.buffers()], metadata=self.cfg.case.data
            )
        ]

        # Client Info shared with Server:
        shared_data = [
            dict(
                gradients=tuple(grad),
                buffers=None,
                metadata=dict(num_data_points=label.shape[0], labels=label, num_users=self.num_clients, local_hyperparams=None),
            )
        ]

        # Attack:
        reconstructed_user_data, stats = self.attacker.reconstruct(server_payload, shared_data, {}, dryrun=False, neptune=self.neptune)
        reconstructed_user_data['labels'] = None
        self.user.plot( reconstructed_user_data, neptune=self.neptune, subdir=f"{subdir}_rec" )

        min_lpips = None
        opt_metrics = None
        for i in range(reconstructed_user_data['data'].shape[0]): 
            tmp = reconstructed_user_data['data'].clone()
            reconstructed_user_data['data'] = reconstructed_user_data['data'][i : i + 1] 
            metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, server_payload, model, compute_full_iip=False,cfg_case=None, setup=self.setup, neptune=self.neptune)
            reconstructed_user_data['data'] = tmp
            
            del metrics['label_acc']
            del metrics['parameters']
            del metrics['IIP-none']

            if min_lpips is None or metrics['lpips'] < min_lpips:
                min_lpips = metrics['lpips']
                opt_metrics = metrics
                opt_metrics['order'][0] = i

        return reconstructed_user_data['data'], opt_metrics
