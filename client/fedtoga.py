import torch
from .client import Client
from utils import *
from optimizer import *
from thop import profile

class fedtoga(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):   
        super(fedtoga, self).__init__(device, model_func, received_vecs, dataset, lr, args)
        
        # rebuild
        delta_model_sel = set_client_from_params(self.device, model_func(), self.received_vecs['Local_dual_correction'])
        self.base_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay+self.args.lamb)
        self.base_optimizer.add_param_group({'params': delta_model_sel.parameters()})
        self.optimizer = TSAM(self.model.parameters(), self.base_optimizer, rho=self.args.rho, dataset= self.args.dataset)

    

    def train(self):
        # local training
        self.model.train()
        if self.args.dataset == "AG_News":
            for k in range(self.args.local_epochs):
                for i, (inputs, labels,lengths) in enumerate(self.dataset):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).reshape(-1).long()
                    lengths = lengths.to(self.device)

                    self.optimizer.paras = [inputs, labels, lengths, self.loss, self.model]
                    self.optimizer.step()

                    param_list = param_to_vector(self.model)
                    loacal_delta_list = self.received_vecs['Delta_list'].to(self.device)
                    delta_list = self.received_vecs['Local_dual_correction'].to(self.device)
                    loss_algo = self.args.lamb * torch.sum(param_list * delta_list)
                    loss_delta = self.args.lamb1 * torch.sum(param_list * loacal_delta_list)

                    loss_correct = loss_algo + loss_delta
                    
                    loss_correct.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm) 
                    self.base_optimizer.step()

        else:
            for k in range(self.args.local_epochs):
                for i, (inputs, labels) in enumerate(self.dataset):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).reshape(-1).long()

                    self.optimizer.paras = [inputs, labels, self.loss, self.model]
                    self.optimizer.step()

                    param_list = param_to_vector(self.model)
                    loacal_delta_list = self.received_vecs['Delta_list'].to(self.device)
                    delta_list = self.received_vecs['Local_dual_correction'].to(self.device)
                    loss_algo = self.args.lamb * torch.sum(param_list * delta_list)
                    loss_delta = self.args.lamb1 * torch.sum(param_list * loacal_delta_list)

                    loss_correct = loss_algo + loss_delta
                    
                    loss_correct.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm) 
                    self.base_optimizer.step()
                
        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs


    
    