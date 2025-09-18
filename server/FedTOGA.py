import torch
from client import *
from .server import Server


class FedTOGA(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):   
        super(FedTOGA, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        
        self.h_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        self.delta_list = torch.zeros((init_par_list.shape[0]))
        self.local_iteration = self.args.local_epochs * (self.datasets.client_x[0].shape[0] / self.args.batchsize)
        print(" Dual Variable Param List  --->  {:d} * {:d}".format(
                self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))
        
        # rebuild
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'Local_dual_correction': torch.zeros((init_par_list.shape[0])),
            'Delta_list': torch.zeros((init_par_list.shape[0]))
        }
        self.Client = fedtoga
    
    
    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta\
                                    * (self.server_model_params_list - self.clients_params_list[client]))
        
        # self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.server_model_params_list)
        self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.comm_vecs['Params_list'])
        self.comm_vecs['Delta_list'].copy_(self.delta_list)

    
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # FedSpeed (ServerOpt)
        # w(t+1) = average_s[wi(t)] + average_c[h(t)]
        self.delta_list = -1. / self.local_iteration * torch.mean(self.clients_updated_params_list[selected_clients], dim=0)
        return Averaged_model + torch.mean(self.h_params_list, dim=0)
    
    
    def postprocess(self, client,received_vecs):
        self.h_params_list[client] += self.clients_updated_params_list[client]