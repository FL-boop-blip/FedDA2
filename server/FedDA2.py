import torch
from client import *
from .server import *
import numpy as np
from collections import Counter

class FedDA2(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):   
        super(FedDA2, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        self.dual_steps = torch.ones(args.total_client, dtype=torch.float32)
        self.h_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        self.h_params_list_s = torch.zeros((args.total_client, init_par_list.shape[0]))
        self.T = 4.0
        print(" Dual Variable Param List  --->  {:d} * {:d}".format(
                self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))
        
        # rebuild
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'Local_dual_correction': torch.zeros((init_par_list.shape[0])),
        }
        self.client_models = list(range(self.args.total_client))
        self.Client = fedda2

    def _select_clients_(self, t):
        # select active clients ID
        inc_seed = 0
        while(True):
            np.random.seed(t + self.args.seed + inc_seed)
            act_list = np.random.uniform(size=self.args.total_client)
            act_clients = act_list <= self.args.active_ratio
            selected_clients = np.sort(np.where(act_clients)[0])
            
            inc_seed += 1
            if len(selected_clients) != 0:
                return selected_clients
            
    def _gradient_diversity_(self, t):
        client_gradients = []
        normalized_sums = 0.0
        g_gradient = np.zeros(self.server_model_params_list.shape)
        Averaged_update = torch.zeros(self.server_model_params_list.shape)
        w_l = np.asarray([len(self.datasets.client_y[client]) for client in range(self.args.total_client)])
        w_l = w_l / np.sum(w_l) * self.args.total_client
        for client in self.client_models:
            print('pre-training for gradient diversity {}'.format(client))
            dataset = data.DataLoader(Dataset(self.datasets.client_x[client],\
                         self.datasets.client_y[client], train=True, dataset_name=self.args.dataset),\
                             batch_size=self.args.batchsize, shuffle=True)
            self.process_for_communication(client, Averaged_update)
            _edge_device = self.Client(device=self.device, model_func=self.model_func, received_vecs=self.comm_vecs,
                                          dataset=dataset, lr=self.lr, args=self.args)
            client_gradient = _edge_device.compute_gradient()
            client_gradients.append(client_gradient)
            del _edge_device
            if len(client_gradients) == 50:
                for i, grad in enumerate(client_gradients):
                    g_gradient += 1/ self.args.total_client * client_gradients[i]
                    grad_norm_squared = np.linalg.norm(grad,2) ** 2
                    normalized_sums += 1 / self.args.total_client * grad_norm_squared * w_l[i]
                client_gradients.clear()
        total_gradient_norm = np.linalg.norm(g_gradient,2) ** 2
        diversity = np.sqrt(normalized_sums / total_gradient_norm)
        print('diversity is {}'.format(diversity))
        return diversity


    
    
    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta\
                                    * (self.server_model_params_list - self.clients_params_list[client]))
        
        # combination of dual variable and global server model
        # local gradient is g - hi + alpha(wk - wt)
        #                  ---       --------
        #                grad      weight-decay
        # Therefore, -hi - alpha*wt are communicated as Local_dual_correction term
        # self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.server_model_params_list)
        self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.comm_vecs['Params_list'])

    
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # FedDyn (ServerOpt)
        # w(t+1) = average_s[wi(t)] + average_c[h(t)]
        return Averaged_model + torch.mean(self.h_params_list_s, dim=0)
    
    
    def postprocess(self, client,received_vecs,diversity):
        self.h_params_list[client] += diversity * self.clients_updated_params_list[client]
        self.h_params_list_s[client] += 1 / self.args.active_ratio * self.args.beta4 * self.clients_updated_params_list[client]                           


    def train(self):
        print("##=============================================##")
        print("##           Training Process Starts           ##")
        print("##=============================================##")
        
        Averaged_update = torch.zeros(self.server_model_params_list.shape)
        
        for t in range(self.args.comm_rounds):
            if t % 100 == 0:
                diversity = self._gradient_diversity_(t)
            start = time.time()
            # select active clients list
            selected_clients = self._select_clients_(t)
            print('============= Communication Round', t + 1, '=============', flush = True)
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clients])))
            for client in selected_clients:
                dataset = data.DataLoader(Dataset(self.datasets.client_x[client],\
                        self.datasets.client_y[client], train=True, dataset_name=self.args.dataset),\
                            batch_size=self.args.batchsize, shuffle=True)
                if t > (self.args.comm_rounds / 2) and self.T:
                    self.Client = fedda2_t
                else:
                    self.Client = fedda2

                self.process_for_communication(client, Averaged_update)
                _edge_device = self.Client(device=self.device, model_func=self.model_func, received_vecs=self.comm_vecs,
                                        dataset=dataset, lr=self.lr, args=self.args)
                self.received_vecs = _edge_device.train()
                self.clients_updated_params_list[client] = self.received_vecs['local_update_list']
                self.clients_params_list[client] = self.received_vecs['local_model_param_list']
                self.postprocess(client,self.received_vecs,diversity)
                
                # release the salloc
                del _edge_device
            
            # calculate averaged model
            Averaged_update = torch.mean(self.clients_updated_params_list[selected_clients], dim=0)
            Averaged_model  = torch.mean(self.clients_params_list[selected_clients], dim=0)
            
            self.server_model_params_list = self.global_update(selected_clients, Averaged_update, Averaged_model)
            set_client_from_params(self.device, self.server_model, self.server_model_params_list)
            
            
            self._test_(t, selected_clients)
            self._lr_scheduler_()
            
            # time
            end = time.time()
            self.time[t] = end-start
            print("            ----    Time: {:.2f}s".format(self.time[t]), flush = True)
    
            
        
        self._save_results_()
        self._summary_()