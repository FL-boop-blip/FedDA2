from utils import *
from .client import Client
from optimizer import *

class fedda2_t(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):   
        super(fedda2_t, self).__init__(device, model_func, received_vecs, dataset, lr, args)
        
        # rebuild
        self.base_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)
        self.optimizer = ESAM(self.model.parameters(), self.base_optimizer, rho=self.args.rho, dataset=self.args.dataset)
        self.loss_t = DistillKL(4.0)

    
    def compute_gradient(self):
        total_gradient = None
        for i, (inputs, labels) in enumerate(self.dataset):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).reshape(-1).long()
            
            predictions = self.model(inputs)
            loss_pred = self.loss(predictions, labels)
            self.model.zero_grad()
            loss_pred.backward()
            current_grad = np.concatenate([param.grad.cpu().numpy().flatten() for param in self.model.parameters()])
            
            if total_gradient is None:
                total_gradient = current_grad.copy()
            else:
                total_gradient += current_grad
        
        return total_gradient
            


    def train(self):
        # local training
        self.model.train()
        if self.args.dataset == "AG_News":
            for k in range(self.args.local_epochs):
                for i, (inputs, labels, lengths) in enumerate(self.dataset):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).reshape(-1).long()
                    lengths = lengths.to(self.device)


                    self.optimizer.paras = [inputs, labels, lengths, self.loss, self.model]
                    self.optimizer.step()
                    predictions = self.model(inputs)
                    predictions_t = self.server_model(inputs,lengths)
                    loss_pred_t = self.loss_t(predictions, predictions_t)

                    param_list = param_to_vector(self.model)
                    delta_list = self.received_vecs['Local_dual_correction'].to(self.device)
                    loss_correct = self.args.lamb / 2 * torch.sum((param_list + delta_list) * (param_list + delta_list)) + loss_pred_t
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
                    predictions = self.model(inputs)
                    predictions_t = self.server_model(inputs,lengths)
                    loss_pred_t = self.loss_t(predictions, predictions_t)

                    param_list = param_to_vector(self.model)
                    delta_list = self.received_vecs['Local_dual_correction'].to(self.device)
                    loss_correct = self.args.lamb / 2 * torch.sum((param_list + delta_list) * (param_list + delta_list)) + loss_pred_t
                    loss_correct.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm) 
                    self.base_optimizer.step()
                
        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs