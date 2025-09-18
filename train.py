import torch
import argparse

from utils import *
from models import *
from server import *
from dataset import *
from server.FedSGAM_KL import FedSGAM_KL
from server.SparsyFed_adan_dp import SparsyFed_adan_dp
#### ================= Open Float32 in A100 ================= ####
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#### ================= Open ignore warining ================= ####
import warnings
warnings.filterwarnings('ignore')
#### ======================================================== ####
print("##=============================================##")
print("##     Federated Learning Simulator Starts     ##")
print("##=============================================##")

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['normal'], type=str, default='normal')
parser.add_argument('--dataset', choices=['mnist', 'CIFAR10', 'CIFAR100','AG_News'], type=str, default='CIFAR10')             # select dataset
parser.add_argument('--model', choices=['mnist_2NN','ResNet18','ResNet18P','ResNet18_100','AG_News_NN', 'LeNet','ResNet18_sparsy','ResNet18_fusion','LeNet_fusion'], type=str, default='ResNet18')                    # select model
parser.add_argument('--non-iid', action='store_true', default=False)                                       # activate if use heterogeneous dataset 
parser.add_argument('--split-rule',default='Dirichlet')  # select the dataset splitting rule
parser.add_argument('--split-coef', default=0.6, type=float)                                                 # --> if Dirichlet: select the Dirichlet coefficient (i.e. 0.1, 0.3, 0.6, 1)
                                                                                                             # --> if Pathological: select the Dirichlet coefficient (i.e. 3, 5)
parser.add_argument('--active-ratio', default=0.1, type=float)                                              # select the partial participating ratio (i.e. 0.1, 0.05)
parser.add_argument('--total-client', default=100, type=int)                                               # select the total number of clients (i.e. 100, 500)
parser.add_argument('--comm-rounds', default=1000, type=int)                                               # select the global communication rounds T
parser.add_argument('--local-epochs', default=5, type=int)                                                 # select the local interval K
parser.add_argument('--batchsize', default=50, type=int)                                                   # select the batchsize
parser.add_argument('--weight-decay', default=0.001, type=float)                                           # select the weight-decay (i.e. 0.01, 0.001)
parser.add_argument('--local-learning-rate', default=0.1, type=float)                                      # select the local learning rate (generally 0.1 expect for local-adaptive-based)
parser.add_argument('--global-learning-rate', default=1.0, type=float)                                     # select the global learning rate (generally 1.0 expect for global-adaptive-based)
parser.add_argument('--lr-decay', default=0.998, type=float)                                               # select the learning rate decay (generally 0.998 expect for proxy-based)
parser.add_argument('--seed', default=20, type=int)                                                        # select the random seed
parser.add_argument('--cuda', default=0, type=int)                                                         # select the cuda ID
parser.add_argument('--data-file', default='./', type=str)                                                 # select the path of the root of Dataset
parser.add_argument('--out-file', default='out/', type=str)                                                # select the path of the log files
parser.add_argument('--save-model', action='store_true', default=False)                                    # activate if save the model
parser.add_argument('--use-RI', action='store_true', default=False)                                        # activate if use relaxed initialization (RI)

parser.add_argument('--alpha', default=0.1, type=float)                                                    # select the coefficient for client-momentum 
parser.add_argument('--beta', default=0.1, type=float)                                                     # select the coefficient for relaxed initialization 
parser.add_argument('--beta1', default=0.9, type=float)                                                    # select the coefficient for the first-order momentum
parser.add_argument('--beta2', default=0.99, type=float)      
parser.add_argument('--beta3', default=0.2, type=float)
parser.add_argument('--beta4', default=0.1, type=float)                                            # select the coefficient for the second-order momentum
parser.add_argument('--lamb', default=0.1, type=float)
parser.add_argument('--lamb1',default=0.9,type=float)                                                     # select the coefficient for the prox-term
parser.add_argument('--rho', default=0.1, type=float)                                                      # select the SAM perturbation rate
parser.add_argument('--gamma', default=1.0, type=float)
parser.add_argument('--delta',default=0.01,type=float)
parser.add_argument('--rho1',default=1.0,type=float)
parser.add_argument('--alpha1',default=0.1,type=float)                                                    # select the coefficient for the correction of SAM
parser.add_argument('--epsilon', default=0.01, type=float)                                                 # select the minimal value for avoiding zero-division
parser.add_argument('--rho2', default=2.0, type=float)                                                       # select the coefficient for the adaptive learning rate
parser.add_argument('--tau', default=2.0,type=float)
parser.add_argument('--num_cluster',default=3,type=int)                                                          # select the coefficient for the adaptive learning rate
parser.add_argument('--method', choices=['FedAvg', 'FedCM', 'FedDyn', 'SCAFFOLD', 'FedAdam', 'FedProx', 'FedSAM', 'MoFedSAM', \
                                         'FedSpeed','FedSMOO','FedTOGA','FedLESAM','FedGamma','FedLESAM_S','FedLESAM_D','FedLAP',\
                                            'FedPVU','FedVRA','FedPVU_SAM','A_FedPD','A_FedPD_SAM','FedADMMINSA',\
                                                'FedPDHyper','FedSGAM_KL','SparsyFed_adan_dp','FedNLCP','FedCADS','FedALS','FedPDHyper_SAM','FedPDHyper_SCA','FedGKD_PD','FedDisco'], type=str, default='FedAvg')
                                         
args = parser.parse_args()
# print(args)
torch.manual_seed(37)

if torch.cuda.is_available():
    device = torch.device(args.cuda)
else:
    device = torch.device("cpu")

if __name__=='__main__':
    ### Generate IID or Heterogeneous Dataset
    if not args.non_iid:
        data_obj = DatasetObject(dataset=args.dataset, n_client=args.total_client, seed=args.seed, unbalanced_sgm=0, rule='iid',
                                     data_path=args.data_file)
        print("Initialize the Dataset     --->  {:s} {:s} {:d} clients".format(args.dataset, 'IID', args.total_client))
    else:
        data_obj = DatasetObject(dataset=args.dataset, n_client=args.total_client, seed=args.seed, unbalanced_sgm=0, rule=args.split_rule,
                                     rule_arg=args.split_coef, data_path=args.data_file)
        print("Initialize the Dataset     --->  {:s} {:s}-{:s} {:d} clients".format(args.dataset, args.split_rule, str(args.split_coef), args.total_client))

    if args.dataset == 'mnist':
        classes = 10
    elif args.dataset == 'emnist':
        classes = 10
    elif args.dataset == 'SVHN':
        classes = 10
    elif args.dataset == 'rafdb':
        classes = 7
    elif args.dataset == 'fer2013':
        classes = 7
    elif args.dataset == 'fer2013plus':
        classes = 8
    elif args.dataset == 'celea':
        classes = 4 
    elif args.dataset == 'CIFAR10':
        classes = 10
    elif args.dataset == 'CIFAR100':
        classes = 100
    elif args.dataset == 'tinyimagenet':
        classes = 200
    elif args.dataset == 'AG_News':
        classes = 4
    else:
        raise NotImplementedError('not implemented dataset yet')
  

    ### Generate Model Function
    model_func = lambda: client_model(args.model,classes)
    print("Initialize the Model Func  --->  {:s} model".format(args.model))
    init_model = model_func()
    total_trainable_params = sum(p.numel() for p in init_model.parameters() if p.requires_grad)
    print("                           --->  {:d} parameters".format(total_trainable_params))
    init_par_list = get_mdl_params(init_model)
    
    ### Generate Server
    server_func = None
    if args.method == 'FedAvg':
        server_func = FedAvg
    elif args.method == 'FedCM':
        server_func = FedCM
    elif args.method == 'FedDyn':
        server_func = FedDyn
    elif args.method == 'SCAFFOLD':
        server_func = SCAFFOLD
    elif args.method == 'FedAdam':
        server_func = FedAdam
    elif args.method == 'FedProx':
        server_func = FedProx
    elif args.method == 'FedSAM':
        server_func = FedSAM
    elif args.method == 'MoFedSAM':
        server_func = MoFedSAM
    elif args.method == 'FedLESAM':
        server_func = FedLESAM
    elif args.method == 'FedLESAM_S':
        server_func = FedLESAM_S
    elif args.method == 'FedLESAM_D':
        server_func = FedLESAM_D
    elif args.method == 'FedALS':
        server_func = FedALS
    elif args.method == 'FedPVU':
        server_func = FedPVU
    elif args.method == 'FedVRA':
        server_func = FedVRA
    elif args.method == 'FedGamma':
        server_func = FedGamma
    elif args.method == 'FedSMOO':
        server_func = FedSMOO
    elif args.method == 'FedTOGA':
        server_func = FedTOGA
    elif args.method == 'FedSpeed':
        server_func = FedSpeed
    elif args.method == 'FedPVU_SAM':
        server_func = FedPVU_SAM
    elif args.method == 'A_FedPD':
        server_func = A_FedPD
    elif args.method == 'A_FedPD_SAM':
        server_func = A_FedPD_SAM
    elif args.method == 'FedPVU':
        server_func = FedPVU
    elif args.method == 'FedCADS':
        server_func = FedCADS
    elif args.method == 'FedGKD_PD':
        server_func = FedGKD_PD
    elif args.method == 'FedPDHyper_SAM':
        server_func = FedPDHyper_SAM
    elif args.method == 'FedPDHyper_SCA':
        server_func = FedPDHyper_SCA
    elif args.method == 'FedNLCP':
        server_func = FedNLCP
    elif args.method == 'FedDisco':
        server_func = FedDisco
    # elif args.method == 'FedADMMINSA':
    #     server_func = FedADMMINSA
    elif args.method == 'FedPDHyper':
        server_func = FedPDHyper
    elif args.method == 'FedSGAM_KL':
        server_func = FedSGAM_KL
    elif args.method == 'SparsyFed_adan_dp':
        server_func = SparsyFed_adan_dp
    else:
        raise NotImplementedError('not implemented method yet')
    
    _server = server_func(device=device, model_func=model_func, init_model=init_model, init_par_list=init_par_list,
                          datasets=data_obj, method=args.method, args=args)
    _server.train()
    