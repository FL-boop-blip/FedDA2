import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils_models import *

class client_model(nn.Module):
    def __init__(self, name, n_cls,alpha = 1.0, sparsity = 0.0, pruning_type = "unstructured",  args=True):
        super(client_model, self).__init__()
        self.name = name
        self.alpha = alpha
        self.n_cls = n_cls
        self.sparsity = sparsity
        self.pruning_type = pruning_type
        if self.name == 'ResNet18P':
            resnet18 = models.resnet18(num_classes=2048, pretrained=False)
            param_dicts = torch.load('/mnt/workspace/colla_group/ckpt/resnet18-5c106cde.pth')

            for item in list(param_dicts):
               if "fc" in item:
                   del param_dicts[item]
            resnet18.load_state_dict(param_dicts, strict=False)

            resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            self.model=resnet18
            #if args.dataset=="officehome":
            n_class=65
            self.fc=nn.Linear(2048,n_class)
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)
          
        if self.name == 'mnist_2NN':
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, self.n_cls)
            
        if self.name == 'emnist_NN':
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, self.n_cls)
        
        if self.name == 'LeNet':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.name == 'LeNet_fusion':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(64+64, 192, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        
        if self.name == "ResNet18_sparsy":
            resnet18 = models.resnet18(pretrained=False)
            resnet18.fc = nn.Linear(512, self.n_cls)
            # resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False)
            # nn.init.kaiming_normal_(resnet18.conv1.weight, mode='fan_out', nonlinearity='relu')
            # resnet18.maxpool = nn.Identity()  # Remove the maxpool layer
            resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            replace_layer_with_sparsyfed(
                module=resnet18,
                name="ResNet18",
                alpha=self.alpha,
                sparsity=self.sparsity,
                pruning_type=self.pruning_type,
            )
            def init_model(
                module: nn.Module,
            ) -> None:
                """Initialize the weights of the layers."""
                init_weights(module)
                for _, immediate_child_module in module.named_children():
                    init_model(immediate_child_module)

            init_model(resnet18)
            self.model = resnet18

        if self.name == 'AG_News_NN':
            self.embedding = nn.Embedding(30626, 100, padding_idx=0)
        
        # 分类器
            self.classifier = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, self.n_cls)
            )

        if self.name == 'ResNet18_fusion':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 10)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'

            self.conv1 = resnet18.conv1
            self.bn1 = resnet18.bn1
            self.relu = resnet18.relu
            self.maxpool = resnet18.maxpool
            self.layer1 = resnet18.layer1
            self.layer2 = resnet18.layer2
            self.layer3 = resnet18.layer3
            self.layer4 = resnet18.layer4
            self.avgpool = resnet18.avgpool
            self.fc = resnet18.fc

            self.fuse_conv1 = nn.Sequential(
                nn.Conv2d(64+128, 512, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            self.fuse_conv2 = nn.Sequential(
                nn.Conv2d(128+256, 512, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            self.fuse_conv3 = nn.Sequential(
                nn.Conv2d(256+512,512,kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )

            
        if self.name == 'ResNet18':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, self.n_cls)

            # Change BN to GN 
            resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
            
            self.model = resnet18
        if self.name == 'ResNet18_100':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, self.n_cls)

            # Change BN to GN 
            resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
            
            self.model = resnet18
            
    def forward(self, x,lengths = None, is_feat = False):

        if self.name == 'Linear':
            x = self.fc(x)
        if self.name == "ResNet18_sparsy":
            x = self.model(x)
            
        if self.name == 'mnist_2NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
  
        if self.name == 'emnist_NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        if self.name == 'LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'LeNet_fusion':
            x1 = self.pool(F.relu(self.conv1(x)))
            x2 = self.pool(F.relu(self.conv2(x1)))
            x1_resized = F.interpolate(x1, size= x2.shape[2:], mode= 'bilinear')
            fuse_conv = torch.cat([x1_resized, x2], dim=1)
            out1 = self.fuse_conv(fuse_conv)
            x2 = x2.view(-1, 64*5*5)
            x2 = F.relu(self.fc1(x2))
            x2 = F.relu(self.fc2(x2))
            x = self.fc3(x2)
            out1 = self.fc3(out1)
            if is_feat:
                return [out1], x
            else:
                return x
        
        if self.name == 'AG_News_NN':
            embedded = self.embedding(x)
            mask = (x != 0).float()  # [batch_size, seq_len]
            mask = mask.unsqueeze(2)  # [batch_size, seq_len, 1]
            masked_embedded = embedded * mask
            sum_embeddings = torch.sum(masked_embedded, dim=1)
            avg_embeddings = sum_embeddings / lengths.view(-1, 1).float()
            logits = self.classifier(avg_embeddings)
            return logits
            
        if self.name in ['ResNet18',"ResNet18_100"]:
            x = self.model(x)
            #print(x)
        if self.name in ["ResNet18P"]:
            x=self.model(x)
            x=self.fc(x)
            return x
            
        if self.name == 'ResNet18_fusion':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            f1 = x
            x = self.layer2(x)
            f2 = x
            x = self.layer3(x)
            f3 = x
            x = self.layer4(x)
            f4 = x
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            f1_resized = F.interpolate(f1,size= f2.shape[2:],mode='bilinear',align_corners=False)
            fuse12 = torch.cat([f1_resized,f2],dim=1)
            out1 = self.fuse_conv1(fuse12)
            out1 = self.fc(out1)
            f2_resized = F.interpolate(f2, size= f3.shape[2:],mode = 'bilinear',align_corners=False)
            fuse23 = torch.cat([f2_resized,f3], dim=1)
            out2 = self.fuse_conv2(fuse23)
            out2 = self.fc(out2)
            f3_resized = F.interpolate(f3,size= f4.shape[2:], mode = 'bilinear', align_corners=False)
            fuse34 = torch.cat([f3_resized, f4], dim=1)
            out3 = self.fuse_conv3(fuse34)
            out3 = self.fc(out3)
            if is_feat:
                return [out1, out2, out3], x
            else:
                return x
        return x
    

def count_parameters(model, dtype=torch.float32):
    """计算模型的参数量，并返回 MB 大小"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算存储大小（MB）
    if dtype == torch.float32:
        bytes_per_param = 4  # float32 占 4 字节
    elif dtype == torch.float16:
        bytes_per_param = 2  # float16 占 2 字节
    else:
        raise ValueError("Unsupported dtype. Use torch.float32 or torch.float16.")
    
    total_size_mb = (total_params * bytes_per_param) / (1024 ** 2)  # 转换为 MB
    return total_size_mb

if __name__ == "__main__":
    model = client_model(name='ResNet18', n_cls=10)
    print(model)
    print("Number of parameters:", count_parameters(model))
    