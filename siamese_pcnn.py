import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SiameseNetwork(nn.Module):
    def __init__(self, args):
        super(SiameseNetwork, self).__init__()
        
        C = args.output_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        D = args.embeding_dim
        self.embed = args.embed
        self.embed.weight.requires_grad = not args.static
        
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

#         self.fc1 = nn.Sequential(
#             nn.Linear(len(Ks) * Co * 3, len(Ks) * Co * 3),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(len(Ks) * Co * 3, len(Ks) * Co * 2),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(len(Ks) * Co * 2, len(Ks) * Co),
#             nn.ReLU(inplace=True),

#             nn.Linear(len(Ks) * Co, len(Ks) * Co),
#             nn.ReLU(inplace=True),

#             nn.Linear(len(Ks) * Co, C),
#             nn.ReLU(inplace=True),
#             nn.Dropout(args.dropout),
#         )
        self.fc1 = nn.Sequential(
            nn.Linear(len(Ks) * Co * 3, len(Ks) * Co * 3),
            nn.ReLU(inplace=True),
            
            nn.Linear(len(Ks) * Co * 3, len(Ks) * Co),
            nn.ReLU(inplace=True),

            nn.Linear(len(Ks) * Co, len(Ks) * Co),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),

            nn.Linear(len(Ks) * Co, C)
        )
        
    def piece_pooling(self, x, e1_size, e2_size):
        e1_idx = e1_size
        e2_idx = x.size(2) - e2_size
        
        t1_tensor = x[:, :, :e1_idx]
        t2_tensor = x[:, :, e1_idx:e2_idx]
        t3_tensor = x[:, :, e2_idx:]
        
        pool_1 = F.max_pool1d(t1_tensor, t1_tensor.size(2)).squeeze(2)
        pool_2 = F.max_pool1d(t2_tensor, t2_tensor.size(2)).squeeze(2)
        pool_3 = F.max_pool1d(t3_tensor, t3_tensor.size(2)).squeeze(2)
        
        return torch.cat([pool_1, pool_2, pool_3], 1)
        
    def forward_once(self, x_list):
        e1_size = x_list[0].size(1)
        e2_size = x_list[2].size(1)
        
        x = torch.cat(x_list, 1)
        x = self.embed(x)  # (N, W, D)

        if self.embed.weight.requires_grad:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...] * len(Ks)
        x = [self.piece_pooling(i, e1_size, e2_size) for i in x]
        x = torch.cat(x, 1)
        output = self.fc1(x)
        
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
