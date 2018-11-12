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
        V = args.embeding_num
        D = args.embeding_dim
        self.embed = args.embed
        self.embed.weight.requires_grad = not args.static
        
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        self.fc1 = nn.Sequential(
            nn.Linear(len(Ks) * Co * 3, len(Ks) * Co * 3),
            nn.ReLU(inplace=True),
            
            nn.Linear(len(Ks) * Co * 3, len(Ks) * Co),
            nn.ReLU(inplace=True),

            nn.Linear(len(Ks) * Co, len(Ks) * Co),
            nn.ReLU(inplace=True),

            nn.Linear(len(Ks) * Co, C)
        )
        
    def piece_pooling(self, x, e1_pos, e2_pos):
        e1_pos = e1_pos.tolist()
        e2_pos = e2_pos.tolist()
        
        t1_list = []
        t2_list = []
        t3_list = []
        
        for j, i in enumerate(x):
            e1_idx = e1_pos[j]
            e2_idx = e2_pos[j]
            t_x, max_pos = i.size()
            if e2_idx >= max_pos:
                e2_idx = e1_idx + 1
            
            t1 = i[:, 0:e1_idx]
            t1_size = t1.size()
            if len(t1_size) == 2:
                t_y = max_pos - t1.size(1)
                if t_y > 0:
                    t1 = F.pad(t1, (0, t_y))
                t1_list.append(t1)
            else:
                t1_list.append(torch.zeros(t_x, max_pos, device=device))

            t2 = i[:, e1_idx:e2_idx]
            t2_size = t2.size()
            if len(t2_size) == 2:
                t_y = max_pos - t2_size[1]
                if t_y > 0:
                    t2 = F.pad(t2, (0, t_y))
                t2_list.append(t2)
            else:
                t2_list.append(torch.zeros(t_x, max_pos, device=device))

            t3 = i[:, e2_idx:]
            t3_size = t3.size()
            if len(t3_size) == 2:
                t_y = max_pos - t3_size[1]
                if t_y > 0:
                    t3 = F.pad(t3, (0, t_y))
                t3_list.append(t3)
            else:
                t3_list.append(torch.zeros(t_x, max_pos, device=device))

        t1_tensor = torch.stack(t1_list)
        t2_tensor = torch.stack(t2_list)
        t3_tensor = torch.stack(t3_list)
        
        pool_1 = F.max_pool1d(t1_tensor, t1_tensor.size(2)).squeeze(2)
        pool_2 = F.max_pool1d(t2_tensor, t2_tensor.size(2)).squeeze(2)
        pool_3 = F.max_pool1d(t3_tensor, t3_tensor.size(2)).squeeze(2)
        
        return torch.cat([pool_1, pool_2, pool_3], 1)
        
    def forward_once(self, x, e1_pos, e2_pos):
        x = self.embed(x)  # (N, W, D)

        if self.embed.weight.requires_grad:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...] * len(Ks)
        x = [self.piece_pooling(i, e1_pos, e2_pos) for i in x]
        x = torch.cat(x, 1)
        output = self.fc1(x)
        
        return output

    def forward(self, input1, input2, position):
        output1 = self.forward_once(input1, position[0], position[1])
        output2 = self.forward_once(input2, position[2], position[3])
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
