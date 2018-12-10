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
            nn.Linear(len(Ks) * Co * 3, len(Ks) * Co),
            nn.ReLU(inplace=True),
            
            nn.Linear(len(Ks) * Co, len(Ks) * Co),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            
            nn.Linear(len(Ks) * Co, C)
        )
    
    def convolute(self, x_list):
        x = torch.cat(x_list, 1)
        x = self.embed(x)  # (N, W, D)

        if self.embed.weight.requires_grad:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...] * len(Ks)
        
        e1_size = x_list[0].size(1)
        e2_size = x_list[2].size(1)
        
        return x, e1_size, e2_size

    def forward(self, input1, input2):
        conv_kb, e1_size_kb, e2_size_kb = self.convolute(input1)
        conv_oie, e1_size_oie, e2_size_oie = self.convolute(input2)
        
        x_kb = []
        x_oie = []
        
        for i in range(len(conv_kb)):
            xi_kb = conv_kb[i]
            xi_oie = conv_oie[i]
            
            e1_idx_kb = e1_size_kb
            e2_idx_kb = xi_kb.size(2) - e2_size_kb
            if e1_idx_kb == e2_idx_kb:
                e2_idx_kb += 1
                
            e1_idx_oie = e1_size_oie
            e2_idx_oie = xi_oie.size(2) - e2_size_oie
            if e1_idx_oie == e2_idx_oie:
                e2_idx_oie += 1
            
            t1_tensor_kb = xi_kb[:, :, :e1_idx_kb]
            t1_tensor_oie = xi_oie[:, :, :e1_idx_oie]
            
            t3_tensor_kb = xi_kb[:, :, e2_idx_kb:]
            t3_tensor_oie = xi_oie[:, :, e2_idx_oie:]
            
            pool_1_kb = F.max_pool1d(t1_tensor_kb, t1_tensor_kb.size(2)).squeeze(2)
            pool_1_oie = F.max_pool1d(t1_tensor_oie, t1_tensor_oie.size(2)).squeeze(2)
            
            pool_3_kb = F.max_pool1d(t3_tensor_kb, t3_tensor_kb.size(2)).squeeze(2)
            pool_3_oie = F.max_pool1d(t3_tensor_oie, t3_tensor_oie.size(2)).squeeze(2)
            
            # Cross alignment for kb and oie rel
            # Make sure matrices have the same size
            rel_kb = xi_kb[:, :, e1_idx_kb:e2_idx_kb]
            rel_oie = xi_oie[:, :, e1_idx_oie:e2_idx_oie]
            n, d = rel_kb.size(0), rel_kb.size(1)
            kb_len = rel_kb.size(2)
            oie_len = rel_oie.size(2)
            if kb_len > oie_len:
                rel_oie = F.pad(rel_oie, (0, kb_len - oie_len), 'constant', 0)
            elif oie_len > kb_len:
                rel_kb = F.pad(rel_kb, (0, oie_len - kb_len), 'constant', 0)
            
            z = rel_kb.size(2)
            align = F.relu(torch.bmm(rel_kb.view(n * d, z, 1), rel_oie.view(n * d, 1, z)))
            align = align.view(n, d, z, z)
            align_t = align.transpose(3, 2)

            align_kb = F.max_pool2d(align_t, (z, 1)).squeeze(2)
            align_oie = F.max_pool2d(align, (z, 1)).squeeze(2)
            
            pool_2_kb = F.softmax(F.max_pool1d(align_kb, z).squeeze(2), 1)
            pool_2_oie = F.softmax(F.max_pool1d(align_oie, z).squeeze(2), 1)
            
            x_kb.append(torch.cat([pool_1_kb, pool_2_kb, pool_3_kb], 1))
            x_oie.append(torch.cat([pool_1_oie, pool_2_oie, pool_3_oie], 1))
            
            # free memory
            del t1_tensor_kb
            del t1_tensor_oie
            del align
            del align_kb
            del align_oie
            del t3_tensor_kb
            del t3_tensor_oie
            del pool_1_kb
            del pool_1_oie
            del pool_2_kb
            del pool_2_oie
            del pool_3_kb
            del pool_3_oie
            del xi_kb
            del xi_oie
            del e1_idx_kb
            del e1_idx_oie
            del e2_idx_kb
            del e2_idx_oie
            del z
            del n
            del d
            del kb_len
            del oie_len
        
        x_kb = torch.cat(x_kb, 1)
        x_oie = torch.cat(x_oie, 1)
        
        return self.fc1(x_kb), self.fc1(x_oie)

    
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
