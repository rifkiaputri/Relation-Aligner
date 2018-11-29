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
            nn.Dropout(args.dropout),
            
            nn.Linear(len(Ks) * Co * 3, len(Ks) * Co),
            nn.Dropout(args.dropout),
            
            nn.Linear(len(Ks) * Co, len(Ks) * Co),
            nn.Dropout(args.dropout),
            
            nn.Linear(len(Ks) * Co, len(Ks) * Co),
            nn.Dropout(args.dropout),

            nn.Linear(len(Ks) * Co, C)
        )
        
    def piece_pooling(self, x, e1_size, e2_size):
        e1_idx = e1_size
        e2_idx = x.size(2) - e2_size
        if e1_idx == e2_idx:
            e2_idx += 1
        
        t1_tensor = x[:, :, :e1_idx]
        t2_tensor = x[:, :, e1_idx:e2_idx]
        t3_tensor = x[:, :, e2_idx:]
        
        pool_1 = F.max_pool1d(t1_tensor, t1_tensor.size(2)).squeeze(2)
        pool_2 = F.max_pool1d(t2_tensor, t2_tensor.size(2)).squeeze(2)
        pool_3 = F.max_pool1d(t3_tensor, t3_tensor.size(2)).squeeze(2)
        
        return torch.cat([pool_1, pool_2, pool_3], 1)
    
    def piece_pooling_att(self, x, q, e1_size, e2_size):
        e1_idx = e1_size
        e2_idx = x.size(2) - e2_size
        if e1_idx == e2_idx:
            e2_idx += 1
            
        t1_tensor = x[:, :, :e1_idx]
        t2_tensor = x[:, :, e1_idx:e2_idx]
        t3_tensor = x[:, :, e2_idx:]
        
        # Calculate attention for t2_tensor
        q_size = q.size(2)
        t_size = t2_tensor.size(2)
        
        if q_size == t_size:
            att_size = q_size
        elif q_size < t_size:
            att_size = t_size
            q = F.pad(q, (0, t_size - q_size), 'constant', 0)
        else:
            # q_size > t_size
            att_size = q_size
            t2_tensor = F.pad(t2_tensor, (0, q_size - t_size), 'constant', 0)
        
        attention = Attention(att_size)
        attention.to(device)
        t2_tensor, weights = attention(t2_tensor, q)
        
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
    
    def forward_once_att(self, x_list):
        e1_size = x_list[0].size(1)
        e2_size = x_list[2].size(1)
        
        x = torch.cat(x_list[:3], 1)
        x = self.embed(x)  # (N, W, D)
        q = self.embed(x_list[3])

        if self.embed.weight.requires_grad:
            x = Variable(x)
            q = Variable(c)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        q = q.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...] * len(Ks)
        q = [F.relu(conv(q)).squeeze(3) for conv in self.convs1]
        x = [self.piece_pooling_att(i, q[idx], e1_size, e2_size) for idx, i in enumerate(x)]
        x = torch.cat(x, 1)
        output = self.fc1(x)
        
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once_att(input2)
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
    

class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}
    Args:
        dim(int): The number of expected features in the output
    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
    Examples::
         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)
        
        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn
