import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, entity_vocab, relation_vocab, adj_mat):
        super(GCN, self).__init__()

        self.entity_emb = nn.Embedding(num_embeddings=len(entity_vocab),
                                    embedding_dim=nfeat,
                                    padding_idx=len(entity_vocab))
        uniform_range = 6 / np.sqrt(nfeat)
        self.entity_emb.weight.data.uniform_(-uniform_range, uniform_range)

        self.relation_emb = nn.Embedding(num_embeddings=len(relation_vocab),
                                    embedding_dim=nfeat,
                                    padding_idx=len(relation_vocab))
        uniform_range = 6 / np.sqrt(nfeat)
        self.relation_emb.weight.data.uniform_(-uniform_range, uniform_range)

        self.adj_mat = adj_mat

        self.gc1 = GraphConvolution(nfeat, nfeat)
        self.gc2 = GraphConvolution(nfeat, nfeat)
        
        self.criterion = nn.MarginRankingLoss(margin=1.0)

    def distance(self, triplets):
        assert triplets.size()[1] == 3
        heads = int(triplets[:, 0])
        relations = triplets[:, 1]
        tails = int(triplets[:, 2])
        return (self.entity_emb(heads) + self.relation_emb(relations) - self.entity_emb(tails)).norm(p=self.norm, dim=1)

    def TransE_forward(self, pos_triplet, neg_triplet):
        # -1 to avoid nan for OOV vector
        self.entity_emb.weight.data[:-1, :].div_(self.entity_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))

        pos_distance = self.distance(pos_triplet)
        neg_distance = self.distance(neg_triplet)

        target = torch.tensor([-1], dtype=torch.long)

        return self.criterion(pos_distance, neg_distance, target)

    def forward(self, x):
        x = F.relu(self.gc1(self.entity_emb, self.adj_mat))
        x = self.gc2(x, self.adj_mat)
        return F.log_softmax(x, dim=1)
# policy net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(20, 128) 
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(128, len(candidates_embeddings)?)
    
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x
