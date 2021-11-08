import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NGCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, node_dropout, norm_adj):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_layers = len(self.weight_size)
        self.norm_adj = norm_adj.to(device)
        # self.laplacian = laplacian
        self.node_dropout = node_dropout
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()

        self.weight_size = [self.embedding_dim] + self.weight_size
        for i in range(self.n_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weight_()

    def _init_weight_(self):
        # embedding이 normalize되어있는데 uniformazation할 필요있나?
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    # def _droupout_sparse(self, X):
    #     """
    #     Drop individual locations in X
    #
    #     Arguments:
    #     ---------
    #     X = adjacency matrix (PyTorch sparse tensor)
    #     dropout = fraction of nodes to drop
    #     noise_shape = number of non non-zero entries of X
    #     """
    #
    #     node_dropout_mask = ((self.node_dropout) + torch.rand(X._nnz())).floor().bool().to(device)
    #     i = X.coalesce().indices()
    #     v = X.coalesce()._values()
    #     i[:, node_dropout_mask] = 0
    #     v[node_dropout_mask] = 0
    #     X_dropout = torch.sparse.FloatTensor(i, v, X.shape).to(X.device)
    #
    #     return X_dropout.mul(1 / (1 - self.node_dropout))

    def forward(self):

        #node dropout ( option )
        # self.norm_adj_hat = self._droupout_sparse(self.norm_adj) if self.node_dropout > 0 else self.norm_adj
        # self.laplacian_hat = self._droupout_sparse(self.laplacian) if self.node_dropout > 0 else self.laplacian

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings] # E

        for i in range(self.n_layers):
            # weighted sum messages of neighbours
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings) # (L + I)E
            # side_L_embeddings = torch.sparse.mm(self.laplacian, ego_embeddings)  # LE

            # transformed sum weighted sum messages of neighbours
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings)) # (L + I)EW

            # bi messages of neighbours
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings) # LEE
            # transformed bi messages of neighbours
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings)) # LEEW

            # non-linear activation & message dropout
            ego_embeddings = sum_embeddings + bi_embeddings # (L + I)EW + LEEW
            ego_embeddings = self.dropout_list[i](ego_embeddings) # ACT( (L + I)EW + LEEW )
            # repo1에서는 이걸 message dropout으로 보는데 확인해보자!

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return u_g_embeddings, i_g_embeddings



class MF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self):
        u_g_embeddings = self.user_embedding.weight
        i_g_embeddings = self.item_embedding.weight

        return u_g_embeddings, i_g_embeddings