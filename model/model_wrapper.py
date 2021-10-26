import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import torch.optim as optim
from time import time
import sys
import math
from datetime import datetime
import os

from torch import nn

from model.models import *
from utility.helper import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model_Wrapper(object):
    def __init__(self, data_config, args, data_generator):
        self.args = args
        self.data_generator = data_generator
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.norm_adj = data_config['norm_adj']
        self.laplacian = self.norm_adj - sp.eye(self.norm_adj.shape[0])

        #convert sparse matrix to tensor and then allocate on device
        self.norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).float()
        self.laplacian = self._convert_sp_mat_to_sp_tensor(self.laplacian).float()

        self.args.mess_dropout = eval(self.args.mess_dropout)
        self.args.layers_output_size = eval(self.args.layers_output_size)

        if args.model_type in ['ngcf']:
            self.model = NGCF(self.n_users, self.n_items, self.args.embed_dim, self.args.layers_output_size, self.args.mess_dropout, self.args.node_dropout, self.norm_adj, self.laplacian)
        elif args.model_type in ['mf']:
            self.model = MF(self.n_users, self.n_items, self.args.embed_dim)
        else:
            raise Exception('Dont know which model to train')

        self.model = self.model.to(device)
        # self.norm_adj = self.norm_adj.to(device) # 이중으로 device에 올려주
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.lr_scheduler = self.set_lr_scheduler()

        self.results = {"Epoch": [],
                   "Loss": [],
                   "Recall": [],
                   "NDCG": [],
                   "Training Time": []}

        print(self.model)

    def load_model(self):
        pass

    def save_model(self):
        pass

    def set_lr_scheduler(self):
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda= lambda epoch: 0.96 ** (epoch / 50))

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.mul(users, pos_items).sum(dim = 1)
        neg_scores = torch.mul(users, neg_items).sum(dim = 1)

        log_prob = F.logsigmoid(pos_scores - neg_scores).mean()
        bpr_loss = -log_prob

        if self.args.reg > 0.: #self.reg 추가하
            # u_emb는 self.batch_size
            l2norm = (torch.sum(users**2)/2. + torch.sum(pos_items**2)/2. + torch.sum(neg_items**2)/2.) / users.shape[0]
            l2reg  = self.args.reg * l2norm
            bpr_loss = -log_prob + l2reg

        return bpr_loss

    def train(self):
        # Set values for early stopping
        cur_best_loss, stopping_step, should_stop = 1e3, 0, False
        cur_best_metric = 0
        today = datetime.now()

        print("Start at " + str(today))
        print("Using " + str(device) + " for computations")

        for epoch in range(self.args.epochs):
            t1 = time()
            self.model.train()

            n_batch = self.data_generator.n_train // self.data_generator.batch_size + 1
            running_loss = 0
            for _ in range(n_batch):
                self.optimizer.zero_grad()
                # capture the collaborative signal from the graph
                self.ua_embeddings, self.ia_embeddings = self.model()

                users, pos_items, neg_items = self.data_generator.sample()
                u_g_embeddings = self.ua_embeddings[users]
                pos_i_g_embeddings = self.ia_embeddings[pos_items]
                neg_i_g_embeddings = self.ia_embeddings[neg_items]

                loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            training_time = time() - t1
            print("Epoch: {}, Training time: {:.2f}s, Loss: {:.4f}".
                  format(epoch, training_time, running_loss))

            self.lr_scheduler.step()

            if math.isnan(running_loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            # 여기부터 밑까지는 일단 주석처리해서 loss함수가 잘 작동하는지 비교해보자.
            if epoch % self.args.eval_N  == (self.args.eval_N - 1):
                self.model.eval()
                with torch.no_grad():
                    t2 = time()
                    # ua_embeddings, ia_embeddings에 nn.parameter을 해야하나?
                    # recall, ndcg = eval_model(self.model.u_g_embeddings.detach(),
                    #                           self.model.i_g_embeddings.detach(),
                    #                           self.data_generator.R_train,
                    #                           self.data_generator.R_test,
                    #                           self.args.k)
                    recall, ndcg = eval_model(nn.Parameter(self.ua_embeddings).detach(),
                                              nn.Parameter(self.ia_embeddings).detach(),
                                              self.data_generator.R_train,
                                              self.data_generator.R_test,
                                              self.args.k)
                print(
                    "Evaluate current model:\n",
                    "Epoch: {}, Validation time: {:.2f}s".format(epoch, time()-t2),"\n",
                    "Loss: {:.4f}:".format(running_loss), "\n",
                    "Recall@{}: {:.4f}".format(self.args.k, recall), "\n",
                    "NDCG@{}: {:.4f}".format(self.args.k, ndcg)
                    )

                del self.ua_embeddings, self.ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings

                cur_best_metric, stopping_step, should_stop = \
                early_stopping(recall, cur_best_metric, stopping_step, flag_step=5)

                # save results in dict
                self.results['Epoch'].append(epoch)
                self.results['Loss'].append(running_loss)
                self.results['Recall'].append(recall.item())
                self.results['NDCG'].append(ndcg.item())
                self.results['Training Time'].append(training_time)
            else:
                # save results in dict
                self.results['Epoch'].append(epoch)
                self.results['Loss'].append(running_loss)
                self.results['Recall'].append(None)
                self.results['NDCG'].append(None)
                self.results['Training Time'].append(training_time)

            print("results: ", self.results)
            if should_stop == True:
                break

        # save
        if self.args.save_results:
            date = today.strftime("%d%m%Y_%H%M")

            # save model as .pt file
            if os.path.isdir("./models"):
                torch.save(self.model.state_dict(), "./models/" + str(date) + "_" + self.args.model_type + "_" + self.args.dataset + ".pt")
            else:
                os.mkdir("./models")
                torch.save(self.model.state_dict(), "./models/" + str(date) + "_" + self.args.model_type + "_" + self.args.dataset + ".pt")
            ##
            results_path = '%s/%s/%s/%s_result.txt' % (self.args.results_dir, self.args.dataset, self.args.model_type, str(date))

            ensureDir(results_path)
            f = open(results_path, 'a')

            f.write(
                'datetime: %s\n\nembed_size=%d, lr=%.5f, mess_dropout=%s, regs=%s, adj_type=%s\n\n'
                % (datetime.now(), self.args.embed_dim, self.args.lr, self.args.mess_dropout, self.args.reg,
                   self.args.adj_type)) #, time_consume 시간도 추가하면 좋겠다.

            for i in range(len(self.results['Epoch'])):
                f.write('Epoch: %s loss:%s recall:%s ndcg:%s training time:%s\n'
                        % (self.results['Epoch'][i], self.results['Loss'][i], self.results['Recall'][i], self.results['NDCG'][i], self.results['Training Time'][i]))

            f.write('\n\n%s\n\n' % self.results)
            f.close()

# convert sparse matrix into sparse PyTorch tensor
    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        Convert scipy sparse matrix to PyTorch sparse matrix

        Arguments:
        ----------
        X = Adjacency matrix, scipy sparse matrix
        """
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        res = torch.sparse.FloatTensor(i, v, coo.shape).to(device)
        return res