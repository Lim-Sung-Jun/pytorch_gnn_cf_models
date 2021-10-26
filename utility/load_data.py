from time import time
import numpy as np
import random as rd
import scipy.sparse as sp

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        self.train_file = path + '/train.txt'
        self.test_file = path + '/test.txt'

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        self.load()

    # load train, test dataset from data_path
    def load(self):
        # search train_file for max user_id/item_id
        with open(self.train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    # first element is the user_id, rest are items
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    # item/user with highest number is number of items/users
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    # number of interactions
                    self.n_train += len(items)

        # search test_file for max item_id
        with open(self.test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    if not items:
                        print("empyt test exists")
                        pass
                    else:
                        self.n_items = max(self.n_items, max(items))
                        self.n_test += len(items)
        # adjust counters: user_id/item_id starts at 0
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        # create interactions/ratings matrix 'R' # dok = dictionary of keys
        print('Creating interaction matrices R_train and R_test...')
        t1 = time()
        self.R_train = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_test = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(self.train_file) as f_train:
            with open(self.test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]
                    # enter 1 if user interacted with item
                    for i in train_items:
                        self.R_train[uid, i] = 1.
                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    uid, test_items = items[0], items[1:]
                    for i in test_items:
                        self.R_test[uid, i] = 1.0
                    self.test_set[uid] = test_items
        print('Complete. Interaction matrices R_train and R_test created in', time() - t1, 'sec')
        print()

    # get adj_mat, norm_self_adj_mat, norm_adj_mat
    def get_adj_mat(self):
        try:
            start_T = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_self_adj_mat = sp.load_npz(self.path + '/s_norm_self_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            print(f"loaded adjacency matrix (shape: {adj_mat.shape}, time: {time() - start_T}")
            print()
        except Exception:
            print("no existing adj_matrix found")
            print()
            adj_mat, norm_self_adj_mat, norm_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_self_adj_mat.npz', norm_self_adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
        return adj_mat, norm_self_adj_mat, norm_adj_mat

    # create adj_mat, norm_self_adj_mat, norm_adj_mat
    def create_adj_mat(self):
        start_T = time()

        # create plain adjacency matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        A = A.tolil()

        A[:self.n_users, self.n_users:] = self.R_train.tolil()
        A[self.n_users:, :self.n_users] = self.R_train.tolil().T

        A = A.todok()
        print(f"create plain adjacency matrix (shape: {A.shape}, time: {time() - start_T}")

        start_T = time()
        # inner function for normalizing matrix
        def compute_norm_adj_matrix_single(adj):
            # 인접행렬의 행 값을 모두 더한다. (행 기준으로 합하면 degree 값을 구할 수 있다.)
            rowsum = np.array(adj.sum(1))

            # 행 값에 -1을 제곱하고 무한대인 경우 0으로 바꿔준다. (-(1/2)를 제곱해야하는데 양 옆이 아니고 한번에 하려고 -1을 제곱한다.)
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.

            # 위에서 구한 값을 대각원소로 diagonal matrix를 만들어준다.
            d_mat_inv = sp.diags(d_inv)

            # 인접행렬과 dot product를 해서 정규화한다.
            norm_adj = d_mat_inv.dot(adj)

            # 리턴하고 sparse matrix, coo로 변환
            return norm_adj.tocoo()

        norm_self_adj_mat = compute_norm_adj_matrix_single(A + sp.eye(A.shape[0]))
        norm_adj_mat = compute_norm_adj_matrix_single(A)
        print(f"create norm adjacency matrix (norm_self_shape: {norm_self_adj_mat.shape}, norm_shape: {norm_adj_mat.shape}, time: {time() - start_T}")
        print()
        plain_adj_mat = A

        return plain_adj_mat.tocsr(), norm_self_adj_mat.tocsr(), norm_adj_mat.tocsr()

# sample data for mini-batches
    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))
        print()