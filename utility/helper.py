import numpy as np
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_matrix(X, n_splits=100):
    """
    Split a matrix/Tensor into n_folds (for the user embeddings and the R matrices)

    Arguments:
    ---------
    X: matrix to be split
    n_folds: number of folds

    Returns:
    -------
    splits: split matrices
    """
    splits = []
    chunk_size = X.shape[0] // n_splits
    for i in range(n_splits):
        start = i * chunk_size
        end = X.shape[0] if i == n_splits - 1 else (i + 1) * chunk_size
        splits.append(X[start:end])
    return splits


def compute_ndcg_k(pred_items, test_items, test_indices, k):
    """
    Compute NDCG@k

    Arguments:
    ---------
    pred_items: binary tensor with 1s in those locations corresponding to the predicted item interactions
    test_items: binary tensor with 1s in locations corresponding to the real test interactions
    test_indices: tensor with the location of the top-k predicted items
    k: k'th-order

    Returns:
    -------
    NDCG@k
    """
    r = (test_items * pred_items).gather(1, test_indices) # 상위 인덱스를 가져온다. 얼마나 맞췄는지 알 수 있다.
    f = torch.from_numpy(np.log2(np.arange(2, k + 2))).float().to(device)  # cuda()
    dcg = (r[:, :k] / f).sum(1) # 곱해서 정답이면 다 더해야하는데 k열까지만 더하네? test_indices만 가져오기때뭉네 어차피 20개다.
    dcg_max = (torch.sort(r, dim=1, descending=True)[0][:, :k] / f).sum(1) # dcg_max면 당연히 전부 1아닌가?
    ndcg = dcg / dcg_max
    ndcg[torch.isnan(ndcg)] = 0
    return ndcg


def eval_model(u_emb, i_emb, Rtr, Rte, k):
    """
    Evaluate the model

    Arguments:
    ---------
    u_emb: User embeddings
    i_emb: Item embeddings
    Rtr: Sparse matrix with the training interactions
    Rte: Sparse matrix with the testing interactions
    k : kth-order for metrics

    Returns:
    --------
    result: Dictionary with lists correponding to the metrics at order k for k in Ks
    """
    # split matrices
    ue_splits = split_matrix(u_emb)
    tr_splits = split_matrix(Rtr)
    te_splits = split_matrix(Rte)

    recall_k, ndcg_k = [], []
    # compute results for split matrices
    for ue_f, tr_f, te_f in zip(ue_splits, tr_splits, te_splits):
        scores = torch.mm(ue_f, i_emb.t())

        test_items = torch.from_numpy(te_f.todense()).float().to(device)  # cuda()
        non_train_items = torch.from_numpy(1 - (tr_f.todense())).float().to(device)
        scores = scores * non_train_items

        _, test_indices = torch.topk(scores, dim=1, k=k)
        pred_items = torch.zeros_like(scores).float()
        pred_items.scatter_(dim=1, index=test_indices, src=torch.ones_like(test_indices).float().to(device))
        #pred_items.scatter_(dim=1, index=test_indices, src=torch.tensor(1.0).to(device))
        #shape을 맞춰줘야한다!

        topk_preds = torch.zeros_like(scores).float()
        topk_preds.scatter_(dim=1, index=test_indices[:, :k], src=torch.ones_like(test_indices[:, :k]).float().to(device))
        # topk_preds.scatter_(dim=1, index=test_indices[:, :k], src=torch.tensor(1.0))

        TP = (test_items * topk_preds).sum(1)
        rec = TP / test_items.sum(1)
        ndcg = compute_ndcg_k(pred_items, test_items, test_indices, k)

        recall_k.append(rec)
        ndcg_k.append(ndcg)

    return torch.cat(recall_k).mean(), torch.cat(ndcg_k).mean()

def early_stopping(log_value, best_value, stopping_step, flag_step, expected_order='asc'):
    """
    Check if early_stopping is needed
    Function copied from original code
    """
    assert expected_order in ['asc', 'des']
    if (expected_order == 'asc' and log_value >= best_value) or (expected_order == 'des' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False

    return best_value, stopping_step, should_stop

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)