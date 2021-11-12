import numpy as np
import torch
import os
import math
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_model(pred_items, test_u_i_interaction, k):
    ndcg_list = []
    ndcg = 0

    pred_items = pred_items * test_u_i_interaction
    test_values, _ = torch.topk(test_u_i_interaction, dim=1, k=k)
    pred_values, _ = torch.topk(pred_items, dim=1, k=k)
    test_values = test_values.sum(dim = 1)
    pred_values = pred_values.sum(dim = 1)

    for ndcg_max, ndcg_topk  in zip(test_values, pred_values):
        idcg = torch.from_numpy(1/np.log(np.arange(2, ndcg_max + 2))).float().to(device) # lr-gccf log10 사용 원래는 log2
        idcg = idcg.sum()

        dcg = torch.from_numpy(1/np.log(np.arange(2, ndcg_topk + 2))).float().to(device)
        dcg = dcg.sum() # dcg 구하는 과정이 잘못됨! 왜냐하면 순서에 따라서 분모의 가중치가 달라지기때문이다.

        ndcg_list.append(dcg/idcg)

    TP = pred_items.sum(dim = 1)
    recall = TP / test_u_i_interaction.sum(dim = 1)
    recall[torch.isnan(recall)] = 1

    recall = recall.mean()
    ndcg = round(np.mean(ndcg_list), 4)
    return recall, ndcg

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