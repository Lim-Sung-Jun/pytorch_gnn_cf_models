import numpy as np
import torch
import os
import math
from time import time

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def eval_model(indices_sort_top,index_end_i,top_k):
    # https://ride-or-die.info/normalized-discounted-cumulative-gain/
    # cg: 상위 k개의 추천결과의 관련성을 합한 값
    # relavance는 binary거나 세분화된 값을 가질 수 있다. 그런데 여기서는 0과 1로 구분하였다.
    # discounted를 추가해서 상위와 하위 즉 위치에 따라 패널티를 부여하였다.
    # relevance를 binary로 하기때문에 2^rel - 1 == rel
    hr_topK = 0
    ndcg_topK = 0

    ndcg_max = [0] * top_k
    temp_max_ndcg = 0
    for i_topK in range(top_k):
        temp_max_ndcg += 1.0 / math.log(i_topK + 2)
        ndcg_max[i_topK] = temp_max_ndcg
    # 2nd. binary니깐 결과를 다 더하면 무조건 제일 큰 값인 ideal dcg
    max_hr = top_k
    max_ndcg = ndcg_max[top_k - 1]
    if index_end_i < top_k:
        max_hr = (index_end_i) * 1.0
        # 예를 들어서 top_k가 20인데, testset에 그보다 적은 12개가 있으면 밸런스가 안맞아서 indcg를 다시 맞춰준다.
        max_ndcg = ndcg_max[index_end_i - 1]  # 2nd. 12개까지 합한 값을 ideal로 해준다.
    count = 0
    #
    for item_id in indices_sort_top:
        # 실제 최적화된 리스트랑 예측한 리스트랑 비교해야하는거아닌가? 왜 이렇게 하는 거지? 2nd. index가 index_end_i보다 아래여야지 test set과 같기때문이다. 그 이유는 비교는 rating으로 하고 상위 k개 추출은 index로 했기때문이다.
        if item_id < index_end_i:
            # 몇번통과하는지?
            # item_id는 말그대로 아이템 id로 6400처럼 다양하게 나올 수 있다.
            # index_end_i는 len(testing_set[user_id])로 test data의 갯수다.
            hr_topK += 1.0
            ndcg_topK += 1.0 / math.log(count + 2)
        count += 1
        if count == top_k:
            break

    hr_t = hr_topK / max_hr
    ndcg_t = ndcg_topK / max_ndcg
    # hr_t,ndcg_t,index_end_i,indices_sort_top
    # pdb.set_trace()
    return hr_t, ndcg_t

# def eval_model(pred_items, test_u_i_interaction, k):
#     ndcg_list = []
#     ndcg = 0
#
#     pred_items = pred_items * test_u_i_interaction
#     test_values, _ = torch.topk(test_u_i_interaction, dim=1, k=k)
#     pred_values, _ = torch.topk(pred_items, dim=1, k=k)
#     test_values = test_values.sum(dim = 1)
#     pred_values = pred_values.sum(dim = 1)
#
#     for ndcg_max, ndcg_topk  in zip(test_values, pred_values):
#         idcg = torch.from_numpy(1/np.log(np.arange(2, ndcg_max + 2))).float().to(device) # lr-gccf log10 사용 원래는 log2
#         idcg = idcg.sum()
#
#         dcg = torch.from_numpy(1/np.log(np.arange(2, ndcg_topk + 2))).float().to(device)
#         dcg = dcg.sum() # dcg 구하는 과정이 잘못됨! 왜냐하면 순서에 따라서 분모의 가중치가 달라지기때문이다.
#
#         ndcg_list.append(dcg/idcg)
#
#     TP = pred_items.sum(dim = 1)
#     recall = TP / test_u_i_interaction.sum(dim = 1)
#     recall[torch.isnan(recall)] = 1
#
#     recall = recall.mean()
#     ndcg = round(np.mean(ndcg_list), 4)
#     return recall, ndcg

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