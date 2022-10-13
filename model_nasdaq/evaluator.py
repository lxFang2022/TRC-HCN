import math
import numpy as np
import scipy.stats as sps
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import ndcg_score
from empyrical.stats import max_drawdown, downside_risk, calmar_ratio


def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    # print('gt_rt',np.max(ground_truth))
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask) ** 2 \
                         / np.sum(mask)
    mrr_top = 0.0
    mrr_top5 = 0.0
    mrr_top10 = 0.0
    mrr_top20 = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0
    sharpe_li5 = []
    sharpe_li1 = []
    ndcg = []
    btl = []

    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top5 = set()
        gt_top10 = set()
        gt_top20 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)
            if len(gt_top20) < 20:
                gt_top20.add(cur_rank)

        rank_pre = np.argsort(prediction[:, i])

        pre_top1 = set()
        pre_top5 = set()
        pre_top10 = set()
        pre_top20 = set()
        gt = []
        pt = []

        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)
            if len(pre_top20) < 20:
                pre_top20.add(cur_rank)


        # performance['ndcg_score_top5'] = ndcg_score(np.array(list(gt_top5)).reshape(1, -1),
        #                                             np.array(list(pre_top5)).reshape(1, -1))
        # calculate mrr of top1
        top1_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt
        # calculate mrr of top5
        top5_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top5_pos_in_gt += 1
                if cur_rank in pre_top5:
                    break
        if top5_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top5 += 1.0 / top5_pos_in_gt
        # calculate mrr of top10
        top10_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top10_pos_in_gt += 1
                if cur_rank in pre_top10:
                    break
        if top10_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top10 += 1.0 / top10_pos_in_gt
        # calculate mrr of top20
        top20_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top20_pos_in_gt += 1
                if cur_rank in pre_top20:
                    break
        if top20_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top20 += 1.0 / top20_pos_in_gt
        # back testing on top 1

        real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
        sharpe_li1.append(real_ret_rat_top)


        bt_long += real_ret_rat_top
        btl.append(bt_long - 1)


        # back testing on top 5
        real_ret_rat_top5 = 0
        bt_long5 += real_ret_rat_top5


        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= 5
        bt_long5 += real_ret_rat_top5
        sharpe_li5.append(real_ret_rat_top5)

        real_ret_rat_top5_gt = 0
        for pre in gt_top5:
            real_ret_rat_top5_gt += ground_truth[pre][i]
        real_ret_rat_top5_gt /= 5
        #
        # for t in range(5):
        #     gt.append(ground_truth[gt_top5.pop(), i])
        #
        # for t in range(5):
        #     pt.append(ground_truth[pre_top5.pop(), i])
        # ndcg.append(ndcg_score(np.array(gt).reshape(1, -1),
        #                        np.array(pt).reshape(1, -1)))
    performance['btl'] = bt_long - 1
    performance['btl5'] = bt_long5 - 1
    performance['mrrt'] = mrr_top / (prediction.shape[1] - all_miss_days_top)
    performance['mrrt5'] = mrr_top5 / (prediction.shape[1] - all_miss_days_top)/5
    performance['mrrt10'] = mrr_top10 / (prediction.shape[1] - all_miss_days_top)/10
    performance['mrrt20'] = mrr_top20 / (prediction.shape[1] - all_miss_days_top)/20
    # performance['ndcg_score_top5'] = np.mean(np.array(ndcg))

    sharpe_li5 = np.array(sharpe_li5)

    sharpe_li1 = np.array(sharpe_li1)

    performance['sharpe1'] = (np.mean(sharpe_li1) / np.std(sharpe_li1)) * 15.87
    performance['sharpe5'] = (np.mean(sharpe_li5) / np.std(sharpe_li5)) * 15.87  # To annualize
    return performance,btl
