import argparse
import copy
import os
from load_data_nasdaq import load_EOD_data
from evaluator import evaluate
import numpy as np
from PGRL import KMeansPriorPenalty
import gc
import torch.nn as nn
import torch.nn.functional as F
from hgat_nasdaq import HGAT
from scipy import sparse
from torch_geometric import utils
import torch.optim as optim
import torch
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

try:
    from tensorflow.python.ops.nn_ops import leaky_relu
except ImportError:
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops

def leaky_relu(features, alpha=0.2, name=None):
    with ops.name_scope(name, "LeakyRelu", [features, alpha]):
        features = ops.convert_to_tensor(features, name="features")
        alpha = ops.convert_to_tensor(alpha, name="alpha")
        return math_ops.maximum(alpha * features, features)

seed = 3407
np.random.seed(seed)
tf.set_random_seed(seed)
device = 'cuda'
torch.backends.cudnn.benchmark = False

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)


def trr_loss_mse_rank(pred, base_price, ground_truth, mask, alpha, no_stocks):
    return_ratio = torch.div((pred - base_price), base_price)
    reg_loss = weighted_mse_loss(return_ratio, ground_truth, mask)
    all_ones = torch.ones(no_stocks, 1).to(device)
    pre_pw_dif = (torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1))
                  - torch.matmul(all_ones, torch.transpose(return_ratio, 0, 1)))
    gt_pw_dif = (
            torch.matmul(all_ones, torch.transpose(ground_truth, 0, 1)) -
            torch.matmul(ground_truth, torch.transpose(all_ones, 0, 1))
    )

    mask_pw = torch.matmul(mask, torch.transpose(mask, 0, 1))
    rank_loss = torch.mean(
        F.relu(
            ((pre_pw_dif * gt_pw_dif) * mask_pw)))
    # print("reg_loss",reg_loss)
    # print("rank_loss",rank_loss)
    loss = reg_loss + alpha * rank_loss
    del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
    return loss, reg_loss, rank_loss, return_ratio







class ReRaLSTM:
    def __init__(self, data_path, market_name, tickers_fname, relation_name,
                 emb_fname, parameters, steps=1, epochs=50, batch_size=None,l=4, flat=False, gpu=False, in_pro=False):

        seed = 123456789
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.l = l
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.relation_name = relation_name
        # load data
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)

        print('#tickers selected:', len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(data_path, market_name, self.tickers, steps)


        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        if batch_size is None:
            self.batch_size = len(self.tickers)  ##always,
        else:
            self.batch_size = batch_size

        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5
        self.gpu = gpu

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + seq_len, :], \
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(
                   self.price_data[:, offset + seq_len - 1], axis=1
               ), \
               np.expand_dims(
                   self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
               )

    def del_tensor_ele(self, arr, index):
        arr1 = arr[0:index]
        arr2 = arr[index + 1:]
        return torch.cat((arr1, arr2), dim=0)

    def clusters(self, num, X, step, n_iter, eta, loss_penalty):
        clustermodel = KMeansPriorPenalty(num)
        out1, out2 = KMeansPriorPenalty.fit(clustermodel, X=X, steps=step, n_iter=n_iter, eta=eta,
                                            loss_penalty=loss_penalty)
        out2 = out2.tolist()
        book = [[] for i in range(num)]
        for i in range(num):
            for x in range(len(out2)):
                if out2[x] == i:
                    book[i].append(x)
        index = copy.deepcopy(book)
        for i in range(len(book)):
            for j in range(len(book[i])):
                index[i][j] = i
        for i in range(1, len(index)):
            index[0].extend(index[i])
        col = np.array(index[0], dtype='int64')
        for i in range(1, len(book)):
            book[0].extend(book[i])
        row = np.array(book[0], dtype='int64')
        data = [1 for a in index[0]]
        data = np.array(data, dtype='int64')
        return col, row, data

    def clustersHyperedge(self, hedge_nums=400):
        col, row, data = self.clusters(hedge_nums, X=self.eod_data[:, :, 0], step=10, n_iter=10, eta=0.01, loss_penalty=0.1)
        for i in range(1,5):
            c,r,d = self.clusters(hedge_nums, X=self.eod_data[:,:,i], step=10, n_iter=10, eta=0.01, loss_penalty=0.1)
            c = c + hedge_nums*i
            col = np.hstack((col, c))
            row = np.hstack((row, r))
            data = np.hstack((data, d))
        inci_mat = data, (row, col)
        inci_sparse = sparse.coo_matrix((inci_mat), shape=(1026, hedge_nums * 5))
        incidence_edge = utils.from_scipy_sparse_matrix(inci_sparse)
        hyp_input = incidence_edge[0].to(device)
        return hyp_input

    def train(self):
        global df
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        model = HGAT(self.batch_size, self.l).to(device)


        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        optimizer_hgat = optim.Adam(model.parameters(),
                                    lr=self.parameters['lr'],
                                    weight_decay=5e-4)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_hgat, step_size=50, gamma=0.1)

        # hyp_input = self.clustersHyperedge()
        hyp_input = torch.tensor(np.load("hyp_input_global.npy", allow_pickle=True)).cuda()
        hyp_input_T = np.load("hyp_input_clusters_T.npy", allow_pickle=True)


        batch_offsets = np.arange(start=0, stop=self.valid_index-self.l, dtype=int)

        epochs_value = []
        btl_ = []
        maxtest_btl = 0
        for i in range(self.epochs):
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            model.train()
            print("epoch：",i)
            epochs_value.append(i)


            for j in tqdm(range(self.valid_index - self.parameters['seq'] - self.steps + 1)):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    batch_offsets[j])

                optimizer_hgat.zero_grad()
                output = model(torch.FloatTensor(emb_batch).to(device), hyp_input_T[batch_offsets[j]:batch_offsets[j]+self.l],hyp_input)  # , hyp_input

                cur_loss, cur_reg_loss, cur_rank_loss, curr_rr_train = trr_loss_mse_rank(output.reshape((1026, 1)),
                                                                                         torch.FloatTensor(
                                                                                             price_batch).to(device),
                                                                                         torch.FloatTensor(gt_batch).to(
                                                                                             device),
                                                                                         torch.FloatTensor(
                                                                                             mask_batch).to(device),
                                                                                         self.parameters['alpha'],
                                                                                         self.batch_size)
                tra_loss += cur_loss.item()
                tra_reg_loss += cur_reg_loss.item()
                tra_rank_loss += cur_rank_loss.item()
                cur_loss.backward()
                optimizer_hgat.step()


            print('Train Loss:',
                  tra_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_reg_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_rank_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1))

            scheduler.step()
            lr = scheduler.get_lr()
            print("lr:", lr)
            with torch.no_grad():
                # test on validation set
                cur_valid_pred = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                cur_valid_gt = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                cur_valid_mask = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                val_loss = 0.0
                val_reg_loss = 0.0
                val_rank_loss = 0.0
                model.eval()


                for cur_offset in range(
                        self.valid_index - self.parameters['seq'] - self.steps + 1,
                        self.test_index - self.parameters['seq'] - self.steps + 1
                ):
                    emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                        cur_offset)

                    output_val = model(torch.FloatTensor(emb_batch).to(device), hyp_input_T[batch_offsets[j]:batch_offsets[j]+self.l],hyp_input)  # , hyp_input
                    cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = trr_loss_mse_rank(output_val,
                                                                                      torch.FloatTensor(price_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(gt_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(mask_batch).to(
                                                                                          device),
                                                                                      self.parameters['alpha'],
                                                                                      self.batch_size)

                    cur_rr = cur_rr.detach().cpu().numpy().reshape((1026, 1))
                    val_loss += cur_loss.detach().cpu().item()
                    val_reg_loss += cur_reg_loss.detach().cpu().item()
                    val_rank_loss += cur_rank_loss.detach().cpu().item()
                    cur_valid_pred[:, cur_offset - (self.valid_index -
                                                    self.parameters['seq'] -
                                                    self.steps + 1)] = \
                        copy.copy(cur_rr[:, 0])
                    cur_valid_gt[:, cur_offset - (self.valid_index -
                                                  self.parameters['seq'] -
                                                  self.steps + 1)] = \
                        copy.copy(gt_batch[:, 0])
                    cur_valid_mask[:, cur_offset - (self.valid_index -
                                                    self.parameters['seq'] -
                                                    self.steps + 1)] = \
                        copy.copy(mask_batch[:, 0])
                print('Valid MSE:',
                      val_loss / (self.test_index - self.valid_index),
                      val_reg_loss / (self.test_index - self.valid_index),
                      val_rank_loss / (self.test_index - self.valid_index))
                cur_valid_perf,btl = evaluate(cur_valid_pred, cur_valid_gt,
                                          cur_valid_mask)
                print('\t Valid preformance:', cur_valid_perf)

                # test on testing set
                cur_test_pred = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )
                cur_test_gt = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )
                cur_test_mask = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )
                test_loss = 0.0
                test_reg_loss = 0.0
                test_rank_loss = 0.0
                model.eval()


                for cur_offset in range(self.test_index - self.parameters['seq'] - self.steps + 1,
                                             self.trade_dates - self.parameters['seq'] - self.steps + 1):
                    emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(cur_offset)

                    output_test = model(torch.FloatTensor(emb_batch).to(device), hyp_input_T[batch_offsets[j]:batch_offsets[j]+self.l],hyp_input)  # , hyp_input
                    cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = trr_loss_mse_rank(output_test,
                                                                                      torch.FloatTensor(price_batch).to(
                                                                                          device),
                                                                                       torch.FloatTensor(gt_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(mask_batch).to(
                                                                                          device),
                                                                                      self.parameters['alpha'],
                                                                                      self.batch_size)

                    cur_rr = cur_rr.detach().cpu().numpy().reshape((1026, 1))

                    test_loss += cur_loss.detach().cpu().item()
                    test_reg_loss += cur_reg_loss.detach().cpu().item()
                    test_rank_loss += cur_rank_loss.detach().cpu().item()

                    cur_test_pred[:, cur_offset - (self.test_index -
                                                   self.parameters['seq'] -
                                                   self.steps + 1)] = \
                        copy.copy(cur_rr[:, 0])
                    cur_test_gt[:, cur_offset - (self.test_index -
                                                 self.parameters['seq'] -
                                                 self.steps + 1)] = \
                        copy.copy(gt_batch[:, 0])
                    cur_test_mask[:, cur_offset - (self.test_index -
                                                   self.parameters['seq'] -
                                                   self.steps + 1)] = \
                        copy.copy(mask_batch[:, 0])
                print('Test MSE:',
                      test_loss / (self.trade_dates - self.test_index),
                      test_reg_loss / (self.trade_dates - self.test_index),
                      test_rank_loss / (self.trade_dates - self.test_index))
                cur_test_perf,btl = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
                print('\t Test performance:', cur_test_perf)

                btl_.append(btl)
                if cur_test_perf['btl'] > maxtest_btl:
                    maxtest_btl = cur_test_perf['btl']
                    maxtest_btlindex = i

            gc.collect()
            torch.cuda.empty_cache()


        print("best epoch:", maxtest_btlindex)
        print("best IRR1:", maxtest_btl)
        plt.rcParams['figure.figsize'] = (12.0, 8.0)
        plt.plot(btl_[maxtest_btlindex], label="Top1 IRR")
        plt.legend()
        plt.show()


    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True


if __name__ == '__main__':

    desc = 'train a relational rank lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='path of EOD data',
                        default='../data/2013-01-01')
    parser.add_argument('-m', help='market name', default='NASDAQ')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=4,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=64,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.0005,
                        help='learning rate')
    parser.add_argument('-a', default=0.1,
                        help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=1, help='use gpu')

    parser.add_argument('-e', '--emb_file', type=str,
                        default='NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy',
                        help='fname for pretrained sequential embedding')
    parser.add_argument('-rn', '--rel_name', type=str,
                        default='sector_industry',
                        help='relation type: sector_industry or wikidata')
    parser.add_argument('-ip', '--inner_prod', type=int, default=0)
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)

    args.inner_prod = (args.inner_prod == 1)

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a)}
    # 'alpha': float(alpha_arr[j])}

    RR_LSTM = ReRaLSTM(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        relation_name=args.rel_name,
        emb_fname=args.emb_file,
        parameters=parameters,
        steps=1, epochs=40, batch_size=None, l=args.l, gpu=args.gpu,
        in_pro=args.inner_prod
    )

    pred_all = RR_LSTM.train()






