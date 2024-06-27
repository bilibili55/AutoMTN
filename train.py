from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import configparser
import math
import random

from lib.utils import log_string, loadData, count_parameters, calc_acc, sampling_prob, calc_mrr
from model.model import SIGIR, LSTM, RNN, STGN, Flashback, STAN

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)
parser.add_argument('--cuda', type = int, default = config['train']['cuda'],
                    help = 'choose GPU')
parser.add_argument('--max_epoch', type = int, default = config['train']['max_epoch'],
                    help = 'epoch to run')
parser.add_argument('--batch_size', type = int, default = config['train']['batch_size'],
                    help = 'batch size')
parser.add_argument('--learning_rate', type=float, default = config['train']['learning_rate'],
                    help = 'initial learning rate')
parser.add_argument('--seed', type = int, default = config['train']['seed'],
                    help='random seed')

parser.add_argument('--userNum', type = int, default = config['data']['userNum'],
                    help = 'dims of each head attention outputs')
parser.add_argument('--poiNum', type = int, default = config['data']['poiNum'],
                    help = 'dims of each head attention outputs')
parser.add_argument('--catNum', type = int, default = config['data']['catNum'],
                    help = 'dims of each head attention outputs')
parser.add_argument('--length', type = int, default = config['data']['length'],
                    help = 'history steps')
# parser.add_argument('--Q', type = int, default = config['data']['Q'],
#                     help = 'prediction steps')
parser.add_argument('--train_ratio', type = float, default = config['data']['train_ratio'],
                    help = 'training set [default : 0.6]')
parser.add_argument('--val_ratio', type = float, default = config['data']['val_ratio'],
                    help = 'validation set [default : 0.2]')
parser.add_argument('--test_ratio', type = float, default = config['data']['test_ratio'],
                    help = 'testing set [default : 0.2]')

parser.add_argument('--features', type = int, default = config['param']['features'],
                    help = 'number of LSTM')
parser.add_argument('--layers', type = int, default = config['param']['layers'],
                    help = 'number of LSTM')
parser.add_argument('--heads', type = int, default = config['param']['heads'], # 08 128
                    help = 'dims of each head attention outputs')
parser.add_argument('--eta', type = float, default = config['param']['eta'], # 08 128
                    help = 'dims of each head attention outputs')
# parser.add_argument('--ed', type = int, default = config['param']['ed'], # 08 128
#                     help = 'dims of each head attention outputs')
# parser.add_argument('--regions', type = int, default = config['param']['regions'],
#                     help = 'dims of each head attention outputs')

parser.add_argument('--traj_file', default = config['file']['traj_file'],
                    help = 'traffic file')
parser.add_argument('--distance_file', default = config['file']['distance_file'],
                    help = 'traffic file')
# parser.add_argument('--category_distance_file', default = config['file']['category_distance_file'],
#                     help = 'traffic file')
parser.add_argument('--model_file', default = config['file']['model_file'],
                    help = 'save the model to disk')
parser.add_argument('--log_file', default = config['file']['log_file'],
                    help = 'log file')

args = parser.parse_args()

log = open(args.log_file, 'w')

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

def res(model, valX, valY):
    model.eval() # 评估模式, 这会关闭dropout
    # it = test_iter.get_iterator()
    num_val = valX.shape[0]
    pred = []
    label = []
    num_batch = math.ceil(num_val / args.batch_size)
    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(valX[start_idx : end_idx]).to(device)
                y = torch.from_numpy(valY[start_idx : end_idx]).long().to(device)

                prob, _ = model(X[...,0].long(), X[...,1].long(), X[...,2].long(), X[...,3], X[...,4], X[...,5].long(), X[...,6].long(), X[...,7].long())

                pred.append(prob)
                label.append(y)
    
    pred = torch.cat(pred, axis = 0)
    label = torch.cat(label, axis = 0)

    acc = calc_acc(pred, label)
    mrr = calc_mrr(pred, label)
    # log_string(log, 'acc 5: %.4f, acc 10: %.4f' % (acc[0] / valY.shape[0], acc[1] / valY.shape[0]))
    log_string(log, '%.4f,%.4f,%.4f,%.4f' % (acc[0] / valY.shape[0], acc[1] / valY.shape[0],mrr[0] / valY.shape[0], mrr[1] / valY.shape[0]))
    # log_string(log, '' % (mrr[0] / valY.shape[0], mrr[1] / valY.shape[0]))
    # log_string(log, 'mrr 5: %.4f, mrr 10: %.4f' % (mrr[0] / valY.shape[0], mrr[1] / valY.shape[0]))

    return acc

def train(model, trainX, trainY, traincY, valX, valY, valcY):
    num_train = trainX.shape[0]
    max_acc, global_seed = 0.0, 1
    # model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)
#     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 7, 8], # pems04 [7,8,9]
#                                                             gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,35],
                                                            gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,    
    #                                 verbose=False, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=2e-6, eps=1e-08)
    
    for epoch in tqdm(range(1,args.max_epoch+1)):
        model.train()
        train_l_sum, batch_count, start = 0.0, 0, time.time()
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        trainY = trainY[permutation]
        traincY = traincY[permutation]
        num_batch = math.ceil(num_train / args.batch_size)
        with tqdm(total=num_batch) as pbar:
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

                x = torch.from_numpy(trainX[start_idx : end_idx]).to(device)
                y = torch.from_numpy(trainY[start_idx : end_idx]).long().to(device)
                cy = torch.from_numpy(traincY[start_idx : end_idx]).long().to(device)
                
                optimizer.zero_grad()

                 # X[:,:,0] 用户id X[:,:,1] POI X[:,:,2] Cato X[:,:,3] lat X[:,:,4] log  X[:,:,5] tod  X[:,:,6] dow 时间    X[:,:,7] unixtime
                probpoi, probcat = model(x[...,0].long(), x[...,1].long(), x[...,2].long(), x[...,3], x[...,4], x[...,5].long(), x[...,6].long(), x[...,7].long())  #
#                 prob_sample, label_sample, global_seed = sampling_prob(probpoi, y, 20, global_seed, device)
#                 prob_samplecat, label_samplecat, global_seed = sampling_prob(probcat, cy, 20, global_seed)
                loss = F.cross_entropy(probpoi, y) + args.eta * F.cross_entropy(probcat, cy)
                    # print(loss)
                
                loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                
                train_l_sum += loss.cpu().item()
                batch_count += 1
                    
                pbar.update(1)
        lr = lr_scheduler.get_lr()
        log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
              % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        print('epoch %d, lr %.6f, loss %.4f, time %.1f sec'
              % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        acc = res(model, valX, valY)
        lr_scheduler.step()
        # lr_scheduler.step(mae[-1])
        if np.mean(acc) > max_acc:
            max_acc = np.mean(acc)
            torch.save(model.state_dict(), args.model_file)
        if epoch>=1:
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}:")
                print(f"Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
                log_string(log,f"Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
                print(f"Cached: {torch.cuda.memory_cached(i) / 1024 ** 2:.2f} MB")
                log_string(log,f"Cached: {torch.cuda.memory_cached(i) / 1024 ** 2:.2f} MB")

def test(model, valX, valY):
    model.load_state_dict(torch.load(args.model_file))
    acc = res(model, valX, valY)
    return

# def _compute_loss(y_true, y_predicted):
    # return masked_mae(y_predicted, y_true, 0.0)

if __name__ == '__main__':
    log_string(log, "data loading begin....")
    trainX, valX, testX, trainY, valY, testY, traincY, valcY, testcY, distance_matrix = loadData(args)
    log_string(log, "data loading end....")
    
    log_string(log, "model constructing begin....")
    model = SIGIR(args.heads, args.heads * args.features, args.layers, args.userNum, args.poiNum, args.catNum, distance_matrix, device).to(device)
#     model = STAN(args.features, args.layers, args.userNum, args.poiNum, args.catNum, distance_matrix, device).to(device)
#     model = Flashback(args.features, args.layers, args.userNum, args.poiNum, args.catNum, device).to(device)
#     model = STGN(args.features, args.layers, args.userNum, args.poiNum, args.catNum, device).to(device)
#     model = LSTM(args.features, args.layers, args.userNum, args.poiNum, args.catNum, device).to(device)
#     model = RNN(args.features, args.layers, args.userNum, args.poiNum, args.catNum, device).to(device)
    print(count_parameters(model))
    log_string(log, "model constructing end....")
    
    log_string(log, "training begin....")
    train(model, trainX, trainY, traincY, valX, valY, valcY)   # 28578*25*8  valX: 9
    log_string(log, "training end....")
    
    test(model, testX, testY)
