from nas_201_api import NASBench201API as API201
from tifffile import FileHandle
import tqdm
from tqdm import tqdm
import models
from models import NB101Network
from utils import get_datasets
from nats_bench import create as create_nats
from scipy import stats
from thop import profile
import os
import argparse
import torch 
import torch.nn as nn
import random 
import numpy as np
from nasbench import api as api101
from ptflops import get_model_complexity_info
from metrics import *
from models.ImageNet_ResNet import ResNet
import pandas as pd
from models.Masternet import MasterNet
import time

parser = argparse.ArgumentParser(description='ZS-NAS')
parser.add_argument('--searchspace', metavar='ss', type=str, choices=['101','201','nats','nats_tss', 'mbnv2', 'resnet', 'hybrid'],
                    help='define the target search space of benchmark')
parser.add_argument('--dataset', metavar='ds', type=str, choices=['cifar10','cifar100','ImageNet16-120','imagenet-1k', 'cifar10-valid'],
                    help='select the dataset')
parser.add_argument('--data_path', type=str, default='~/dataset/',
                    help='the path where you store the dataset')
parser.add_argument('--cutout', type=int, default=0,
                    help='use cutout or not on input data')
parser.add_argument('--batchsize', type=int, default=64,
                    help='batch size for each input batch')
parser.add_argument('--num_worker', type=int, default=8,
                    help='number of threads for data pipelining')
parser.add_argument('--metric', type=str, choices=['basic','ntk','lr', 'logdet', 'grad', 'zen','IB', 'zico', 'intgrad_mean', 'intgrad_pri', 'intgrad_param', 'mix'],
                    help='define the zero-shot proxy for evaluation')
parser.add_argument('--startnetid', type=int, default=0,
                    help='the index of the first network to be evaluated in the search space. currently only works for nb101')
parser.add_argument('--manualSeed', type=int, default=0,
                    help='random seed')
parser.add_argument('--maxbatch', type=int, default=2,
                    help='maxbatch for zico trainloader')
parser.add_argument('--metrics-root', default='cached_metrics')
parser.add_argument('--cache-metric', action='store_true')
parser.add_argument('--resume-metric', action='store_true')
parser.add_argument('--metricfile', choices=['intgrad_mean', 'intgrad_pri', 'intgrad_param'])
args = parser.parse_args()


def getmisc(dataset, batchsize, manualSeed=0, num_worker=8):
    manualSeed=manualSeed
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if dataset == "cifar10":
        root = "/home/fuwei/data/pytorch_cifar10/"
        imgsize=32
    elif dataset == "cifar100":
        root = "/home/fuwei/data/pytorch_cifar100/"
        imgsize=32
    elif dataset.startswith("imagenet-1k"):
        root = "/home/fuwei/dataset/img1k/ImageNet1k/"
        imgsize=224
    elif dataset.startswith("ImageNet16"):
        root = "/home/fuwei/dataset/ImageNet16"
        imgsize=16
    
    train_data, test_data, xshape, class_num = get_datasets(dataset, root, 0)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=batchsize, shuffle=True, num_workers=num_worker)
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=batchsize, shuffle=False, num_workers=num_worker)

    ce_loss = nn.CrossEntropyLoss().cuda()
    # filename = 'misc/'+'{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(args.metric, args.searchspace, args.dataset,args.batchsize, \
    #                                 args.cutout, args.gamma1, args.gamma2, args.maxbatch)
    return imgsize, ce_loss, trainloader, testloader


def search201(api, netid, dataset):
    if dataset=='cifar10':
        dsprestr='ori'
    else: 
        dsprestr='x'
    results = api.query_by_index(netid, dataset, hp= '200') 
    train_loss, train_acc, test_loss, test_acc =0, 0, 0, 0
    for seed, result in results.items():
        train_loss += result.get_train()['loss']
        train_acc += result.get_train()['accuracy']
        test_loss += result.get_eval(dsprestr+'-test')['loss']
        test_acc += result.get_eval(dsprestr+'-test')['accuracy']
    config = api.get_net_config(netid, dataset)
    network = models.get_cell_based_tiny_net(config) 
    num_trails = len(results)
    train_loss, train_acc, test_loss, test_acc = \
            train_loss/num_trails, train_acc/num_trails, test_loss/num_trails, test_acc/num_trails
    return network, [train_acc, train_acc, test_loss, test_acc]


def search_nats(api, netid, dataset, hpval):
    # Simulate the training of the 1224-th candidate:
    # validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(netid, dataset=dataset, hp=hpval)
    testacc = api.get_more_info(netid, dataset, hp=hpval, is_random=False)['test-accuracy']
    config = api.get_net_config(netid, dataset)
    network = models.get_cell_based_tiny_net(config)
    return network, testacc


def get101acc(data_dict:dict):
    # acc4=(data_dict[4][0]['final_test_accuracy']+data_dict[4][1]['final_test_accuracy']+data_dict[4][2]['final_test_accuracy'])/3.0
    # acc12=(data_dict[12][0]['final_test_accuracy']+data_dict[12][1]['final_test_accuracy']+data_dict[12][2]['final_test_accuracy'])/3.0
    # acc36=(data_dict[36][0]['final_test_accuracy']+data_dict[36][1]['final_test_accuracy']+data_dict[36][2]['final_test_accuracy'])/3.0
    acc108=(data_dict[108][0]['final_test_accuracy']+data_dict[108][1]['final_test_accuracy']+data_dict[108][2]['final_test_accuracy'])/3.0
    return acc108
    

def get_basic(network, imgsize, space = 'cv'):
    if space == 'cv':
        from copy import deepcopy
        cp_network = deepcopy(network)
        macs, netparams = get_model_complexity_info(cp_network, (3, imgsize, imgsize), as_strings=False,
                                            print_per_layer_stat=False, verbose=False)
        cp_network = None
        return macs, netparams, metrics

def compute_score(proxy, network, imgsize=None, trainbatches=None, lossfunc=None, bincount=None):
    if proxy =='basic':
        #macs, params, matrics(acc)
        score = get_basic(network, imgsize)[1]
    elif proxy =='ntk':
        score = compute_ntk(trainbatches[0], network, train_mode=True, num_batch=1)
    elif proxy =='logdet':
        score = compute_naswot_score(network, args, trainbatches[0][0], trainbatches[0][1])
    elif proxy =='zen':
        score = compute_zen(network, imgsize, args.batchsize)
    elif proxy =='grad':
        #gradnorm, snip, grasp, fisher, jacob_cov, plain, synflow
        scorelist = get_grad_score(network,  trainbatches[0][0], trainbatches[0][1], lossfunc, split_data=1, device='cuda')
        score = scorelist[-1]  
    elif proxy == 'zico':
        score = compute_zico(network, trainbatches, lossfunc=lossfunc)
    elif proxy == 'intgrad_mean':
        score = compute_intgrad_mean(network, trainbatches, lossfunc=lossfunc, bincount=bincount)
    elif proxy == 'intgrad_param':
        score = compute_intgrad_param(network, trainbatches, lossfunc=lossfunc, bincount=bincount)
    return score

def enumerate_networks(args):
    imgsize, ce_loss, trainloader, testloader = getmisc(args.dataset, args.batchsize, args.manualSeed, args.num_worker)
    print(f'imgsize: {imgsize}, celoss: {ce_loss}, trainloader size: {len(trainloader)}')
    if args.metric == 'mix':
        evaluated_metrics = ['logdet', 'intgrad']
    else:
        evaluated_metrics = [args.metric]        
    
    scores = {}
    file_handlers = {}
    eva_times = {}

    out_path = os.path.join(args.metrics_root, args.searchspace, args.dataset)
    os.makedirs(out_path, exist_ok=True)

    if '101' in args.searchspace.lower():
        assert args.dataset == "cifar10"
        NASBENCH_TFRECORD = '/home/fuwei/dataset/nasbench/nasbench_full.tfrecord'
        nasbench = api101.NASBench(NASBENCH_TFRECORD)

        allnethash = list(nasbench.hash_iterator())
        len_allnethash = len(allnethash)
        len_allnethash = 500
        for keys in evaluated_metrics:
            scores[keys] = np.zeros(len(allnethash))
        accs = np.zeros(len(allnethash))

        file_handlers['acc'] = open(os.path.join(out_path, f'acc.csv'), 'a')
        for proxy in evaluated_metrics:
            data = None
            label = None
            fpath = os.path.join(out_path, f'{proxy}.csv')
            if os.path.exists(fpath):
                print(f'metric {proxy} already compute, skip...')
                continue
            file_handlers[proxy] = open(fpath, 'a')
            
            trainbatches = []
            if proxy in ['zico', 'intgrad_param', 'intgrad_mean', 'logdet', 'grad']:
                for batchid, batch in enumerate(trainloader):
                    if batchid == args.maxbatch:
                        break
                    datax, datay = batch[0].cuda(), batch[1].cuda()
                    # print("init batch{}...\nx:{}, y:{}".format(batchid, datax,datay))
                    trainbatches.append([datax, datay])

            for netid in tqdm(range(len_allnethash)):
                unique_hash = allnethash[netid]
                fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
                acc_metrics= get101acc(computed_metrics)
                accs[netid] = torch.mean(torch.as_tensor(acc_metrics))
                
                ops = fixed_metrics['module_operations']
                adjacency = fixed_metrics['module_adjacency']

                network = NB101Network((adjacency, ops))
                network.cuda()
                score = compute_score(proxy, network, None, imgsize, trainloader, trainbatches, data, label, ce_loss)      
                scores[proxy][netid] = score

        num_net = len_allnethash
        # print("acc: "+str(accs))
        # print("scores: "+str(scores))
        tmp_scores = np.zeros(num_net)
        for proxy in evaluated_metrics:
            tmp_scores[:num_net] = scores[proxy][:num_net]/np.max(np.abs(scores[proxy][:num_net])) + tmp_scores[:num_net]
        tau, p = stats.kendalltau(accs[:num_net], tmp_scores[:num_net])
        spearman = stats.spearmanr(accs[:num_net], tmp_scores[:num_net])
        print('{} archs have been evaluated, \nkendalltau is {}\nspearman is {}'.format(num_net, tau, spearman[0]))

            # if netid > 4000:
            #     print(accs, scores)
            #     break
                
    elif '201' in args.searchspace.lower():
        api = API201('/home/fuwei/dataset/nasbench/NAS-Bench-201-v1_1-096897.pth', verbose=False)
        
        len_api = (len(api))
        print("dataset: {}".format(args.dataset))
        print("using {}".format(args.searchspace))
        print("total {} archs in benchmark".format(len_api))
        len_api = 100
        print("evaluating {} archs......".format(len_api))

        for keys in evaluated_metrics:
            scores[keys] = np.zeros(len_api)
            eva_times[keys] = np.zeros(len_api)
        accs = np.zeros(len_api)

        for proxy in evaluated_metrics:
            print("------proxy: {}".format(proxy))
            data = None
            label = None
            fpath = os.path.join(out_path, f'{proxy}.csv')
            if os.path.exists(fpath):
                print(f'metric {proxy} already compute, skip...')
                continue
            file_handlers[proxy] = open(fpath, 'a')

            trainbatches = []
            if proxy in ['zico', 'intgrad_param', 'intgrad_mean', 'logdet', 'grad']:
                for batchid, batch in enumerate(trainloader):
                    if batchid == args.maxbatch:
                        break
                    datax, datay = batch[0].cuda(), batch[1].cuda()
                    # print("init batch{}...\nx:{}, y:{}".format(batchid, datax,datay))
                    trainbatches.append([datax, datay])

            for netid in tqdm(range(len_api)):
            #TODO: net11 output zero
            # for netid in tqdm(range(10,11)):
                network, metric = search201(api, netid, args.dataset)
                network.cuda()

                accs[netid] = torch.mean(torch.as_tensor(metric[3]))

                start_time = time.time()
                score = compute_score(proxy, network, metric, imgsize, trainloader, trainbatches, data, label, ce_loss)  
                scores[proxy][netid] = score
        
        num_net = len_api
        tmp_scores = np.zeros(num_net)
        for proxy in evaluated_metrics:
            tmp_scores[:num_net] = scores[proxy][:num_net]/np.max(np.abs(scores[proxy][:num_net])) + tmp_scores[:num_net]
        tau, p = stats.kendalltau(accs[:num_net], tmp_scores[:num_net])
        spearman = stats.spearmanr(accs[:num_net], tmp_scores[:num_net])
        print('{} archs have been evaluated, \nkendalltau is {}\nspearman is {}'.format(num_net, tau, spearman[0]))

    elif 'nats' in args.searchspace.lower():
        if 'tss' in args.searchspace.lower():
            # Create the API instance for the topology search space in NATS
            api = create_nats('/home/fuwei/dataset/nasbench/NATS/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)
            hpval='200'
        else:
            # Create the API instance for the size search space in NATS
            api = create_nats('/home/fuwei/dataset/nasbench/NATS/NATS-sss-v1_0-50262-simple', 'sss', fast_mode=True, verbose=False)
            hpval='90'

        len_api = (len(api))
        print("dataset: {}".format(args.dataset))
        print("using {}".format(args.searchspace))
        print("total {} archs in benchmark".format(len_api))
        print("evaluating {} archs......".format(len_api))
        for keys in evaluated_metrics:
            scores[keys] = np.zeros(len_api)
            eva_times[keys] = np.zeros(len_api)
        accs = np.zeros(len_api)

        if not os.path.exists(os.path.join(os.path.join(out_path, f'acc.csv'), 'a')):
            file_handlers['acc'] = open(os.path.join(out_path, f'acc.csv'), 'a')


        for proxy in evaluated_metrics:
            print("------proxy: {}".format(proxy))
            data = None
            label = None
            fpath = os.path.join(out_path, f'{proxy}.csv')
            if not os.path.exists(fpath):
                file_handlers[proxy] = open(fpath, 'a')
            
                trainbatches = []
                if proxy in ['zico', 'intgrad_param', 'intgrad_mean', 'logdet', 'grad']:
                    for batchid, batch in enumerate(trainloader):
                        if batchid == args.maxbatch:
                            break
                        datax, datay = batch[0].cuda(), batch[1].cuda()
                        # print("init batch{}...\nx:{}, y:{}".format(batchid, datax,datay))
                        trainbatches.append([datax, datay])
                for netid in tqdm(range(len_api)):
                # for netid in [64, 81]:
                    network, metric = search_nats(api, netid, args.dataset, hpval)
                    network.cuda()
                    accs[netid] = metric
                    start_time = time.time()
                    score = compute_score(proxy, network, imgsize, trainbatches, ce_loss) 
                    eva_times[proxy][netid] = time.time() - start_time
                    scores[proxy][netid] = score
                    print(f'net: {netid}, score: {score}, acc: {metric}, time: {eva_times[proxy][netid]}')
            else:
                for netid in tqdm(range(len_api)):
                # for netid in [64, 81]:
                    # network, metric = search_nats(api, netid, args.dataset, hpval)
                    # network.cuda()
                    # start_time = time.time()
                    score, acc = evaluate_one_network(args.searchspace, netid, args.dataset, proxy, args.batchsize)
                    accs[netid] = acc
                    # score = compute_score(proxy, network, imgsize, trainbatches, ce_loss) 
                    # eva_times[proxy][netid] = time.time() - start_time
                    scores[proxy][netid] = score
                    # print(f'net: {netid}, score: {score}, acc: {metric}, time: {eva_times[proxy][netid]}')

        num_net = len(accs)
        accs = np.asarray(accs)
        tmp_scores = np.zeros(num_net)
        for proxy in evaluated_metrics:
            npscore = np.asarray(scores[proxy])
            print(np.argmax(npscore))
            tmp_scores = npscore /np.max(np.abs(npscore)) + tmp_scores
        tau, p = stats.kendalltau(accs[:num_net], tmp_scores[:num_net])
        spearman = stats.spearmanr(accs[:num_net], tmp_scores[:num_net])
        print('{} archs have been evaluated, \nkendalltau is {}\nspearman is {}'.format(num_net, tau, spearman[0]))

    elif 'hybrid' in args.searchspace.lower():
        #cifar10
        databasePath = "/home/fuwei/save_dir_sample/hybridDb.csv"
        # databasePath = "/home/fuwei/survey-zero-shot-nas/test.csv"
        netinfo = pd.read_csv(databasePath, delimiter=', ')
        lennetwork = len(netinfo["netid"])
        
        for keys in evaluated_metrics:
            scores[keys] = []
        accs = []

        print("dataset: {}".format(args.dataset))
        print("using {}".format(args.searchspace))
        print("evaluating {} archs......".format(lennetwork))

        file_handlers['acc'] = open(os.path.join(out_path, f'acc.csv'), 'a')

        for proxy in evaluated_metrics:
            print("------proxy: {}".format(proxy))
            data = None
            label = None
            fpath = os.path.join(out_path, f'{proxy}.csv')
            if os.path.exists(fpath):
                print(f'metric {proxy} already compute, skip...')
                continue
            file_handlers[proxy] = open(fpath, 'a')
            
            trainbatches = []
            if proxy in ['zico', 'intgrad_param', 'intgrad_mean', 'logdet', 'grad']:
                for batchid, batch in enumerate(trainloader):
                    if batchid == args.maxbatch:
                        break
                    datax, datay = batch[0].cuda(), batch[1].cuda()
                    # print("init batch{}...\nx:{}, y:{}".format(batchid, datax,datay))
                    trainbatches.append([datax, datay])
            for i in tqdm(range(lennetwork)):
                netstr = netinfo["netstr"][i]
                params = netinfo["param"][i]
                flops = netinfo["flops"][i]
                acc = netinfo["acc"][i]
                if acc < 85:
                    continue

                network = MasterNet(opt=args, num_classes=10, plainnet_struct=netstr, no_reslink=False)
                network.cuda()


                score = compute_score(proxy, network, imgsize, trainbatches, ce_loss)  
                # print(score)
                # score = params
                # print(score)
                scores[proxy].append(score)
                accs.append(acc)
        num_net = len(accs)
        accs = np.asarray(accs)
        tmp_scores = np.zeros(num_net)
        for proxy in evaluated_metrics:
            npscore = np.asarray(scores[proxy])
            tmp_scores[:num_net] = scores[proxy][:num_net]/np.max(np.abs(scores[proxy][:num_net])) + tmp_scores[:num_net]
        tau, p = stats.kendalltau(accs[:num_net], tmp_scores[:num_net])
        spearman = stats.spearmanr(accs[:num_net], tmp_scores[:num_net])
        print('{} archs have been evaluated, \nkendalltau is {}\nspearman is {}'.format(num_net, tau, spearman[0]))


    if args.cache_metric:
        print("caching computed metrics......")
        for proxy in evaluated_metrics:
            if proxy not in file_handlers.keys():
                continue
            file_handlers[proxy].write(f'index,{proxy},time=[s]\n')
            lenscore = len(scores[proxy])
            for netid in range(lenscore):
                line = f'{netid},{scores[proxy][netid]},{eva_times[proxy][netid]}\n'
                file_handlers[proxy].write(line)
            file_handlers[proxy].flush()
        lenacc = len(accs)
        file_handlers['acc'].write("netid,acc\n")
        for netid in range(lenacc):
            line = f'{netid},{accs[netid]}\n'
            file_handlers['acc'].write(line)
        file_handlers['acc'].flush()

        for proxy in evaluated_metrics:
            file_handlers[proxy].close()
        file_handlers['acc'].close()

def evaluate_one_network(searchspace, netid, dataset, proxy, batchsize, metrics_root="cached_metrics"):
# compute a network's metric score
    obtain_score = False
    obtain_acc = False

    out_path = os.path.join(metrics_root, searchspace, dataset)
    fpath = os.path.join(out_path, f'{proxy}.csv')
    if os.path.exists(fpath):
        netinfo = pd.read_csv(fpath, delimiter=',')
        if "index" in netinfo.keys() and proxy in netinfo.keys():
            npindex = np.array(netinfo["index"])
            idx = np.where(npindex==netid)[0][0]
            score = netinfo[proxy][idx]
            print(f'read score {score}')
            obtain_score = True

    accpath = os.path.join(out_path, f'acc.csv')
    if os.path.exists(accpath):
        netinfo = pd.read_csv(accpath, delimiter=',')
        if "index" in netinfo.keys() and "acc" in netinfo.keys():
            npindex = np.array(netinfo["index"])
            idx = np.where(npindex==netid)[0][0]
            acc = netinfo["acc"][idx]
            print(f'read acc {acc}')
            obtain_acc = True

    if obtain_acc and obtain_acc:
        return score, acc

    imgsize, ce_loss, trainloader, _ = getmisc(dataset, batchsize)

    if '101' in searchspace.lower():
        assert args.dataset == "cifar10"
        NASBENCH_TFRECORD = '/home/fuwei/dataset/nasbench/nasbench_full.tfrecord'
        nasbench = api101.NASBench(NASBENCH_TFRECORD)

        allnethash = list(nasbench.hash_iterator())

        trainbatches = []
        if proxy in ['zico', 'intgrad', 'logdet', 'grad']:
            for batchid, batch in enumerate(trainloader):
                if batchid == args.maxbatch:
                    break
                datax, datay = batch[0].cuda(), batch[1].cuda()
                # print("init batch{}...\nx:{}, y:{}".format(batchid, datax,datay))
                trainbatches.append([datax, datay])

        unique_hash = allnethash[netid]
        fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
        acc_metrics= get101acc(computed_metrics)
        acc = torch.mean(torch.as_tensor(acc_metrics))
        
        if not obtain_score:
            ops = fixed_metrics['module_operations']
            adjacency = fixed_metrics['module_adjacency']

            network = NB101Network((adjacency, ops))
            network.cuda()
            score = compute_score(proxy, network, None, imgsize, trainloader, trainbatches, data, label, ce_loss)

    elif '201' in args.searchspace.lower():
        api = API201('/home/fuwei/dataset/nasbench/NAS-Bench-201-v1_1-096897.pth', verbose=False)

        trainbatches = []
        if proxy in ['zico', 'intgrad', 'logdet', 'grad']:
            for batchid, batch in enumerate(trainloader):
                if batchid == args.maxbatch:
                    break
                datax, datay = batch[0].cuda(), batch[1].cuda()
                # print("init batch{}...\nx:{}, y:{}".format(batchid, datax,datay))
                trainbatches.append([datax, datay])

        network, metric = search201(api, netid, args.dataset)
        network.cuda()

        acc = torch.mean(torch.as_tensor(metric[3]))
        if not obtain_score:
            score = compute_score(proxy, network, metric, imgsize, trainloader, trainbatches, data, label, ce_loss)
    elif 'nats' in args.searchspace.lower():
        if 'tss' in args.searchspace.lower():
            # Create the API instance for the topology search space in NATS
            api = create_nats('/home/fuwei/dataset/nasbench/NATS/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)
            hpval='200'
        else:
            # Create the API instance for the size search space in NATS
            api = create_nats('/home/fuwei/dataset/nasbench/NATS/NATS-sss-v1_0-50262-simple', 'sss', fast_mode=True, verbose=False)
            hpval='90'

        trainbatches = []
        if proxy in ['zico', 'intgrad', 'logdet', 'grad']:
            for batchid, batch in enumerate(trainloader):
                if batchid == args.maxbatch:
                    break
                datax, datay = batch[0].cuda(), batch[1].cuda()
                # print("init batch{}...\nx:{}, y:{}".format(batchid, datax,datay))
                trainbatches.append([datax, datay])

        network, metric = search_nats(api, netid, args.dataset, hpval)
        network.cuda()
        acc = metric
        if not obtain_score:
            score = compute_score(proxy, network, imgsize, trainbatches, ce_loss) 

    elif 'hybrid' in args.searchspace.lower():
        trainbatches = []
        if proxy in ['zico', 'intgrad', 'logdet', 'grad']:
            for batchid, batch in enumerate(trainloader):
                if batchid == args.maxbatch:
                    break
                datax, datay = batch[0].cuda(), batch[1].cuda()
                trainbatches.append([datax, datay])
        acc = netinfo["acc"][netid]

        network = MasterNet(opt=args, num_classes=10, plainnet_struct=netstr, no_reslink=False)
        network.cuda()

        if not obtain_score:
            score = compute_score(proxy, network, acc, imgsize, trainloader, trainbatches, data, label, ce_loss)

    return score, acc

if __name__ == '__main__':
    # print("enumerating")
    score, acc = evaluate_one_network("nats_tss", 1, "cifar100", "intgrad_param", 64)
    # print(score)
    # enumerate_networks(args)



