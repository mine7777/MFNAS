import torch
import numpy as np
import torch.nn as nn

def getgrad_param(model:torch.nn.Module, grad_dict:dict, step_iter=0, step = 1):
    print("get grad")
    cnt = 0
    for seq, mod in model._modules.items():
        if seq not in ['cells', 'module_list']:
            continue
        for _, cellmod in mod._modules.items():
            for name, submod in cellmod.named_modules():
                if isinstance(submod, nn.Conv2d) or isinstance(submod, nn.Linear):
                    grad = submod.weight.grad.data.cpu().reshape(-1).numpy()
                    if(np.any(grad)): 
                        if step_iter==0:
                            grad_dict[name+str(cnt)]=[grad]
                        elif(np.any(grad)):
                            grad_dict[name+str(cnt)].append(grad)
                        cnt += 1

    return grad_dict


def cal_interval_param(model:torch.nn.Module, grad_dict:dict):
    
    THR_list = [0, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e-0, 5e-0]
    # THR_list = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 10]
    # THR_list = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 5e-0]

    bins = []
    for i in range(len(THR_list) - 1):
        bins.append([THR_list[i], THR_list[i+1]])

    def cal_score(r_list):
        score = 0
        l = len(r_list) - 1
        for i in range(l):
            # score += np.sign(r_list[i+1] - r_list[i])
            score += 1 if (r_list[i+1] - r_list[i]) > 0 else 0
        score /= max(1, l)
        return [score, np.sum(r_list)]

    ratio_info = [[] for _ in bins]

    cnt = 0
    for seq, mod in model._modules.items():
        if seq not in ['cells', 'module_list']:
            continue
        for _, cellmod in mod._modules.items():
            total_params = 0
            params_in_bins = [0 for _ in bins]

            for name, submod in cellmod.named_modules():
                if name+str(cnt) in grad_dict.keys():
                    abs_grad = np.abs(grad_dict[name+str(cnt)])
                    total_params += abs_grad.size
                    for i, (lb, rb) in enumerate(bins):
                        params_in_bins[i] += ((abs_grad >= lb).sum() - (abs_grad >= rb).sum())
                    cnt += 1
                            # print(f'params_in_bins[{i}] = {params_in_bins[i]}')
            for j in range(len(bins)):
                ratio_info[j].append(params_in_bins[j]/max(1, total_params))

    scores = []
    for i in range(len(bins)):
        scores.append(cal_score(ratio_info[i]))
    scores = np.array(scores)
    pidx = np.argsort(-scores[:,1])

    best_idx = pidx[0]
    for it in range(3):
        i = pidx[it]
        v = scores[i]
        best_value = scores[best_idx][0]
        if v[0] > best_value:
            best_idx = i
        elif v[0] == best_value:
            if v[1] > scores[best_idx][1]:
                best_idx = i
                    
    return bins[best_idx], best_idx, bins[pidx[0]], pidx[0]

def caculate_intgrad_param(grad_dict, bins = None):
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
        batchsize = grad_dict[modname].shape[0]
        grad_dict[modname] = grad_dict[modname].reshape((batchsize, -1))

    res = []
    for j, modname in enumerate(grad_dict.keys()):
        grad_mean = np.abs(np.mean(grad_dict[modname], axis=0))
        grad_std = np.std(grad_dict[modname], axis=0) + 1e-5
        tmpsum = 0
        maxtmpsum = 0
        for b in bins:
            stable_idx = (grad_mean >= b[0]) * (grad_mean <= b[1])
            tmpsum = np.sum(grad_mean[stable_idx] / grad_std[stable_idx])
            if tmpsum > maxtmpsum:
                maxtmpsum = tmpsum
        tmpsum = maxtmpsum
        if tmpsum==0 or np.isnan(tmpsum):
            pass
        else:
            tmpsum = np.log(tmpsum+1)
            res.append(tmpsum)
    lenth = len(res)
    if lenth == 0:
        return 1e-5
    else:
        return np.sum(res)

def compute_intgrad_param(network, trainloader, lossfunc, bincount=None):
    print("compute intgrad param")
    grad_dict= {}
    network.train()
    batch_size = 0
    network.cuda()
    meanloss = 0
    for i, batch in enumerate(trainloader):
        network.zero_grad()
        data,label = batch[0],batch[1]
        data,label=data.cuda(),label.cuda()
        logits = network(data)
        if isinstance(logits, tuple):
            logits = logits[1]
        loss = lossfunc(logits, label)
        loss.backward()
        step = loss - 2.25
        meanloss += loss
        grad_dict = getgrad_param(network, grad_dict, i, (step))
        batch_size += 1        

    bins, idx, subbins, subidx = cal_interval_param(network, grad_dict)
    res1 = caculate_intgrad_param(grad_dict, [subbins])

    if bincount is not None:
        bincount.append(subidx)
    return res1
