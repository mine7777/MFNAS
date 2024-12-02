import time
import numpy as np
import torch.nn as nn

from metrics import *
from scipy.stats import mode

from nas_spaces import NASSpaceBase

def get_random_population(api: NASSpaceBase, N, generation=0):
    exemplars = [api.random().set_generation(generation) for _ in range(N)]
    return exemplars


def population_init(api, N, max_flops=float('inf'), max_params=float('inf'), analyzer=None, start=0.0, metrics=None):
    population = []
    while len(population) < N:
        new = get_random_population(api, 1)[0]
        if is_feasible(new, max_flops, max_params):
            population.append(new)
        if analyzer:
            analyzer.update(population, start, metrics)
    return population


def is_feasible(exemplar, max_flops=float('inf'), max_params=float('inf')):
    cost = exemplar.get_cost_info()
    if cost['flops'] <= max_flops and cost['params'] <= max_params:
        return True
    return False

def return_top_k(exemplars, K=3, metric_names=[], besorted = True):
    exemplars = [exemplar for exemplar in exemplars if exemplar.born]

    values_dict = {}
    for metric_n in metric_names:
        if metric_n == 'skip':
            values_dict['skip'] = np.array([exemplar.skip() for exemplar in exemplars])
        else:
            values_dict[metric_n] = np.array([exemplar.get_metric(metric_n) for exemplar in exemplars])

    scores = np.zeros(len(exemplars))
    scores_dict = {}
    for metric_n in metric_names:
        if metric_n == 'ntk':
            values_dict[metric_n] = 1 - values_dict[metric_n] / (np.max(np.abs(values_dict[metric_n])) + 1e-9)
        else:
            values_dict[metric_n] = values_dict[metric_n] / (np.max(np.abs(values_dict[metric_n])) + 1e-9)
        scores_dict[metric_n] = values_dict[metric_n]
        scores += values_dict[metric_n]

    for idx, (exemplar, rank) in enumerate(zip(exemplars, scores)):
        exemplar.rank = rank

    if besorted:
        exemplars.sort(key=lambda x: -x.rank)
    return exemplars[:K]


def get_max_accuracy(api: NASSpaceBase, max_flops: float, max_params: float):
    best, best_accuracy = None, 0.0
    for idx in range(len(api)):
        info = api.get_cost_info(idx)
        if info['flops'] <= max_flops and info['params'] <= max_params:
            accuracy = api.get_accuracy(api[idx])
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best = idx
                print(best, best_accuracy)
    return best_accuracy


def kaiming_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def init_model(model):
    model.apply(kaiming_normal)
    return model


def dictionary_update(exemplars, history, replace=True):
    if replace:
        # Update already seen genotypes
        history.update({exemplar.genotype: exemplar for exemplar in exemplars})
    else:
        # Add new genotypes
        history.update({exemplar.genotype: exemplar for exemplar in exemplars if exemplar.genotype not in history})
        # For already seen exemplars, just update the generation
        for exemplar in exemplars:
            history[exemplar.genotype].generation = exemplar.generation

    # This is also removing exemplars with same genotype from population
    current_genotypes = set([exemplar.genotype for exemplar in exemplars])
    exemplars = [history[genotype] for genotype in current_genotypes]
    return exemplars, history


def clean_history(history, max_params, max_flops):
    return {genotype: exemplar for genotype, exemplar in history.items()
            if exemplar.get_cost_info()['params'] <= max_params
            and exemplar.get_cost_info()['flops'] <= max_flops}


def get_top_k_accuracies(exemplars, K=3, metrics=[], besorted = True):
    best_K, acc = return_top_k(exemplars, K, metrics, besorted), []
    for exemplar in best_K:
        idx = exemplar.idx
        acc.append((idx, round(exemplar.get_accuracy(), 3)))
    return acc, [exemplar.genotype for exemplar in best_K], [exemplar.rank for exemplar in best_K], best_K

def multif_gettop(api, exemplars, history, start, metrics, max_flops, max_params, itper=None):
    history_feasible = clean_history(history, max_params, max_flops).values()
    if itper is None:
        acc_pop, _, _, top_filtered = get_top_k_accuracies(history_feasible, K=3, metrics=metrics)
        return acc_pop[0][1]
    if itper <= 0.5:
        acc_pop, _, _, top_filtered = get_top_k_accuracies(history_feasible, K=3, metrics=metrics)
        return acc_pop[0][1]

    if itper > 0.5:
        acc_pop, _, _, top_filtered = get_top_k_accuracies(history_feasible, K=3, metrics=metrics)
        # print(f'top filtered: {acc_pop}')
        top1, _, _, _ = get_top_k_accuracies(top_filtered, K=1, metrics=[metrics[0]])
        # print(f'top 1: {top1}')
        return top1[0][1]


    if itper is not None and itper > 0.2:
        idxlist, _, _, top_filtered = get_top_k_accuracies(history_feasible, K=20, metrics=[metrics[0]])
        acc_pop, genotypes_hist, ranks_hist, _ = get_top_k_accuracies(top_filtered, K=3, metrics=[metrics[1]])
        return acc_pop[0][1]
    elif itper is not None:
        acc_pop, _, _, top_filtered = get_top_k_accuracies(history_feasible, K=3, metrics=[metrics[0]])
        return acc_pop[0][1]


def multif_info(api, exemplars, history, start, metrics, max_flops, max_params):
    history_feasible = clean_history(history, max_params, max_flops).values()
    # idxlist, _, _, top_filtered = get_top_k_accuracies(history_feasible, K=20, metrics=[metrics[0]])
    acc_pop, genotypes_hist, ranks_hist, top_filtered = get_top_k_accuracies(history_feasible, K=3, metrics=metrics)
    top1, _, _, _ = get_top_k_accuracies(top_filtered, K=1, metrics=[metrics[0]])
    exemplars_feasible = [exemplar for exemplar in exemplars if exemplar.born and
                          exemplar.get_cost_info()['flops'] <= max_flops and
                          exemplar.get_cost_info()['params'] <= max_params]
    accuracies = [exemplar.get_accuracy() for exemplar in history_feasible]

    metrics_time = api.total_metrics_time(history_feasible, [metrics[0]]) + api.total_metrics_time(history_feasible, [metrics[1]])
    search_time = time.time() - start
    total_time = metrics_time + search_time

    # This is not a correct number if we loose the constraints at the beginning
    print(f'\n|||| {len(history_feasible)} different cells explored..')
    print(f'|||| Max acc: {round(max(accuracies), 2)}, Avg acc: {round(np.mean(accuracies), 2)}, Std acc: {round(np.std(accuracies), 2)}..')
    print(f'|||| After metric[0] filtered, 20 candidates are: ')
    # print(f'     |||| {idxlist}')
    print(f'|||| Ranks of top 3 best cells: {ranks_hist}')
    print(f'|||| Genotype of top 3 best cells:')
    for index, genotype in enumerate(genotypes_hist):
        print(f'     |||| {acc_pop[index][0]} : {genotype}')
    print(f'|||| Accuracies of top 3 best cells: {acc_pop}')
    print(f'|||| Accuracies of top 1 best cells: {top1}')
    print(f'|||| Search Time: {round(search_time / 60, 2)} minutes..')
    print(f'|||| Metrics Time: {round(metrics_time / 60, 2)} minutes..')
    print(f'|||| Total Time: {round(total_time / 60, 2)} minutes..')

    debug = False
    if(debug):
        # round2accs = [round(acc, 2) for acc in accuracies]
        acc, _, rank = get_top_k_accuracies(history_feasible, K=len(history_feasible), metrics=metrics, besorted=False)
        rank = [round(r, 2) for r in rank]
        print(f'|||| Accuracies of visited cells: {acc}')
        print(f'|||| Scores of visited cells: {rank}')

    return top1[0][1], total_time / 60


def edit_distance(exemplar1, exemplar2):
    genes1 = genotype_to_gene_list(exemplar1.genotype)
    genes2 = genotype_to_gene_list(exemplar2.genotype)

    count = 0
    for gene1, gene2 in zip(genes1, genes2):
        if gene1 != gene2:
            count += 1
    return count


def genotype_to_gene_list(genotype):
    out = []
    levels = genotype.split('+')
    for level in levels:
        level = level.split('|')[1:-1]
        for i in range(len(level)):
            out.append(level[i])
    return out


def gene_list_to_genotype(gene_list):
    gene_list = gene_list[0][0]
    out = '|' + gene_list[0] + '|+|'
    for idx, gene in enumerate(gene_list[1:]):
        out += gene + '|'
        if idx == 1:
            out += '+|'
    return out


def mean_exemplar(exemplars, K=5, metrics=[]):
    _, genotypes_hist, _ = get_top_k_accuracies(exemplars, K=K, metrics=metrics)

    top_genotypes = [genotype_to_gene_list(genotype) for genotype in genotypes_hist]

    mean_genes = mode(top_genotypes, axis=0)
    mean_genotype = gene_list_to_genotype(mean_genes)
    return mean_genotype


class EarlyStop:
    def __init__(self, patience=20):
        super().__init__()
        self.patience = patience
        self.waiting = 0
        self.n_pop = 0

    def stop(self, history):
        n = len(history)
        if n >= self.n_pop:
            self.n_pop = n
            self.waiting = 0
        else:
            self.waiting += 1
            if self.patience >= self.waiting:   #FIXME: patience < waiting
                print('\n   Early Stop...')
                return True
        return False
