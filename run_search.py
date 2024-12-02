#!/usr/bin/env python3
import os.path

import numpy as np
from argparse import ArgumentParser
from multif import multif
from utils import seed_all, get_nas_archive_path
from nas_spaces import NATSBench, NASBench101, NATS_SSS
from analyzer import Analyzer
import matplotlib.pyplot as plt
from matplotlib import pyplot

parser = ArgumentParser()
parser.add_argument('--algo', type=str, default='multif', choices=('multif'))
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--max-flops', type=float, default=float('inf'))
parser.add_argument('--max-params', type=float, default=float('inf'))
parser.add_argument('--initial-pop', type=int, default=100)
parser.add_argument('--tournament-size', type=int, default=25)
parser.add_argument('--n-random', type=int, default=0)
parser.add_argument('--max-time', type=float, default=5, help="Maximum time in minutes")  # mins
parser.add_argument('--repeat', type=int, default=30, help="Number of repetitions")
parser.add_argument('--space', type=str, default='nats', choices=('nats_tss', 'nasbench101', 'nats_sss'),
                    help="Search space")
parser.add_argument('--dataset', choices=('cifar10', 'cifar100', 'ImageNet16-120'), default='cifar10',
                    help="Image classification dataset. NasBench101 only supports cifar10")
parser.add_argument('--metrics-root', default='cached_metrics',
                    help="Position of pre-computed metrics")
parser.add_argument('--seed', type=int, default=0,
                    help="Random seed for reproducibility")
parser.add_argument('--metric', type=str, nargs='+', default=None, help="evalate metric")
args = parser.parse_args()

space = args.space
archive_path = get_nas_archive_path(space)


# seed_all(args.seed)

metrics_root = os.path.join(args.metrics_root, args.space)

if space == 'nats_tss':
    api = NATSBench(archive_path, args.dataset, metric_root=metrics_root)
elif space == 'nasbench101':
    api = NASBench101(archive_path, metric_root=metrics_root, progress=True, verbose=True)
elif space == 'nats_sss':
    api = NATS_SSS(archive_path, args.dataset, metric_root=metrics_root)
else:
    raise ValueError("This should never happen.")

palette = pyplot.get_cmap('Set1')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

iters = list(range(100))

algtorun = ['multif']
alldata = []

for alg in algtorun:
    args.algo = alg
    accuracies_hist = []
    times_hist = []
    acc_logs = []
    acc_log = []

    analyzer = Analyzer(api, args.dataset, args.algo)
    for _ in range(args.repeat):
        analyzer.new_run()
        if args.algo == 'multif':
            top1, total_time = multif(api,
                                    N=args.initial_pop,
                                    n=args.tournament_size,
                                    max_flops=args.max_flops,
                                    max_params=args.max_params,
                                    max_time=args.max_time,
                                    n_random=args.n_random,
                                    analyzer=analyzer,
                                    metric=args.metric, 
                                    acc_log=acc_log)

        accuracies_hist.append(top1)
        times_hist.append(total_time)
        acc_logs.append(acc_log)
        acc_log = []

    if args.save:
        analyzer.save()
    print(f"----{alg}----")
    print("Accuracies:")
    accuracies_hist = np.array(accuracies_hist)
    print(f"    Mean: {accuracies_hist.mean()}")
    print(f"     Std: {accuracies_hist.std()}")

    print("Times [min]")
    times_hist = np.array(times_hist)
    print(f"    Mean: {times_hist.mean()}")
    print(f"     Std: {times_hist.std()}")