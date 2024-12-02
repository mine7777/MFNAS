# searchspace: ['basic','ntk','lr', 'logdet', 'grad', 'zen','IB', 'zico', 'intgrad','mix']
# dataset: ['101','201','nats','nats_tss', 'mbnv2', 'resnet', 'hybrid']
#dataset: ['cifar10','cifar100','ImageNet16-120','imagenet-1k', 'cifar10-valid']
#!/bin/bash
searchspace="nats_tss"
dataset="cifar10"
data_path="none"
metric="intgrad_param"
metric1="intgrad_mean"
algo="multif"
repeat=3

if [ $dataset = 'cifar10' ]; then
    data_path="/home/fuwei/data/pytorch_cifar10/"
elif [ $dataset = "cifar100" ]; then
    data_path="/home/fuwei/data/pytorch_cifar100/"
elif [ $dataset = "ImageNet16-120" ]; then
    data_path="/home/fuwei/dataset/ImageNet16"
fi

CUDA_VISIBLE_DEVICES=1 python run_search.py --space ${searchspace} --dataset ${dataset} --repeat ${repeat} --metric ${metric} ${metric1} --algo ${algo}

echo "searchspace: "$searchspace
echo "dataset: "$dataset
echo "using metric: "$metric
echo "search alg: "${algo}