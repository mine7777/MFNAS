# searchspace: ['basic','ntk','lr', 'logdet', 'grad', 'zen','IB', 'zico', 'intgrad','mix']
# dataset: ['101','201','nats','nats_tss', 'mbnv2', 'resnet', 'hybrid']
#!/bin/bash
searchspace="nats_tss"
dataset="cifar100"
data_path="none"
metric="intgrad_param"

if [ $dataset = 'cifar10' ]; then
    data_path="/home/fuwei/data/pytorch_cifar10/"
elif [ $dataset = "cifar100" ]; then
    data_path="/home/fuwei/data/pytorch_cifar100/"
elif [ $dataset = "ImageNet16-120" ]; then
    data_path="/home/fuwei/dataset/ImageNet16"
fi

CUDA_VISIBLE_DEVICES=1 python networks_eva.py --searchspace=${searchspace} --dataset=${dataset} --data_path=${data_path} --metric=${metric} --maxbatch=2 --batchsize=128
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --searchspace=${searchspace} --dataset=${dataset} --data_path=${data_path} --metric=${metric} --maxbatch=2 --batchsize=128


echo "searchspace: "$searchspace
echo "dataset: "$dataset
echo "using metric: "$metric