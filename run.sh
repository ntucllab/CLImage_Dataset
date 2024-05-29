alg=$1
dataset=$2
lr=$3
seed=$4

python train.py \
    --algo=${alg} \
    --dataset_name=${dataset} \
    --model=m-resnet18 \
    --lr=${lr} \
    --seed=${seed} \
    --data_aug=autoaug