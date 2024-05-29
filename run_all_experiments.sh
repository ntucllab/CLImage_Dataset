strategy=$1
strategies=($strategy)
datasets=("uniform-micro_imagenet10" "uniform-micro_imagenet20" "clmicro_imagenet10" "clmicro_imagenet20")
lrs=("5e-4")
seeds=("197" "101" "1126" "3333")

for strategy in ${strategies[@]}; do
    for dataset in ${datasets[@]}; do
        for lr in ${lrs[@]}; do
            for seed in ${seeds[@]}; do
                echo "./run.sh ${strategy} ${dataset} ${lr} ${seed}"
                ./run.sh ${strategy} ${dataset} ${lr} ${seed}
            done
        done
    done
done