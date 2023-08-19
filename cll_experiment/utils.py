import argparse
import numpy as np

def get_args():
    dataset_list = [
        "uniform-cifar10",
        "uniform-cifar20",
        "clcifar10",
        "clcifar20",
        "clcifar10-noiseless",
        "clcifar20-noiseless",
        "noisy-uniform-cifar10",
        "noisy-uniform-cifar20",
        "clcifar10-n",
        "clcifar20-n",
        "b-clcifar10-n"
    ]

    algo_list = [
        "scl-exp",
        "scl-nl",
        "ure-ga-u",
        "ure-ga-r",
        "fwd-u",
        "fwd-r",
        "l-w",
        "l-uw",
        "pc-sigmoid",
        "fwd-int",
        "rob-mae",
        "rob-cce",
        "rob-wmae",
        "rob-gce",
        "rob-sl"
    ]

    model_list = [
        "resnet18",
        "m-resnet18"
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', type=str, choices=algo_list, help='Algorithm')
    parser.add_argument('--dataset_name', type=str, choices=dataset_list, help='Dataset name')

    parser.add_argument('--model', type=str, choices=model_list, help='Model name', default="resnet")
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--seed', type=int, help='Random seed', default=0)
    parser.add_argument('--data_aug', type=str, default="std")
    parser.add_argument('--data_cleaning_rate', type=float, default=0)
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--cutmix', type=str, default="false")
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    return args

def get_dataset_T(dataset, num_classes):
    dataset_T = np.zeros((num_classes,num_classes))
    class_count = np.zeros(num_classes)
    for i in range(len(dataset)):
        dataset_T[dataset.dataset.ord_labels[i]][dataset.dataset.targets[i]] += 1
        class_count[dataset.dataset.ord_labels[i]] += 1
    for i in range(num_classes):
        dataset_T[i] /= class_count[i]
    return dataset_T
