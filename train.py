from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from torchvision.transforms import v2
import os
import json

from cll_experiment.datasets import get_dataset
from cll_experiment.models import get_resnet18, get_modified_resnet18
from cll_experiment.algo import ga_loss, robust_ga_loss
from cll_experiment.valid import compute_ure, compute_scel, validate
from cll_experiment.utils import get_args, get_dataset_T

num_classes = 10
eval_n_epoch = 5
epochs = 300
batch_size = 512
num_workers = 4
device = "cuda"

def train(args):
    algo = args.algo
    model = args.model
    lr = args.lr
    seed = args.seed
    dataset_name = args.dataset_name
    os.makedirs("logs/", exist_ok=True)
    # data_aug = True if args.data_aug.lower()=="true" else False

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    trainset, validset, testset, ord_trainset, ord_validset, num_classes = get_dataset(args)

    # Print the complementary label distribution T
    dataset_T = get_dataset_T(trainset, num_classes)
    dataset_T = torch.tensor(dataset_T, dtype=torch.float).to(device)

    # Set Q for forward algorithm
    if algo in ["fwd-u", "ure-ga-u"]:
        Q = torch.full([num_classes, num_classes], 1/(num_classes-1), device=device)
        for i in range(num_classes):
            Q[i][i] = 0
    elif algo in ["fwd-r", "ure-ga-r"] or algo[:3] == "rob":
        Q = dataset_T
    elif algo == "fwd-int":
        U = np.full([num_classes, num_classes], 1/(num_classes-1))
        for i in range(num_classes):
            U[i][i] = 0
        dataset_T = get_dataset_T(trainset, num_classes)
        Q = torch.tensor(args.alpha * U + (1-args.alpha) * dataset_T).to(device).float()
        dataset_T = torch.tensor(dataset_T, dtype=torch.float).to(device)

    count_cls_wrong_label = np.zeros(num_classes)
    count_wrong_label = 0
    for i in range(len(trainset)):
        if trainset.dataset.targets[i] == trainset.dataset.ord_labels[i]:
            count_cls_wrong_label[trainset.dataset.targets[i]] += 1
            count_wrong_label += 1

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    ord_trainloader = DataLoader(ord_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    ord_validloader = DataLoader(ord_validset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f'use augment: {args.data_aug}')
    # if args.cutmix:
    #     print('use cutmix')
    #     cutmix = v2.CutMix(num_classes=num_classes)

    train_labels = torch.tensor(np.array(trainset.dataset.targets), dtype=torch.int).squeeze()
    class_prior = train_labels.bincount().float() / train_labels.shape[0]

    if args.model == "resnet18":
        model = get_resnet18(num_classes).to(device)
    elif args.model == "m-resnet18":
        model = get_modified_resnet18(num_classes).to(device)
    else:
        raise NotImplementedError
    model.device = device
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    validation_obj = ["valid_acc", "ure", "scel", "last"]
    best_epoch = {obj: None for obj in validation_obj}
    wandb.login()
    wandb.init(project=args.dataset_name, name=f"{algo}-{dataset_name}-{lr}-{seed}", config={"lr": lr, "seed": seed}, tags=[algo])

    with tqdm(range(epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            training_loss = 0.0
            scel = 0
            ure = 0
            model.train()

            for inputs, labels in trainloader:

                # if args.cutmix:
                #     inputs, labels = cutmix(inputs, labels)
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                ure += compute_ure(outputs, labels, dataset_T)
                scel += compute_scel(outputs, labels, algo, dataset_T)
                    
                if algo == "scl-exp":
                    outputs = F.softmax(outputs, dim=1)
                    loss = -F.nll_loss(outputs.exp(), labels)
                    loss.backward()
                
                elif algo[:6] == "ure-ga":
                    loss = ga_loss(outputs, labels, class_prior, Q, num_classes)
                    if torch.min(loss) > 0:
                        loss = loss.sum()
                        loss.backward()
                    else:
                        beta_vec = torch.zeros(num_classes, requires_grad=True).to(device)
                        loss = torch.minimum(beta_vec, loss).sum() * -1
                        loss.backward()
                
                elif algo[:3] == "fwd":
                    q = torch.mm(F.softmax(outputs, dim=1), Q) + 1e-6
                    loss = F.nll_loss(q.log(), labels.squeeze())
                    loss.backward()
                
                elif algo == "l-w":
                    outputs1 = 1 - F.softmax(outputs, dim=1)
                    loss = F.cross_entropy(outputs1, labels.squeeze(), reduction='none')
                    w = (1-F.softmax(outputs, dim=1)) / (num_classes-1)
                    w = 1-F.nll_loss(w, labels.squeeze(), reduction='none')
                    loss = (loss * w).mean()
                    loss.backward()
                
                elif algo == "l-uw":
                    outputs = 1 - F.softmax(outputs, dim=1)
                    loss = F.cross_entropy(outputs, labels.squeeze())
                    loss.backward()
                
                elif algo == "scl-nl":
                    p = (1 - F.softmax(outputs, dim=1) + 1e-6).log()
                    loss = F.nll_loss(p, labels)
                    loss.backward()
                
                elif algo == "pc-sigmoid":
                    outputs = outputs + F.nll_loss(outputs, labels, reduction='none').view(-1, 1)
                    loss = torch.sigmoid(-1 * outputs).sum(dim=1).mean() - 0.5
                    loss.backward()
                
                elif algo == "fwd-int":
                    q = torch.mm(F.softmax(outputs, dim=1), Q) + 1e-6
                    loss = F.nll_loss(q.log(), labels.squeeze())
                    loss.backward()
                
                elif algo[:3] == "rob":
                    loss = robust_ga_loss(outputs, labels, class_prior, Q, num_classes, algo)
                    if torch.min(loss) > 0:
                        loss = loss.sum()
                        loss.backward()
                    else:
                        beta_vec = torch.zeros(num_classes, requires_grad=True).to(device)
                        loss = torch.minimum(beta_vec, loss).sum() * -1
                        loss.backward()

                else:
                    raise NotImplementedError
                
                optimizer.step()
                training_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

            ure /= len(trainloader)
            training_loss /= len(trainloader)
            scel /= len(trainloader)
            wandb.log({"ure": ure, "training_loss": training_loss, "scel": scel})

            if (epoch+1) % eval_n_epoch == 0:
                model.eval()
                train_acc, valid_acc = validate(model, ord_trainloader), validate(model, ord_validloader)
                test_acc = validate(model, testloader)
                epoch_info = {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "valid_acc": valid_acc,
                    "test_acc": test_acc,
                    "ure": ure.item(),
                    "scel": scel.item(),
                    "training_loss": training_loss
                }
                print(train_acc, valid_acc, test_acc)
                wandb.log({"train_acc": train_acc, "valid_acc": valid_acc, "test_acc": test_acc})
                if best_epoch["valid_acc"] is None or valid_acc > best_epoch["valid_acc"]["valid_acc"]:
                    best_epoch["valid_acc"] = epoch_info
                if best_epoch["ure"] is None or ure < best_epoch["ure"]["ure"]:
                    best_epoch["ure"] = epoch_info
                if best_epoch["scel"] is None or scel < best_epoch["scel"]["scel"]:
                    best_epoch["scel"] = epoch_info
                best_epoch["last"] = epoch_info
                print(best_epoch)
                with open(f"logs/{algo}-{dataset_name}-{lr}-{seed}.json", "w") as f:
                    json.dump(best_epoch, f)
        wandb.summary["best_epoch-valid_acc"] = best_epoch["valid_acc"]
        wandb.summary["best_epoch-ure"] = best_epoch["ure"]
        wandb.summary["best_epoch-scel"] = best_epoch["scel"]
        wandb.finish()  

if __name__ == "__main__":
    args = get_args()
    train(args)
