import torch
import torch.nn.functional as F

def validate(model, dataloader):
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

def compute_ure(outputs, labels, dataset_T):
    with torch.no_grad():
        outputs = -F.log_softmax(outputs, dim=1)
        if torch.det(dataset_T) != 0:
            T_inv = torch.inverse(dataset_T).to(labels.device)
        else:
            T_inv = torch.pinverse(dataset_T).to(labels.device)
        loss_mat = torch.mm(outputs, T_inv.transpose(1, 0))
        ure = -F.nll_loss(loss_mat, labels)
        return ure

def compute_scel(outputs, labels, algo, dataset_T):
    outputs = outputs.softmax(dim=1)
    if algo[:3] != "cpe":
        outputs = torch.mm(outputs, dataset_T)
    outputs = (outputs + 1e-6).log()
    return F.nll_loss(outputs, labels)