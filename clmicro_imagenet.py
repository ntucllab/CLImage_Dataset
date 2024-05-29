from torch.utils.data import Dataset
import os
import urllib.request
from tqdm import tqdm
import pickle
from PIL import Image
import gdown

class CLMicro_ImageNet10(Dataset):
    def __init__(self, root="./data", train=True, transform=None):
        os.makedirs(os.path.join(root, 'clmicro_imagenet10'), exist_ok=True)
        if train:
            dataset_path = f"{root}/clmicro_imagenet10_train.pkl"
            gid = "1k02mwMpnBUM9de7TiJLBaCuS8myGuYFx"
        else:
            dataset_path = f"{root}/clmicro_imagenet10_test.pkl"
            gid = "1e8fZN8swbg9wc6BSOC0A5KHIqCY2C7me"
        if not os.path.exists(dataset_path):
            os.makedirs(root, exist_ok=True)
            gdown.download(id=gid, output=dataset_path)
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)


        self.targets = [labels[0] for labels in data["cl_labels"]]
        self.data = data["images"]
        self.ord_labels = data["ord_labels"]
        if train:
            self.targets = [labels[0] for labels in data["cl_labels"]]
        else:
            self.targets = data["ord_labels"]
        self.transform = transform
        self.num_classes = 10
        self.input_dim = 64 * 64 * 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]

class CLMicro_ImageNet20(Dataset):
    def __init__(self, root="./data", train=True, transform=None):
        os.makedirs(os.path.join(root, 'clmicro_imagenet20'), exist_ok=True)
        if train:
            dataset_path = f"{root}/clmicro_imagenet20_train.pkl"
            gid = "1Urdxs_QTxbb1gDBpmjP09Q35btckI3_d"
        else:
            dataset_path = f"{root}/clmicro_imagenet20_test.pkl"
            gid = "1EdBCrifSrIIUg1ioPWA-ZLEHO53P4NPl"
        if not os.path.exists(dataset_path):
            os.makedirs(root, exist_ok=True)
            gdown.download(id=gid, output=dataset_path)
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)


        self.targets = [labels[0] for labels in data["cl_labels"]]
        self.data = data["images"]
        self.ord_labels = data["ord_labels"]
        if train:
            self.targets = [labels[0] for labels in data["cl_labels"]]
        else:
            self.targets = data["ord_labels"]
        self.transform = transform
        self.num_classes = 20
        self.input_dim = 64 * 64 * 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]