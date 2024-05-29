from torch.utils.data import Dataset
import os
import urllib.request
from tqdm import tqdm
import pickle
import gdown

class CLCIFAR10(Dataset):
    """CLCIFAR10 training set

    The training set of CIFAR10 with human annotated complementary labels.
    Containing 50000 samples, each with one ordinary label and the first one of the three complementary labels

    Args:
        root: the path to store the dataset
        transform: feature transformation function
    """
    def __init__(self, root="./data", transform=None):

        os.makedirs(os.path.join(root, 'clcifar10'), exist_ok=True)
        dataset_path = os.path.join(root, 'clcifar10', f"clcifar10.pkl")

        if not os.path.exists(dataset_path):
            gdown.download(
                id="1uNLqmRUkHzZGiSsCtV2-fHoDbtKPnVt2", output=dataset_path
            )

        data = pickle.load(open(dataset_path, "rb"))

        self.transform = transform
        self.input_dim = 32 * 32 * 3
        self.num_classes = 10

        self.targets = [labels[0] for labels in data["cl_labels"]]
        self.data = data["images"]
        self.ord_labels = data["ord_labels"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]

class CLCIFAR20(Dataset):
    """CLCIFAR20 training set

    The training set of CIFAR20 with human annotated complementary labels.
    Containing 50000 samples, each with one ordinary label and the first one of the three complementary labels

    Args:
        root: the path to store the dataset
        transform: feature transformation function
    """
    def __init__(self, root="./data", transform=None):

        os.makedirs(os.path.join(root, 'clcifar20'), exist_ok=True)
        dataset_path = os.path.join(root, 'clcifar20', f"clcifar20.pkl")

        if not os.path.exists(dataset_path):
            gdown.download(
                id="1PhZsyoi1dAHDGlmB4QIJvDHLf_JBsFeP", output=dataset_path
            )

        data = pickle.load(open(dataset_path, "rb"))

        self.transform = transform
        self.input_dim = 32 * 32 * 3
        self.num_classes = 20

        self.targets = [labels[0] for labels in data["cl_labels"]]
        self.data = data["images"]
        self.ord_labels = data["ord_labels"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]