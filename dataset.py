from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image

class CatDogDataset(Dataset):
    def __init__(self, dataset_path: str, transform=None):
        """
        Args:
            dataset_path (str): path to the dataset
            transform (callable, optional): transform to be applied on a sample
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.images_path = []
        self.labels = []
        self.classes = ["cat", "dog"]
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}
        self.index_to_class = {i: cls for i, cls in enumerate(self.classes)}

        for image_name in os.listdir(self.dataset_path):
            image_path = os.path.join(self.dataset_path, image_name)
            self.images_path.append(image_path)
            self.labels.append(self.class_to_index[image_name.split(".")[0]])


    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, idx):
        image = read_image(self.images_path[idx]).float()
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_class(self, idx):
        return self.index_to_class[idx]
    
    def get_image(self, idx):
        return self.images_path[idx]
    
    def get_label(self, idx):
        return self.labels[idx]
    
    
class CIFAR10Dataset(Dataset):
    def __init__(self, dataset_path: str, transform=None, target_transform=None, batches=5, dual_input=False):
        """
        Same as CIFAR10Dataset but returns the same image twice if dual input on. In the future it could be accommodated to real dual input. This is just for testing purposes.
        Args:
            dataset_path (str): path to the dataset
            transform (callable, optional): transform to be applied on a sample
            target_transform (callable, optional): transform to be applied on a target
            batches (int): number of batches to load, minimum 1, maximum 5
            dual_input (bool): if True, returns the same image twice
        """

        self.dataset_path = dataset_path
        self.transform = transform
        self.target_transform = target_transform
        self.dual_input = dual_input
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.classes = []
        self.label_to_class = {}
        self.class_to_label = {}

        for i in range(1, batches+1):
            file = os.path.join(self.dataset_path, f"data_batch_{i}")
            data = self.unpickle(file) # b'data', b'labels', b'filenames'
            self.train_images.extend(data[b'data'])
            self.train_labels.extend(data[b'labels'])
            
        file = os.path.join(self.dataset_path, "test_batch")
        data = self.unpickle(file)
        self.test_images.extend(data[b'data'])
        self.test_labels.extend(data[b'labels'])

        self.label_to_class = self.unpickle(os.path.join(self.dataset_path, "batches.meta"))[b'label_names']
        self.classes = [label.decode("utf-8") for label in self.label_to_class]
        self.class_to_label = {i: label for i, label in enumerate(self.label_to_class)}

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def __len__(self):
        return len(self.train_images) + len(self.test_images)

    def __getitem__(self, idx):
        if idx < len(self.train_images):
            image = self.train_images[idx]
            label = self.train_labels[idx]
        else:
            image = self.test_images[idx-len(self.train_images)]
            label = self.test_labels[idx-len(self.train_images)]

        image = image.reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        if self.dual_input:
            return image, image, label
        else:
            return image, label
    
    def get_class(self, idx):
        return self.class_to_label[idx]

    def get_image(self, idx):
        if idx < len(self.train_images):
            return self.train_images[idx]
        else:
            return self.test_images[idx-len(self.train_images)]
        
    def get_label(self, idx):
        if idx < len(self.train_labels):
            return self.train_labels[idx]
        else:
            return self.test_labels[idx-len(self.train_labels)]

    def get_train_size(self):
        return len(self.train_images)
    
    def get_test_size(self):
        return len(self.test_images)
    
    def show_image(self, idx):
        if self.dual_input:
            image, _, label = self[idx]
        else:
            image, label = self[idx]
        print(f"Class: {self.get_class(label)}")
        plt.figure()
        # Assuming it will be a tensor
        plt.imshow(image.permute(1, 2, 0))
        plt.show()

if __name__ == "__main__":
    cifar10 = CIFAR10Dataset("dataset/cifar-10")
    cifar10.show_image(1)
