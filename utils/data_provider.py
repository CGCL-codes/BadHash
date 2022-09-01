import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as util_data
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import transforms


class HashingDataset(Dataset):
    def __init__(self,
                 data_path,
                 img_filename,
                 label_filename,
                 transform=transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     #                      std=[0.229, 0.224, 0.225])
                 ])):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)


        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)

        # label_filepath = label_filename
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        # img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = Image.open(self.img_filename[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index]).float()
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


def load_label(filename, data_dir):
    label_filepath = os.path.join(data_dir, filename)
    # label_filepath = filename
    label = np.loadtxt(label_filepath, dtype=np.int64)
    return torch.from_numpy(label).float()


def get_data(config):
    dsets = {}
    dset_loaders = {}

    for data_set in ["train", "test", "database", "test_p"]:

        dsets[data_set] = HashingDataset(config["data_dir"] + config["dataset"], config[data_set +"_file"], config[data_set +"_label"])
        print(data_set, len(dsets[data_set]))

        if data_set == "train":
            dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                          batch_size=config["batch_size"],
                                                          shuffle=True, num_workers=4)
        else:
            dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                          batch_size=config["batch_size"],
                                                          shuffle=False, num_workers=4)

    return dset_loaders["train"], dset_loaders["test"], dset_loaders["database"], dset_loaders["test_p"],\
           len(dsets["train"]), len(dsets["test"]), len(dsets["database"]), len(dsets["test_p"])
