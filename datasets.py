import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, num_data_exp=-1, transforms=None):
        super(DriveDataset, self).__init__()
        self.transforms = transforms
        ext_opt = ['.png', '.tif', '.gif']
        img_names = []
        manual_names = []
        for ext in ext_opt:
            img_names_ = [i for i in os.listdir(os.path.join(root, "images")) if i.endswith(ext)]
            for im in img_names_:
                img_names.append(im)
        if num_data_exp < 0 or num_data_exp > len(img_names):
            self.img_list = [os.path.join(root, "images", i) for i in img_names]
        else:
            inds = np.random.choice(len(img_names), num_data_exp, replace=False)
            img_names_ = [img_names[i] for i in range(len(img_names)) if i in inds]
            self.img_list = [os.path.join(root, "images", i) for i in img_names_]

        for ext in ext_opt:
            manual_names_ = [i for i in os.listdir(os.path.join(root, "1st_manual")) if i.endswith(ext)]
            for im in manual_names_:
                manual_names.append(im)
        if num_data_exp < 0 or num_data_exp > len(manual_names):
            self.manual = [os.path.join(root, "1st_manual", i) for i in manual_names]
        else:
            manual_names_ = [manual_names[i] for i in range(len(manual_names)) if i in inds]
            self.manual = [os.path.join(root, "1st_manual", i) for i in manual_names_]
        print(len(self.img_list))
        print(len(self.manual))
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                print(f"file {i} does not exists.")

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_list[idx])
            mask = Image.open(self.manual[idx]).convert('L')

            img = img.resize((565, 584))
            mask = mask.resize((565, 584))
            if self.transforms is not None:
                img, mask = self.transforms(img, mask)
        except:
            a=1
        return img, mask

    def __len__(self):
        return len(self.img_list)


class Chasedb1Datasets:
    def __init__(self, root: str, train: bool, transforms=None):
        super().__init__()
        data_root = os.path.join(root, "aug" if train else "test")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".jpg")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_label", i.split(".")[0] + "_1stHO.png")
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                print(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        mask = Image.open(self.manual[idx]).convert('L')
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)
