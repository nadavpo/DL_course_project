import os

import numpy as np
from PIL import Image
from tqdm import tqdm


# def compute(img_dir=r"C:\Users\nadav\Documents\MSc Drive\MSc\Deep Learning\final project\code\SA_Uet-pytorch-master\DRIVE\aug\images"):
def compute(img_dir=r"./DRIVE/aug/images"):
    img_channels = 3
    img_name_list = [i for i in os.listdir(img_dir)]
    img_name_list = [i for i in img_name_list if not i.endswith('ini')]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in tqdm(img_name_list):
        img_path = os.path.join(img_dir, img_name)
        # img转化为0到1
        img = np.array(Image.open(img_path)) / 255.
        cumulative_mean += img.mean(axis=(0, 1))
        cumulative_std += img.std(axis=(0, 1))

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    return mean, std


if __name__ == "__main__":
    mean, std = compute("CHASEDB1/aug/images")
    print(mean, std)
    mean1, std1 = compute("CHASEDB1/test/images")
    print(mean1, std1)
