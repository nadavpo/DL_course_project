import logging
import os
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFile
from tqdm import tqdm

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataAugmentation:

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image)

    @staticmethod
    def randomRotation(image, label, mode=Image.BICUBIC):
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)

    @staticmethod
    def randomCrop(image, label):
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(40, 68)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region), label

    @staticmethod
    def randomColor(image, label):
        random_factor = np.random.randint(0, 31) / 10.
        color_image = ImageEnhance.Color(image).enhance(random_factor)
        random_factor = np.random.randint(10, 21) / 10.
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(10, 21) / 10.
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = np.random.randint(0, 31) / 10.
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor), label

    @staticmethod
    def randomGaussian(image, label, mean=0.2, sigma=0.3):
        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        img = np.asarray(image)
        img.flags.writeable = 1
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img)), label

    @staticmethod
    def saveImage(image, path):
        image.save(path)


def imageOps(func_name, image, label, img_des_path, label_des_path, img_file_name, label_file_name, times=3):
    funcMap = {"randomRotation": DataAugmentation.randomRotation,
               "randomCrop": DataAugmentation.randomCrop,
               "randomColor": DataAugmentation.randomColor,
               "randomGaussian": DataAugmentation.randomGaussian
               }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1

    for _i in range(0, times):
        new_image, new_label = funcMap[func_name](image, label)
        DataAugmentation.saveImage(new_image, os.path.join(img_des_path, func_name + str(_i) + img_file_name))
        DataAugmentation.saveImage(new_label, os.path.join(label_des_path, func_name + str(_i) + label_file_name))


opsList = {"randomRotation", "randomColor", "randomGaussian"}


def threadOPS(img_path, new_img_path, label_path, new_label_path):
    img_names = os.listdir(img_path)
    label_names = os.listdir(label_path)

    img_names = [im for im in img_names if not im.endswith('ini')]
    label_names = [im for im in label_names if not im.endswith('ini')]

    img_num = len(img_names)
    label_num = len(label_names)

    assert img_num == label_num, "img_num == label_num"
    num = img_num

    for i in tqdm(range(num)):
        img_name = img_names[i]
        label_name = label_names[i]

        tmp_img_name = os.path.join(img_path, img_name)
        tmp_label_name = os.path.join(label_path, label_name)

        image = DataAugmentation.openImage(tmp_img_name)

        label = DataAugmentation.openImage(tmp_label_name)

        for ops_name in opsList:
            imageOps(ops_name, image, label, new_img_path, new_label_path, img_name,
                     label_name)


# Please modify the path
if __name__ == '__main__':
    # DRIVE
    threadOPS(r"G:\My Drive\final_project\code\SA_Uet-pytorch-master\DRIVE\gan_data_new\images",  # set your path of training images
              r"G:\My Drive\final_project\code\SA_Uet-pytorch-master\DRIVE\gan_data_new_aug\images",
              r"G:\My Drive\final_project\code\SA_Uet-pytorch-master\DRIVE\gan_data_new\1st_manual",  # set your path of training labels
              r"G:\My Drive\final_project\code\SA_Uet-pytorch-master\DRIVE\gan_data_new_aug\1st_manual")

    # CHANSEDB1
    # os.makedirs("CHASEDB1/aug/images")
    # os.makedirs("CHASEDB1/aug/1st_label")
    # threadOPS("CHASEDB1/train/images",  # set your path of training images
    #           "CHASEDB1/aug/images",
    #           "CHASEDB1/train/1st_label",  # set your path of training labels
    #           "CHASEDB1/aug/1st_label")
