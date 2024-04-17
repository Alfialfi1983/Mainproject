import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(orig_images_path, hazy_images_path):
    train_list = []
    val_list = []

    image_list_haze = glob.glob(os.path.join(hazy_images_path, "*.jpg"))

    tmp_dict = {}
    for image in image_list_haze:
        image_name = os.path.basename(image)
        key = image_name.split("_")[0] + "_" + image_name.split("_")[1] + ".jpg"
        if key in tmp_dict:
            tmp_dict[key].append(image_name)
        else:
            tmp_dict[key] = [image_name]

    keys = list(tmp_dict.keys())
    random.shuffle(keys)

    train_keys = keys[:int(0.9 * len(keys))]
    val_keys = keys[int(0.9 * len(keys)):]

    for key in keys:
        image_list = tmp_dict[key]
        for hazy_image in image_list:
            if key in train_keys:
                train_list.append([os.path.join(orig_images_path, key), os.path.join(hazy_images_path, hazy_image)])
            else:
                val_list.append([os.path.join(orig_images_path, key), os.path.join(hazy_images_path, hazy_image)])

    random.shuffle(train_list)

    print("Total training examples:", len(train_list))
    print("Total validation examples:", len(val_list))

    return train_list, val_list

class dehazing_loader(data.Dataset):
    def __init__(self, orig_images_path, hazy_images_path, mode='train'):
        self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path)

        self.data_list = self.train_list if mode == 'train' else self.val_list

        print(f"Total {mode} examples:", len(self.data_list))

    def __getitem__(self, index):
        data_orig_path, data_hazy_path = self.data_list[index]

        try:
            data_orig = Image.open(data_orig_path)
            data_hazy = Image.open(data_hazy_path)
        except Exception as e:
            print(f"Error opening image: {e}")
            return None, None

        if data_orig is None or data_hazy is None:
            # Skip this example if either image is None
            return None, None

        data_orig = data_orig.resize((480, 640), Image.LANCZOS)
        data_hazy = data_hazy.resize((480, 640), Image.LANCZOS)

        data_orig = (np.asarray(data_orig) / 255.0).astype(np.float32)
        data_hazy = (np.asarray(data_hazy) / 255.0).astype(np.float32)

        data_orig = torch.from_numpy(data_orig).permute(2, 0, 1)
        data_hazy = torch.from_numpy(data_hazy).permute(2, 0, 1)

        return data_orig, data_hazy

    def __len__(self):
        return len(self.data_list)

