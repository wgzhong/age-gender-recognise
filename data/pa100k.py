import math
import numpy as np
import cv2
import yaml
import re
import albumentations as A
from tensorflow.keras.utils import Sequence

transforms = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.03125, scale_limit=0.20, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT,
                       value=0, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HorizontalFlip(p=0.5)
])


class ImageSequence(Sequence):
    def __init__(self, cfg, mode):
        self.path = cfg["data"]["path"]
        self.batch_size = cfg["train"]["batch_size"]
        self.img_size = cfg["model"]["img_size"]
        self.mode = mode
        self.images=[]
        self.labels=[]
        if self.mode == "train":
            self.labels = self.read_txt(self.path+"/train_label.txt")
            self.images = self.read_txt(self.path+"/train_images_name.txt")
        elif self.mode == "test":
            self.labels = self.read_txt(self.path+"/test_label.txt")
            self.images = self.read_txt(self.path+"/test_images_name.txt")
        elif self.mode == "val":
            self.labels = self.read_txt(self.path+"/val_label.txt")
            self.images = self.read_txt(self.path+"/val_images_name.txt")
        assert(len(self.labels)==len(self.images))
        self.num=len(self.labels)

    def __getitem__(self, idx):
        end_idx = (idx + 1) * self.batch_size+1
        if end_idx >= self.num:
            end_idx = self.num
        sample_img_path = self.images[idx * self.batch_size+1: end_idx]
        sample_label = self.labels[idx * self.batch_size+1: end_idx]
        imgs = []
        genders = []
        ages = []
        
        for img_name, label in zip(sample_img_path, sample_label):
            # print(self.path+"/images/"+re.sub('\[|\]|\'','',img_name))
            img = cv2.imread(self.path+"/images/"+re.sub('\[|\]|\'','',img_name))
            img = cv2.resize(img, (self.img_size, self.img_size))

            if self.mode == "train":
                img = transforms(image=img)["image"]

            imgs.append(img)
            genders.append(int(label[0]))
            tmp=[int(label[2]), int(label[4]), int(label[6])]
            ages.append(tmp)

        imgs = np.asarray(imgs)/255.0
        genders = np.asarray(genders)
        ages = np.asarray(ages)
        ages = np.array(ages).astype(np.float32)
        return imgs, (genders, ages)

    def __len__(self):
        return math.ceil(self.num / self.batch_size)
    
    def read_txt(self, path):
        contx=[]
        with open(path, 'r') as f:
            contx = f.read().splitlines()
        return contx

    def on_epoch_end(self):
        np.random.shuffle(self.images)

if __name__=="__main__":
    file = open("../config/config.yaml", 'r', encoding="utf-8")
    cfg = yaml.safe_load(file)      
    train_gen = ImageSequence(cfg, "train")
    for x,y in train_gen:
        print(x)
        print(y)

