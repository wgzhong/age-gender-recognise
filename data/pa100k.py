import math
import numpy as np
import cv2
import yaml
import re
import tensorflow as tf
import albumentations as A
from tensorflow.keras.utils import Sequence
from keras import backend as K
transforms = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.03125, scale_limit=0.20, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT,
                       value=0, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HorizontalFlip(p=0.5)
])


class ImageSequence(Sequence):
    def __init__(self, cfg, mode):
        self.path = cfg["data"]["path"]
        self.datamode = cfg["data"]["mode"]
        self.batch_size = cfg["train"]["batch_size"]
        self.img_size_h = cfg["model"]["img_size_h"]
        self.img_size_w = cfg["model"]["img_size_w"]
        self.mode = mode
        self.shuffle = True
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

        # np.random.seed(200)
        # np.random.shuffle(self.images) 
        # np.random.seed(200)
        # np.random.shuffle(self.labels)
        self.indexes = np.arange(self.num)
    def __getitem__(self, idx):
        end_idx = (idx + 1) * self.batch_size
        if end_idx >= self.num:
            end_idx = self.num
        img_label_idx = self.indexes[idx * self.batch_size: end_idx]
        sample_img_path = [self.images[k] for k in img_label_idx]
        sample_label = [self.labels[k] for k in img_label_idx]
        imgs = []
        genders = []
        ages = []
        image_path = []
        for img_name, label in zip(sample_img_path, sample_label):
            path= self.path+"/images/"+re.sub('\[|\]|\'','',img_name)
            # path = "/home/vastai/zwg/pa100k/images/003275.jpg"
            img = cv2.imread(path)
            # img=cv2.resize(img, (self.img_size, self.img_size))
            img = self.data_enhance(img, re.sub('\[|\]|\'','',img_name))
            # print(img.shape, path)
            # exit(0)
            imgs.append(img)
            # print(path, int(label[0]))
            genders.append(int(label[0]))
            tmp=[int(label[2]), int(label[4]), int(label[6])]
            ages.append(tmp)
            image_path.append(path)

        imgs = np.asarray(imgs)/255.0
        image_path = tf.convert_to_tensor(np.asarray(image_path))
        genders = tf.convert_to_tensor(np.asarray(genders))
        ages = tf.convert_to_tensor(np.array(ages).astype(np.float32))
        return imgs, (genders, ages, image_path)

    def __len__(self):
        return math.ceil(self.num / self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def read_txt(self, path):
        contx=[]
        with open(path, 'r') as f:
            next(f)
            contx = f.read().splitlines()
        return contx
    
    def calcutzone(self, offset, v, is_h=False): #not center crop
        if offset < 0:
            a1=0
            a2=v
            return a1, a2
        if is_h:
            a1=0
            a2=self.img_size_h
            return a1, a2
        else:
            if offset%2>0:
                a1=int((offset-1)/2)
                a2=int((offset-1)/2+1)+self.img_size_w
                return a1, a2
            else:
                a1=a2=int(offset/2)
                a2=a2+self.img_size_w
                return a1, a2

    def calresizezone(self, img, radio, v, is_h=False):
        if is_h:
            v = int(radio*v)
            img=cv2.resize(img, (v, self.img_size_h))
            return img
        else:
            v = int(radio*v)
            img=cv2.resize(img, (self.img_size_w, v))
            return img
    
    def fullpix(self, v, img_size):
        a1=a2=0
        if v<img_size:
            offset=img_size-v
            if offset%2>0:
                a1=int((offset-1)/2)
                a2=int((offset-1)/2)+1
            else:
                a1=a2=int(offset/2)
        return a1,a2

    def data_enhance(self, image, img_name):
        (h,w,_) = image.shape
        # print(h,w)
        if h > self.img_size_h or w > self.img_size_w:
            h_offset = h-self.img_size_h
            w_offset = w-self.img_size_w
            h_radio = self.img_size_h / h
            w_radio = self.img_size_w / w
            # print(h_offset, w_offset)
            if self.datamode=="resize" and (h_offset>0 or w_offset>0):
                if h_radio > w_radio:
                    image=self.calresizezone(image, w_radio, w, True)
                else:
                    image=self.calresizezone(image, h_radio, h)
            elif self.datamode=="crop":
                y1, y2 = self.calcutzone(h_offset, h, True)
                x1, x2 = self.calcutzone(w_offset, w)
                image = image[y1:y2, x1:x2]
        (h,w,_) = image.shape
        h1,h2 = self.fullpix(h, self.img_size_h)
        w1,w2 = self.fullpix(w, self.img_size_w)
        image = cv2.copyMakeBorder(image, h1,h2,w1,w2,cv2.BORDER_CONSTANT,value=[0,0,0])
        # if self.mode=="train":
        #     image = transforms(image=image)["image"]
        # cv2.imwrite("./pre/"+img_name, image)
        # exit(0)
        return image

if __name__=="__main__":
    file = open("../config/config.yaml", 'r', encoding="utf-8")
    cfg = yaml.safe_load(file)      
    train_gen = ImageSequence(cfg, "train")
    for epoch in range(1):
        train_gen.on_epoch_end()
        for batch_number, (x, y) in enumerate(train_gen):
            a=1
            # print(y[0],z)
