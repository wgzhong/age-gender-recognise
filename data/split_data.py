import pandas as pd
import scipy
import numpy as np
  
def read_txt(path):
    contx=[]
    with open(path, 'r') as f:
        next(f)
        contx = f.read().splitlines()
    return contx
 
def write_txt(path, contx):
    np.savetxt(path, contx, fmt='%s', delimiter=',')

if __name__ == "__main__":
    path = "./pa100k"
    labels = read_txt(path+"/labels.txt")
    images = read_txt(path+"/images.txt")
    np.random.seed(42)
    np.random.shuffle(images) 
    np.random.seed(42)
    np.random.shuffle(labels)
    train_labels = labels[:80000]
    train_images = images[:80000]
    val_labels = labels[80000:90000]
    val_images = images[80000:90000]
    test_labels = labels[90000:]
    test_images = images[90000:]
    print(len(train_labels),len(train_images),len(val_labels),len(val_images),len(test_labels),len(test_images))
    write_txt(path+"/train_label.txt", train_labels)
    write_txt(path+"/train_images_name.txt", train_images)
    write_txt(path+"/val_label.txt", val_labels)
    write_txt(path+"/val_images_name.txt", val_images)
    write_txt(path+"/test_label.txt", test_labels)
    write_txt(path+"/test_images_name.txt", test_images)

